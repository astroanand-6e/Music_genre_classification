"""
Microsoft CLAP Fine-tuning / Zero-shot Evaluation on FMA-Small or FMA-Medium

Uses the ``msclap`` package (pip install msclap) which loads weights from
the ``microsoft/msclap`` HuggingFace repo via its own loader (NOT compatible
with ``transformers.ClapModel``).

Usage
-----
    # Zero-shot (text-audio cosine similarity)
    python models/clap/finetune_microsoft.py --mode zero_shot
    python models/clap/finetune_microsoft.py --mode zero_shot --dataset medium

    # Fine-tune audio encoder + classification head
    python models/clap/finetune_microsoft.py --mode finetune --epochs 20
    python models/clap/finetune_microsoft.py --mode finetune --dataset medium --epochs 40 \\
        --early_stop_patience 5 --early_stop_min_delta 0.0001

Model
-----
    microsoft/msclap (version='2023')
    Audio sample rate : 44100
    Audio duration    : 7 s (model default)
    Audio embed dim   : 1024
    Text embed dim    : 1024
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    build_dataloaders,
    save_run_results, checkpoint_path, figures_path,
    compute_per_class_f1, load_waveform,
    resolve_audio_path,
    segment_offset_for_mode, tiled_segment_offsets,
    add_temporal_cli_args,
    add_finetune_callback_args,
    add_device_arg,
    resolve_training_device,
    new_finetune_run_id,
    finetune_epoch_csv_path,
    append_finetune_epoch_log,
    max_optimizer_lr,
    print_test_classification_report,
    save_genre_confusion_matrix_png,
)

# Microsoft CLAP 2023 operates at 44.1 kHz, 7-second clips
TARGET_SR = 44_100
CLIP_SECS = 7.0       # model's native duration
EMBED_DIM = 1024       # audio & text embedding dimension

# Genre descriptions for zero-shot text-audio similarity
# Covers both FMA-Small (8 genres) and FMA-Medium (16 genres) + Bollywood
GENRE_DESCRIPTIONS = {
    "Electronic":        "synthesized electronic dance music with digital drums and synthesizers",
    "Experimental":      "experimental avant-garde music with unconventional sounds and abstract structures",
    "Folk":              "acoustic folk music with traditional instruments storytelling and natural sounds",
    "Hip-Hop":           "hip hop rap music with heavy drum beats rhythmic vocals and bass lines",
    "Instrumental":      "instrumental music without vocals featuring live instruments and melodies",
    "International":     "international world music from diverse global cultural traditions",
    "Pop":               "catchy pop music with hooks melodic choruses and mainstream radio appeal",
    "Rock":              "rock music with electric guitars drum kits distortion and powerful vocals",
    "Bollywood":         "Bollywood Hindi film music with orchestral arrangements and Indian classical influences",
    "Blues":             "blues music with soulful vocals slide guitar and twelve-bar progressions",
    "Classical":         "western classical orchestral and chamber music with complex formal structure",
    "Country":           "country music with acoustic guitars pedal steel banjo and rural storytelling",
    "Jazz":              "jazz music with improvisation swing rhythm brass and piano syncopation",
    "Old-Time / Historic": "historical folk and old-time American fiddle and string band music",
    "Soul-RnB":          "soul and rhythm and blues music with emotional gospel-influenced vocals",
    "Spoken":            "spoken word poetry and non-musical speech recordings",
}

# __ CLI __

def parse_args():
    p = argparse.ArgumentParser(
        description="Microsoft CLAP fine-tuning / zero-shot evaluation"
    )
    add_temporal_cli_args(p)
    add_finetune_callback_args(p)
    add_device_arg(p)
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True,
                   help="zero_shot: frozen model + text-audio similarity; finetune: train audio encoder + head")
    p.add_argument("--epochs", type=int, default=20,
                   help="Maximum training epochs (early stopping may finish earlier)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_audio", type=float, default=5e-5,
                   help="Learning rate for the CLAP audio encoder parameters")
    p.add_argument("--lr_head", type=float, default=1e-3,
                   help="Learning rate for the classification head")
    p.add_argument("--clip_secs", type=float, default=CLIP_SECS,
                   help="Audio clip length in seconds fed to the model (default 7s = model native)")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--dataset", choices=["small", "medium"], default="small",
                   help="FMA subset to use")
    p.add_argument("--include_bollywood", action="store_true",
                   help="Merge Bollywood tracks as an extra genre class")
    p.add_argument("--train_segments", type=int, default=1, choices=[1, 2, 3],
                   help="Number of 7s segments to sample per track during training/val. "
                        "Embeddings are averaged before the classifier. "
                        "2 = ~14s of audio context, 3 = ~21s.")
    return p.parse_args()

# __ Model loading __

def load_model(device):
    """Load Microsoft CLAP via the msclap package."""
    from msclap import CLAP
    use_cuda = device.type == "cuda"
    print(f"Loading Microsoft CLAP (version=2023, cuda={use_cuda}) ...")
    clap = CLAP(version="2023", use_cuda=use_cuda)
    return clap


def _audio_embedding(clap, waveform_tensor):
    """Extract 1024-d audio embedding from a raw waveform tensor.

    Args:
        clap: msclap.CLAP instance
        waveform_tensor: (B, samples) float tensor at TARGET_SR
    Returns:
        (B, 1024) embedding tensor
    """
    # clap.clap.audio_encoder returns (embedding, class_probs)
    return clap.clap.audio_encoder(waveform_tensor)[0]


def _text_embedding(clap, texts):
    """Encode a list of text strings into (N, 1024) embedding tensor."""
    tokens = clap.preprocess_text(texts)
    return clap.clap.caption_encoder(tokens)


# __ Classifier head __

class MicrosoftCLAPClassifier(nn.Module):
    """Lightweight MLP classification head on top of 1024-d CLAP audio embeddings."""

    def __init__(self, audio_dim: int = EMBED_DIM, num_classes: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# __ Helpers __

def _make_preprocess_fn():
    """Return a no-op preprocess callable compatible with build_dataloaders."""
    def fn(waveform_np, sr):
        return {"waveform": torch.from_numpy(waveform_np).float()}
    return fn


def _pad_or_trim(wav, max_len):
    """Ensure waveform is exactly max_len samples."""
    if len(wav) >= max_len:
        return wav[:max_len]
    return np.pad(wav, (0, max_len - len(wav)))


def _multi_segment_embedding(audio_encoder, waveforms, n_segments):
    """Split each waveform into n_segments non-overlapping 7s chunks, encode each,
    and return the mean embedding.

    Args:
        audio_encoder: CLAP audio encoder module
        waveforms: (B, samples) tensor — clip_secs should be >= 7 * n_segments
        n_segments: number of 7s segments to extract and average
    Returns:
        (B, 1024) averaged embedding
    """
    if n_segments == 1:
        return audio_encoder(waveforms)[0]

    seg_len = int(TARGET_SR * CLIP_SECS)  # 7s worth of samples
    total_samples = waveforms.shape[1]
    # Compute evenly spaced start offsets
    if total_samples >= seg_len * n_segments:
        starts = [i * seg_len for i in range(n_segments)]
    else:
        # Overlap if waveform is shorter than n_segments * 7s
        stride = max(1, (total_samples - seg_len) // max(1, n_segments - 1))
        starts = [i * stride for i in range(n_segments)]

    emb_sum = None
    for s in starts:
        chunk = waveforms[:, s:s + seg_len]
        if chunk.shape[1] < seg_len:
            chunk = torch.nn.functional.pad(chunk, (0, seg_len - chunk.shape[1]))
        emb = audio_encoder(chunk)[0]  # (B, 1024)
        emb_sum = emb if emb_sum is None else emb_sum + emb

    return emb_sum / len(starts)


# __ Zero-shot evaluation __

def run_zero_shot(args):
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)
    print(f"[Microsoft CLAP] Mode: zero_shot | Dataset: {dataset_tag}")

    clap = load_model(device)

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    label2id, id2label = get_label_maps(df)
    _, _, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )

    # Only keep genres for which we have a description
    genre_list = sorted(g for g in id2label.values() if g in GENRE_DESCRIPTIONS)
    genre_to_class_idx = {g: label2id[g] for g in genre_list}

    # Encode all genre descriptions once
    with torch.no_grad():
        text_feats = _text_embedding(clap, [GENRE_DESCRIPTIONS[g] for g in genre_list])
        text_feats = text_feats / (text_feats.norm(dim=1, keepdim=True) + 1e-8)

    preds, targets = [], []
    failed = 0
    max_len = int(TARGET_SR * args.clip_secs)

    for track_id, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Zero-shot eval"):
        path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
        try:
            off = segment_offset_for_mode(
                path, track_id, args.clip_secs, "test", args.temporal_eval,
            )
            wav = load_waveform(path, TARGET_SR, args.clip_secs, offset=off)
            wav = _pad_or_trim(wav, max_len)

            wav_t = torch.from_numpy(wav).float().unsqueeze(0)
            if device.type == "cuda":
                wav_t = wav_t.cuda()

            with torch.no_grad():
                audio_feats = _audio_embedding(clap, wav_t)
                audio_feats = audio_feats / (audio_feats.norm(dim=1, keepdim=True) + 1e-8)

            sims = (audio_feats @ text_feats.T)[0].cpu().numpy()
            preds.append(genre_to_class_idx[genre_list[int(np.argmax(sims))]])
            targets.append(int(row["label"]))
        except Exception:
            failed += 1

    acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nTest Accuracy: {acc * 100:.2f}%  (tracks failed to load: {failed})")
    print_test_classification_report(targets, preds, id2label)

    save_run_results(
        model="clap",
        variant="microsoft",
        mode="zero_shot",
        test_accuracy=acc,
        config={
            "clip_secs": args.clip_secs,
            "target_sr": TARGET_SR,
            "embed_dim": EMBED_DIM,
            "msclap_version": "2023",
            "dataset": args.dataset,
            "include_bollywood": args.include_bollywood,
            "temporal_eval": args.temporal_eval,
        },
        per_class_f1=f1s,
        dataset=dataset_tag,
    )

# __ Fine-tuning __

def run_finetune(args):
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)
    print(f"[Microsoft CLAP] Mode: finetune | Dataset: {dataset_tag}")

    clap = load_model(device)
    audio_encoder = clap.clap.audio_encoder

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    label2id, id2label = get_label_maps(df)
    num_classes = len(id2label)

    classifier = MicrosoftCLAPClassifier(audio_dim=EMBED_DIM, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    train_df, val_df, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )
    max_len = int(TARGET_SR * args.clip_secs)
    preprocess_fn = _make_preprocess_fn()

    # Load longer clips when using multiple segments
    load_clip_secs = args.clip_secs * args.train_segments
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn,
        target_sr=TARGET_SR,
        clip_secs=load_clip_secs,
        batch_size=args.batch_size,
        audio_dir=audio_dir,
        train_temporal_sampling=args.temporal_train,
        eval_temporal_sampling=args.temporal_eval,
    )
    n_seg = args.train_segments
    effective_secs = args.clip_secs * n_seg if n_seg > 1 else args.clip_secs
    print(
        f"Temporal: train={args.temporal_train}, val/test={args.temporal_eval}; "
        f"test_multi_crop={args.test_multi_crop}; "
        f"train_segments={n_seg} ({effective_secs:.0f}s effective audio)"
    )
    print(f"Classes: {num_classes} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    run_id = new_finetune_run_id()
    epoch_log_csv = finetune_epoch_csv_path("clap", "microsoft", dataset_tag, run_id)
    print(f"Training log CSV -> {epoch_log_csv}  (run_id={run_id})")

    # Only train the audio encoder + classifier head (text encoder stays frozen)
    optimizer = torch.optim.AdamW(
        [
            {"params": audio_encoder.parameters(), "lr": args.lr_audio},
            {"params": classifier.parameters(), "lr": args.lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.epochs,
        eta_min=1e-7,
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    ckpt_file = checkpoint_path("clap", "microsoft", dataset=dataset_tag)
    train_losses, val_accs = [], []

    # __ Training loop __
    for epoch in range(args.epochs):
        audio_encoder.train()
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{args.epochs} [train]")
        for step, batch in enumerate(loop):
            waveforms = batch["waveform"].to(device)
            labels = batch["labels"].to(device)

            audio_feats = _multi_segment_embedding(audio_encoder, waveforms, n_seg)
            logits = classifier(audio_feats)
            loss = criterion(logits, labels) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * args.grad_accum
            loop.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # __ Validation __
        audio_encoder.eval()
        classifier.eval()
        preds, targets = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1:02d}/{args.epochs} [val]  ", leave=False):
                waveforms = batch["waveform"].to(device)
                labels = batch["labels"].to(device)
                audio_feats = _multi_segment_embedding(audio_encoder, waveforms, n_seg)
                logits = classifier(audio_feats)
                total_val_loss += criterion(logits, labels).item()
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(targets, preds)
        val_accs.append(val_acc)

        flag = ""
        if val_acc > best_val_acc + args.early_stop_min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "audio_encoder_state": audio_encoder.state_dict(),
                    "classifier_state": classifier.state_dict(),
                },
                ckpt_file,
            )
            flag = "  <-- best"
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch + 1:02d} | Train Loss {avg_loss:.4f} | "
            f"Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.4f}{flag}"
        )
        append_finetune_epoch_log(
            epoch_log_csv,
            {
                "run_id": run_id,
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "val_acc": round(val_acc, 6),
                "best_val_acc": round(best_val_acc, 6),
                "lr_max": round(max_optimizer_lr(optimizer), 10),
            },
        )

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping: no val acc improvement for {args.early_stop_patience} epoch(s).")
            break

    # __ Test evaluation (load best checkpoint) __
    ckpt = torch.load(ckpt_file, weights_only=True)
    audio_encoder.load_state_dict(ckpt["audio_encoder_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    audio_encoder.eval()
    classifier.eval()
    preds, targets = [], []

    with torch.no_grad():
        if args.test_multi_crop:
            for track_id, row in tqdm(
                test_df.iterrows(), total=len(test_df), desc="Testing (multi-crop)"
            ):
                path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
                try:
                    offs = tiled_segment_offsets(
                        path, args.clip_secs, args.test_crop_hop_frac, args.test_max_segments,
                    )
                    log_sum = None
                    for off in offs:
                        wav = load_waveform(path, TARGET_SR, args.clip_secs, offset=off)
                        wav = _pad_or_trim(wav, max_len)
                        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)
                        feats = audio_encoder(wav_t)[0]
                        lg = classifier(feats)
                        log_sum = lg if log_sum is None else log_sum + lg
                    pred = (log_sum / len(offs)).argmax(1).item()
                except Exception:
                    pred = 0
                preds.append(pred)
                targets.append(int(row["label"]))
        else:
            for batch in tqdm(test_loader, desc="Testing"):
                waveforms = batch["waveform"].to(device)
                labels = batch["labels"].to(device)
                audio_feats = _multi_segment_embedding(audio_encoder, waveforms, n_seg)
                logits = classifier(audio_feats)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nBest Val Acc : {best_val_acc:.4f}")
    print(f"Test Acc     : {test_acc:.4f}")
    print("\nTest classification report (per genre):")
    print_test_classification_report(targets, preds, id2label)

    # Confusion matrix
    cm_path = figures_path(f"confmat_clap_microsoft_{dataset_tag}_{run_id}.png")
    save_genre_confusion_matrix_png(
        targets, preds, id2label, cm_path,
        title=f"CLAP-Microsoft fine-tuned — {dataset_tag} (acc={test_acc * 100:.1f}%)",
    )

    # Training curves
    n_ep = len(train_losses)
    ep_axis = range(1, n_ep + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep_axis, train_losses, marker="o")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep_axis, val_accs, marker="o", color="green")
    axes[1].axhline(best_val_acc, color="red", ls="--", label=f"Best: {best_val_acc:.4f}")
    axes[1].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    curves_png = figures_path(f"clap_microsoft_{dataset_tag}_training_curves_{run_id}.png")
    plt.savefig(curves_png, dpi=150)
    plt.close()

    save_run_results(
        model="clap",
        variant="microsoft",
        mode="finetune",
        test_accuracy=test_acc,
        best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs,
            "target_sr": TARGET_SR,
            "embed_dim": EMBED_DIM,
            "msclap_version": "2023",
            "epochs": args.epochs,
            "epochs_run": len(train_losses),
            "batch_size": args.batch_size,
            "lr_audio": args.lr_audio,
            "lr_head": args.lr_head,
            "dataset": args.dataset,
            "include_bollywood": args.include_bollywood,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "run_id": run_id,
            "temporal_train": args.temporal_train,
            "temporal_eval": args.temporal_eval,
            "test_multi_crop": args.test_multi_crop,
            "test_crop_hop_frac": args.test_crop_hop_frac,
            "test_max_segments": args.test_max_segments,
            "train_segments": args.train_segments,
        },
        per_class_f1=f1s,
        extra={
            "checkpoint": ckpt_file,
            "training_log_csv": epoch_log_csv,
            "confusion_matrix_png": cm_path,
            "training_curves_png": curves_png,
        },
        dataset=dataset_tag,
    )

# __ Entry point __

def main():
    args = parse_args()
    if args.mode == "zero_shot":
        run_zero_shot(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    main()
