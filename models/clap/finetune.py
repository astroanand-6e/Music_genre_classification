"""
CLAP Fine-tuning / Zero-shot Evaluation on FMA-Small or FMA-Medium

Usage
-----
    # Zero-shot (text-audio similarity)
    python models/clap/finetune.py --mode zero_shot --variant laion

    # Fine-tune audio encoder + classification head
    python models/clap/finetune.py --mode finetune --variant laion --epochs 20
    python models/clap/finetune.py --mode finetune --variant microsoft --epochs 20

    # FMA-Medium + Bollywood
    python models/clap/finetune.py --mode finetune --variant laion \\
        --dataset medium --include_bollywood --epochs 20
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    build_dataloaders, AUDIO_DIR,
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

# __ CLI __

def parse_args():
    p = argparse.ArgumentParser(description="CLAP training / evaluation")
    add_temporal_cli_args(p)
    add_finetune_callback_args(p)
    add_device_arg(p)
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--variant", choices=["laion", "microsoft"], default="laion")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_audio", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--dataset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    return p.parse_args()

# CLAP operates at 48 kHz
TARGET_SR = 48000

# Extended genre descriptions including Bollywood
GENRE_DESCRIPTIONS = {
    "Electronic": "synthesized electronic dance music with digital drums and synths",
    "Experimental": "experimental avant-garde music with unconventional sounds and structures",
    "Folk": "acoustic folk music with traditional instruments and storytelling lyrics",
    "Hip-Hop": "hip hop rap music with heavy beats and rhythmic vocals",
    "Instrumental": "instrumental music without vocals featuring live instruments",
    "International": "international world music from diverse cultural traditions",
    "Pop": "popular catchy pop music with hooks and mainstream radio appeal",
    "Rock": "rock music with electric guitars drums and intense vocals",
    "Bollywood": "Bollywood Hindi film music with orchestral arrangements and Indian classical influences",
    "Blues": "blues music with soulful vocals and guitar",
    "Classical": "western classical orchestral music",
    "Country": "country music with guitars and storytelling",
    "Jazz": "jazz music with improvisation and syncopation",
    "Old-Time / Historic": "historical folk and old-time American music",
    "Soul-RnB": "soul and R&B music with emotional vocals",
    "Spoken": "spoken word and poetry recordings",
}

# __ Model loading helpers __

def load_laion_clap(device):
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
    return processor, model


def load_microsoft_clap(device):
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained("microsoft/msclap")
    model = ClapModel.from_pretrained("microsoft/msclap").to(device)
    return processor, model

# __ Zero-shot __

def run_zero_shot(args):
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)
    print(f"Mode: zero_shot | Variant: {args.variant} | Dataset: {dataset_tag}")

    if args.variant == "laion":
        processor, model = load_laion_clap(device)
    else:
        processor, model = load_microsoft_clap(device)
    model.eval()

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    label2id, id2label = get_label_maps(df)
    _, _, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )

    # Use only the genres present in this dataset's descriptions
    genre_list = sorted(g for g in id2label.values() if g in GENRE_DESCRIPTIONS)
    genre_to_idx = {g: label2id[g] for g in genre_list}

    # encode text descriptions once
    with torch.no_grad():
        text_inputs = processor(
            text=[GENRE_DESCRIPTIONS[g] for g in genre_list],
            return_tensors="pt", padding=True,
        ).to(device)
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / (text_feats.norm(dim=1, keepdim=True) + 1e-8)

    preds, targets = [], []
    failed = 0
    max_len = int(TARGET_SR * args.clip_secs)

    for track_id, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Zero-shot"):
        path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
        try:
            off = segment_offset_for_mode(
                path, track_id, args.clip_secs, "test", args.temporal_eval,
            )
            wav = load_waveform(path, TARGET_SR, args.clip_secs, offset=off)
            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))

            with torch.no_grad():
                audio_in = processor(
                    audios=wav, sampling_rate=TARGET_SR,
                    return_tensors="pt",
                ).to(device)
                audio_out = model.get_audio_features(**audio_in)
                audio_feats = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out
                audio_feats = audio_feats / (audio_feats.norm(dim=1, keepdim=True) + 1e-8)

            sims = (audio_feats @ text_feats.T)[0].cpu().numpy()
            preds.append(genre_to_idx[genre_list[np.argmax(sims)]])
            targets.append(row["label"])
        except Exception:
            failed += 1

    acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nTest Accuracy: {acc*100:.2f}%  (failed: {failed})")
    save_run_results(
        model="clap", variant=args.variant, mode="zero_shot",
        test_accuracy=acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "dataset": args.dataset, "include_bollywood": args.include_bollywood,
            "temporal_eval": args.temporal_eval,
        },
        per_class_f1=f1s,
        dataset=dataset_tag,
    )

# __ Classifier head __

class CLAPClassifier(nn.Module):
    def __init__(self, audio_dim=512, num_classes=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)

# __ CLAP-specific preprocessing __

def make_clap_preprocess(max_len):
    def fn(waveform_np, sr):
        return {"waveform": torch.from_numpy(waveform_np).float()}
    return fn

# __ Fine-tune __

def run_finetune(args):
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)
    print(f"Mode: finetune | Variant: {args.variant} | Dataset: {dataset_tag}")

    if args.variant == "laion":
        processor, clap_model = load_laion_clap(device)
    else:
        processor, clap_model = load_microsoft_clap(device)

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    label2id, id2label = get_label_maps(df)
    num_classes = len(id2label)

    classifier = CLAPClassifier(audio_dim=512, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    train_df, val_df, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )
    max_len = int(TARGET_SR * args.clip_secs)
    preprocess_fn = make_clap_preprocess(max_len)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn, target_sr=TARGET_SR,
        clip_secs=args.clip_secs, batch_size=args.batch_size,
        audio_dir=audio_dir,
        train_temporal_sampling=args.temporal_train,
        eval_temporal_sampling=args.temporal_eval,
    )
    print(
        f"Temporal: train={args.temporal_train}, val/test={args.temporal_eval}; "
        f"test_multi_crop={args.test_multi_crop}"
    )

    run_id = new_finetune_run_id()
    epoch_log_csv = finetune_epoch_csv_path("clap", args.variant, dataset_tag, run_id)
    print(f"Training log CSV -> {epoch_log_csv}  (run_id={run_id})")

    optimizer = torch.optim.AdamW([
        {"params": clap_model.audio_model.parameters(), "lr": args.lr_audio},
        {"params": classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-7,
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    ckpt_file = checkpoint_path("clap", args.variant, dataset=dataset_tag)
    train_losses, val_accs = [], []

    def _freeze_bn(module):
        """Keep BatchNorm in eval mode to avoid size-1 batch crashes in CLAP's fusion model."""
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

    for epoch in range(args.epochs):
        clap_model.train()
        _freeze_bn(clap_model)
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [train]")
        for step, batch in enumerate(loop):
            waveforms = batch["waveform"].to(device)
            labels = batch["labels"].to(device)

            audio_inputs = processor(
                audios=waveforms.cpu().numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            ).to(device)
            audio_out = clap_model.get_audio_features(**audio_inputs)
            audio_feats = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out

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

        # validation
        clap_model.eval()
        classifier.eval()
        preds, targets = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  ", leave=False):
                waveforms = batch["waveform"].to(device)
                labels = batch["labels"].to(device)
                audio_inputs = processor(
                    audios=waveforms.cpu().numpy(),
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                ).to(device)
                audio_out = clap_model.get_audio_features(**audio_inputs)
                audio_feats = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out
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
            torch.save({
                "clap_state": clap_model.state_dict(),
                "classifier_state": classifier.state_dict(),
            }, ckpt_file)
            flag = "  <-- best"
        else:
            epochs_no_improve += 1
        print(f"Epoch {epoch+1:02d} | Train Loss {avg_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.4f}{flag}")

        append_finetune_epoch_log(epoch_log_csv, {
            "run_id": run_id,
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "val_acc": round(val_acc, 6),
            "best_val_acc": round(best_val_acc, 6),
            "lr_max": round(max_optimizer_lr(optimizer), 10),
        })

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping: no val acc improvement for {args.early_stop_patience} epoch(s).")
            break

    # test
    ckpt = torch.load(ckpt_file, weights_only=True)
    clap_model.load_state_dict(ckpt["clap_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    clap_model.eval()
    classifier.eval()
    preds, targets = [], []
    max_len = int(TARGET_SR * args.clip_secs)
    with torch.no_grad():
        if args.test_multi_crop:
            for track_id, row in tqdm(
                test_df.iterrows(), total=len(test_df), desc="Testing (multi-crop)",
            ):
                path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
                try:
                    offs = tiled_segment_offsets(
                        path, args.clip_secs,
                        args.test_crop_hop_frac, args.test_max_segments,
                    )
                    log_sum = None
                    for off in offs:
                        wav = load_waveform(
                            path, TARGET_SR, args.clip_secs, offset=off,
                        )
                        if len(wav) < max_len:
                            wav = np.pad(wav, (0, max_len - len(wav)))
                        else:
                            wav = wav[:max_len]
                        audio_inputs = processor(
                            audios=wav, sampling_rate=TARGET_SR,
                            return_tensors="pt",
                        ).to(device)
                        audio_out = clap_model.get_audio_features(**audio_inputs)
                        audio_feats = (
                            audio_out.pooler_output
                            if hasattr(audio_out, "pooler_output")
                            else audio_out
                        )
                        lg = classifier(audio_feats)
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
                audio_inputs = processor(
                    audios=waveforms.cpu().numpy(),
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                ).to(device)
                audio_out = clap_model.get_audio_features(**audio_inputs)
                audio_feats = (
                    audio_out.pooler_output
                    if hasattr(audio_out, "pooler_output")
                    else audio_out
                )
                logits = classifier(audio_feats)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    print("\nTest classification report (per genre):")
    print_test_classification_report(targets, preds, id2label)

    cm_path = figures_path(f"confmat_clap_{args.variant}_{dataset_tag}_{run_id}.png")
    save_genre_confusion_matrix_png(
        targets, preds, id2label, cm_path,
        title=f"CLAP-{args.variant} fine-tuned — {dataset_tag} (test, acc={test_acc*100:.1f}%)",
    )

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
    curves_png = figures_path(f"clap_{args.variant}_{dataset_tag}_training_curves_{run_id}.png")
    plt.savefig(curves_png, dpi=150)
    plt.close()

    save_run_results(
        model="clap", variant=args.variant, mode="finetune",
        test_accuracy=test_acc, best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "epochs": args.epochs, "epochs_run": len(train_losses),
            "batch_size": args.batch_size,
            "lr_audio": args.lr_audio, "lr_head": args.lr_head,
            "dataset": args.dataset, "include_bollywood": args.include_bollywood,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "run_id": run_id,
            "temporal_train": args.temporal_train,
            "temporal_eval": args.temporal_eval,
            "test_multi_crop": args.test_multi_crop,
            "test_crop_hop_frac": args.test_crop_hop_frac,
            "test_max_segments": args.test_max_segments,
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
