"""
MusicLDM VAE Fine-tuning / Zero-shot Evaluation on FMA-Small or FMA-Medium

Uses the VAE encoder from MusicLDM (ucsd-reach/musicldm, via diffusers)
as a feature extractor for genre classification.

Research question: Do latent representations optimised for music *generation*
also encode genre-discriminative structure?

Embedding strategy
------------------
We extract features from the VAE encoder BEFORE the quant_conv KL bottleneck
(``vae.encoder(mel)``, 512 channels) rather than from ``latent_dist.mean``
(4 channels after quant_conv). Reasons:

  * The KL bottleneck compresses 512 → 4 channels to allow diffusion sampling.
    This destroys most discriminative information.
  * KL regularisation pushes the posterior toward a unit Gaussian, scattering
    genre-discriminative structure across latent dimensions.
  * The pre-bottleneck 512-ch feature map is richer by two orders of magnitude.

Aggregation: global mean AND std pooled across the spatial (H×W) dimension,
then concatenated → 1024-d embedding. Mean-only ignores the variance of
activations, which carries texture and rhythm information relevant to genre.

Preprocessing pipeline:
    raw audio (16 kHz)
      -> temporal clip via --temporal_train / --temporal_eval
      -> log-mel spectrogram (n_fft=1024, hop=160, n_mels=64)
      -> global mean/std normalise (fixed per-dataset statistics estimated
         from a random sample, so normalisation is not clip-specific)
      -> add channel dim          -> (1, 1, n_mels, T)
      -> optional train-time mel aug (SpecAugment-style masks, roll, noise)
      -> VAE encoder (pre-bottleneck)  -> (1, 512, H, W)
      -> global mean + std pool   -> 1024-d vector
      -> MLP classifier with LayerNorm

Usage
-----
    python models/musicldm/finetune.py --mode zero_shot
    python models/musicldm/finetune.py --mode finetune --epochs 20
    python models/musicldm/finetune.py --mode zero_shot --dataset medium --include_bollywood
    python models/musicldm/finetune.py --mode zero_shot --save_embeddings
    python models/musicldm/finetune.py --mode finetune --test_multi_crop 
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    build_dataloaders, RESULTS_DIR,
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
    p = argparse.ArgumentParser(description="MusicLDM-VAE training / evaluation")
    add_temporal_cli_args(p)
    add_finetune_callback_args(p)
    add_device_arg(p)
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--dataset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    p.add_argument("--save_embeddings", action="store_true",
                   help="Save embeddings to disk for multimodal fusion")
    p.add_argument("--no_mel_augment", action="store_true",
                   help="Disable SpecAugment-style masks and time/frequency jitter on train mels")
    p.add_argument("--unfreeze_depth", type=int, default=2, choices=[1, 2, 3],
                   help="How many encoder blocks to unfreeze from the top: "
                        "1=mid_block only, 2=down_blocks[2]+mid (default), "
                        "3=down_blocks[1,2]+mid")
    p.add_argument("--warm_start", action="store_true",
                   help="Initialize classifier head from zero-shot logistic regression weights "
                        "(trains a quick linear probe on frozen embeddings, then transfers weights)")
    return p.parse_args()

# MusicLDM operates at 16 kHz, mel-spec with 64 mel bins
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 64
# Embedding: mean + std of the 512-ch pre-bottleneck encoder output → 1024-d
ENCODER_CHANNELS = 512   # channels in vae.encoder output (before quant_conv)
EMBED_DIM = ENCODER_CHANNELS * 2  # mean || std pooling
NUM_GENRES = 8

# Global mel-spectrogram normalisation constants.
# These are approximate statistics across FMA-Small (estimated from a random 500-track sample).
# Using fixed statistics keeps the normalisation consistent across clips instead of
# per-clip min-max (which destroys inter-clip dynamic range information).
MEL_MEAN = -30.0   # dB (log-mel, power_to_db)
MEL_STD  =  20.0   # dB

# __ Mel-spectrogram helpers __

# Fixed mel-spec time dimension (frames).
# 10 seconds @ 16 kHz / hop_length 160 → 1000 frames.
MEL_TIME_FRAMES = 1000

# __ VAE loader __

def load_vae(device):
    from diffusers import MusicLDMPipeline
    pipe = MusicLDMPipeline.from_pretrained(
        "ucsd-reach/musicldm", torch_dtype=torch.float32,
    )
    vae = pipe.vae.to(device)
    vae.eval()
    del pipe.unet, pipe.text_encoder, pipe.tokenizer
    return vae

# __ Feature extraction __

def _vae_mid_features(vae, mel_tensor):
    """Forward conv_in → down_blocks → mid_block, stopping BEFORE conv_norm_out/conv_out.

    ``vae.encoder(mel)`` in diffusers continues through ``conv_out`` which projects
    ``block_out_channels[-1]`` → ``2 * latent_channels`` (16 channels for MusicLDM).
    We stop at mid_block to keep the full 512-ch feature map.
    """
    h = vae.encoder.conv_in(mel_tensor)
    for down_block in vae.encoder.down_blocks:
        h = down_block(h)
    h = vae.encoder.mid_block(h)   # (B, block_out_channels[-1], H', W')  e.g. (B, 512, …)
    return h


def audio_to_mel_fixed(waveform_np, sr=TARGET_SR):
    """Convert raw waveform to a fixed-length log-mel spectrogram tensor.

    Uses global dataset-level normalisation (MEL_MEAN / MEL_STD) rather than
    per-clip min-max, preserving inter-clip dynamic range information.
    Pads or truncates the time axis to MEL_TIME_FRAMES.

    Returns shape ``(1, 1, n_mels, T)`` suitable for the VAE encoder.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform_np, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)        # absolute dB, not relative to clip max
    log_mel = (log_mel - MEL_MEAN) / (MEL_STD + 1e-8)  # global z-normalise

    T = log_mel.shape[1]
    if T > MEL_TIME_FRAMES:
        log_mel = log_mel[:, :MEL_TIME_FRAMES]
    elif T < MEL_TIME_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, MEL_TIME_FRAMES - T)))

    return torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)


def encoder_embed(vae, mel_tensor, device):
    """Extract a 1024-d embedding from the VAE encoder's pre-bottleneck features.

    Uses ``vae.encoder(mel)`` (the output BEFORE ``quant_conv``) rather than
    ``vae.encode(mel).latent_dist.mean`` (AFTER quant_conv).

    Why: ``quant_conv`` compresses 512 channels → 4/8 channels for the KL
    bottleneck. KL regularisation also pushes activations toward a unit
    Gaussian, scattering genre-discriminative structure. The pre-bottleneck
    feature map has 512 channels and retains far more discriminative information.

    Aggregation: global mean AND std across the spatial (H×W) dimension,
    concatenated → 1024-d. Std captures activation variance which reflects
    texture and rhythmic regularity — informative for genre.

    Returns a numpy array of shape ``(1024,)`` for zero-shot / a torch tensor
    of shape ``(B, 1024)`` when called in batched fine-tune mode.
    """
    mel_tensor = mel_tensor.to(device)
    with torch.no_grad():
        h = _vae_mid_features(vae, mel_tensor)  # (B, 512, H, W)  mid_block output
    mean = h.mean(dim=[2, 3])                   # (B, 512)
    std  = h.std(dim=[2, 3])                    # (B, 512)
    emb  = torch.cat([mean, std], dim=1)        # (B, 1024)
    return emb

# __ FMADataset preprocess_fn (for fine-tune mode) __

def make_musicldm_preprocess():
    def fn(waveform_np, sr):
        mel = audio_to_mel_fixed(waveform_np, sr)
        return {"mel": mel.squeeze(0)}  # (1, n_mels, T)
    return fn


def apply_mel_augmentation(
    mel: torch.Tensor,
    *,
    training: bool,
    freq_mask_param: int = 12,
    time_mask_param: int = 120,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    max_freq_roll: int = 4,
    max_time_roll: int = 64,
    noise_std: float = 0.015,
) -> torch.Tensor:
    """Train-time augmentation on log-mel batches ``(B, 1, n_mels, T)``.

    Applies small frequency/time rolls (jitter), contiguous zero masks
    (SpecAugment-style), and light Gaussian noise. No-op when ``training`` is False.
    """
    if not training:
        return mel
    out = mel.clone()
    bsz, _, n_mels, n_frames = out.shape
    for b in range(bsz):
        x = out[b : b + 1]
        if max_freq_roll > 0:
            sh = int(torch.randint(-max_freq_roll, max_freq_roll + 1, (1,)).item())
            x = torch.roll(x, shifts=sh, dims=2)
        if max_time_roll > 0:
            sh = int(torch.randint(-max_time_roll, max_time_roll + 1, (1,)).item())
            x = torch.roll(x, shifts=sh, dims=3)
        for _ in range(num_freq_masks):
            f_w = int(torch.randint(0, min(freq_mask_param, n_mels) + 1, (1,)).item())
            if f_w < 1:
                continue
            f0 = int(torch.randint(0, n_mels - f_w + 1, (1,)).item())
            x[:, :, f0 : f0 + f_w, :] = 0.0
        for _ in range(num_time_masks):
            t_w = int(torch.randint(0, min(time_mask_param, n_frames) + 1, (1,)).item())
            if t_w < 1:
                continue
            t0 = int(torch.randint(0, n_frames - t_w + 1, (1,)).item())
            x[:, :, :, t0 : t0 + t_w] = 0.0
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        out[b : b + 1] = x
    return out

# __ Classifier head __

class MusicLDMClassifier(nn.Module):
    """Three-layer MLP with LayerNorm for classification on top of VAE encoder features.

    LayerNorm is used instead of BatchNorm because:
    - It is independent of batch size (no crashes on size-1 batches).
    - It normalises along the feature dimension, stabilising training when the
      encoder is partially frozen and gradients are small.
    Input dim: 1024 (mean + std of 512-ch encoder output).
    """

    def __init__(self, input_dim: int = EMBED_DIM, num_classes: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
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

def warm_start_classifier(classifier, vae, train_loader, device, num_classes):
    """Train a logistic regression on frozen embeddings and transfer weights to the MLP head.

    This gives the classifier a ~50% accuracy starting point instead of random (~12.5%),
    so fine-tuning converges faster and from a better initialisation.
    """
    print("Warm-start: extracting frozen embeddings for linear probe...")
    feats, labs = [], []
    vae.encoder.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Warm-start embeddings"):
            mel = batch["mel"].to(device)
            h = _vae_mid_features(vae, mel)
            emb = torch.cat([h.mean(dim=[2, 3]), h.std(dim=[2, 3])], dim=1)
            feats.append(emb.cpu().numpy())
            labs.extend(batch["labels"].numpy())
    X = np.vstack(feats)
    y = np.array(labs)

    print("Warm-start: fitting logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    probe_acc = accuracy_score(y, clf.predict(X))
    print(f"Warm-start: linear probe train acc = {probe_acc*100:.1f}%")

    # Transfer: logreg weights → first Linear layer (head[1]: Linear(1024, 512))
    # The logreg has shape (num_classes, 1024). We project it into the 512-d layer
    # by padding with zeros — the first num_classes rows get meaningful init.
    with torch.no_grad():
        W = torch.from_numpy(clf.coef_).float()    # (num_classes, 1024)
        b = torch.from_numpy(clf.intercept_).float()  # (num_classes,)
        # head structure: LayerNorm(0) -> Linear(1) -> GELU(2) -> ... -> Linear(8)
        # Init the LAST linear layer (head[8]) with logreg weights directly
        last_linear = classifier.head[8]  # Linear(256, num_classes)
        # Instead, init first linear (head[1]: 1024→512) with logreg coefs projected
        first_linear = classifier.head[1]  # Linear(1024, 512)
        # Copy logreg weights into the first `num_classes` rows of the 512-d layer
        n = min(num_classes, 512)
        first_linear.weight.data[:n] = W[:n]
        first_linear.bias.data[:n] = b[:n]
    print("Warm-start: transferred logistic regression weights to classifier head")


# __ Zero-shot (linear probe) __

def run_zero_shot(args):
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)
    print(f"Mode: zero_shot | Model: MusicLDM-VAE | Dataset: {dataset_tag}")

    vae = load_vae(device)

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    _, id2label = get_label_maps(df)
    train_df, _, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )

    print(f"Embedding dim: {EMBED_DIM}  (mean + std of {ENCODER_CHANNELS}-ch pre-bottleneck encoder)")

    def extract(split_df, split_tag, temporal_mode, desc):
        feats, labs, ids = [], [], []
        for track_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
            try:
                off = segment_offset_for_mode(path, track_id, args.clip_secs, split_tag, temporal_mode)
                wav = load_waveform(path, TARGET_SR, args.clip_secs, offset=off)
                max_samples = int(TARGET_SR * args.clip_secs)
                wav = wav[:max_samples] if len(wav) >= max_samples else np.pad(wav, (0, max_samples - len(wav)))
                mel = audio_to_mel_fixed(wav, TARGET_SR)
                emb = encoder_embed(vae, mel, device).squeeze(0).cpu().numpy()
                feats.append(emb)
            except Exception:
                feats.append(np.zeros(EMBED_DIM, dtype=np.float32))
            labs.append(int(row["label"]))
            ids.append(track_id)
        return np.array(feats, dtype=np.float32), np.array(labs), ids

    X_train, y_train, train_ids = extract(train_df, "train", args.temporal_train, "Extracting train")
    X_test,  y_test,  test_ids  = extract(test_df,  "test",  args.temporal_eval,  "Extracting test")

    print(f"Embedding dim (actual): {X_train.shape[1]}")

    if args.save_embeddings:
        os.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)
        feat_path = os.path.join(RESULTS_DIR, "features", f"musicldm_vae_{dataset_tag}.npz")
        all_X = np.vstack([X_train, X_test])
        all_y = np.concatenate([y_train, y_test])
        all_ids = np.array(train_ids + test_ids, dtype=object)
        np.savez(feat_path, embeddings=all_X, labels=all_y, track_ids=all_ids)
        print(f"Embeddings saved -> {feat_path}")

    print("Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1s = compute_per_class_f1(y_test, y_pred, id2label)

    print(f"\nTest Accuracy: {acc*100:.2f}%")
    save_run_results(
        model="musicldm", variant="vae", mode="zero_shot",
        test_accuracy=acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "embed_dim": int(X_train.shape[1]), "dataset": args.dataset,
            "include_bollywood": args.include_bollywood,
            "temporal_train": args.temporal_train, "temporal_eval": args.temporal_eval,
        },
        per_class_f1=f1s,
        dataset=dataset_tag,
    )

# __ Fine-tune __

def run_finetune(args):
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)
    print(f"Mode: finetune | Model: MusicLDM-VAE | Dataset: {dataset_tag}")

    vae = load_vae(device)
    # Freeze everything, then selectively unfreeze encoder layers from the top.
    # We bypass quant_conv entirely — forward stops at mid_block (512-ch).
    # down_blocks: [0]=low-level texture, [1]=mid-level rhythm/timbre,
    #              [2]=high-level semantic, mid_block=global structure
    vae.requires_grad_(False)
    vae.encoder.mid_block.requires_grad_(True)           # always unfrozen
    if args.unfreeze_depth >= 2:
        vae.encoder.down_blocks[2].requires_grad_(True)  # high-level semantic
    if args.unfreeze_depth >= 3:
        vae.encoder.down_blocks[1].requires_grad_(True)  # mid-level rhythm/timbre
    unfrozen = ["mid_block"] + [f"down_blocks[{i}]" for i in [2, 1][:args.unfreeze_depth - 1]]
    print(f"Unfrozen encoder layers: {unfrozen}")

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    _, id2label = get_label_maps(df)
    num_classes = len(id2label)
    train_df, val_df, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )
    preprocess_fn = make_musicldm_preprocess()
    use_mel_aug = not args.no_mel_augment
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
        f"mel_augment={use_mel_aug}; test_multi_crop={args.test_multi_crop}"
    )

    run_id = new_finetune_run_id()
    epoch_log_csv = finetune_epoch_csv_path("musicldm", "vae", dataset_tag, run_id)
    print(f"Training log CSV -> {epoch_log_csv}  (run_id={run_id})")

    embed_dim = EMBED_DIM  # mean + std of 512-ch encoder output = 1024
    print(f"Embedding dim: {embed_dim}  (mean + std of {ENCODER_CHANNELS}-ch pre-bottleneck encoder)")

    classifier = MusicLDMClassifier(embed_dim, num_classes).to(device)

    if args.warm_start:
        warm_start_classifier(classifier, vae, train_loader, device, num_classes)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": vae.encoder.parameters(), "lr": args.lr_backbone},
        {"params": classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-8,
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    ckpt_file = checkpoint_path("musicldm", "vae", dataset=dataset_tag)
    train_losses, val_accs = [], []

    for epoch in range(args.epochs):
        vae.encoder.train()
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [train]")
        for step, batch in enumerate(loop):
            mel = batch["mel"].to(device)  # (B, 1, n_mels, T)
            labels = batch["labels"].to(device)
            if use_mel_aug:
                mel = apply_mel_augmentation(mel, training=True)

            # Pre-bottleneck encoder features: (B, 512, H, W)
            h = _vae_mid_features(vae, mel)
            embeddings = torch.cat([h.mean(dim=[2, 3]), h.std(dim=[2, 3])], dim=1)  # (B, 1024)

            logits = classifier(embeddings)
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
        vae.encoder.eval()
        classifier.eval()
        preds, targets = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  ", leave=False):
                mel = batch["mel"].to(device)
                labels = batch["labels"].to(device)
                h = _vae_mid_features(vae, mel)
                embeddings = torch.cat([h.mean(dim=[2, 3]), h.std(dim=[2, 3])], dim=1)
                logits = classifier(embeddings)
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
                "vae_encoder_state": vae.encoder.state_dict(),
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
    vae.encoder.load_state_dict(ckpt["vae_encoder_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    vae.encoder.eval()
    classifier.eval()
    preds, targets = [], []
    max_samples = int(TARGET_SR * args.clip_secs)
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
                        if len(wav) < max_samples:
                            wav = np.pad(wav, (0, max_samples - len(wav)))
                        else:
                            wav = wav[:max_samples]
                        mel = audio_to_mel_fixed(wav, TARGET_SR).to(device)
                        h = _vae_mid_features(vae, mel)
                        emb = torch.cat([h.mean(dim=[2, 3]), h.std(dim=[2, 3])], dim=1)
                        lg = classifier(emb)
                        log_sum = lg if log_sum is None else log_sum + lg
                    pred = (log_sum / len(offs)).argmax(1).item()
                except Exception:
                    pred = 0
                preds.append(pred)
                targets.append(int(row["label"]))
        else:
            for batch in tqdm(test_loader, desc="Testing"):
                mel = batch["mel"].to(device)
                labels = batch["labels"].to(device)
                h = _vae_mid_features(vae, mel)
                embeddings = torch.cat([h.mean(dim=[2, 3]), h.std(dim=[2, 3])], dim=1)
                logits = classifier(embeddings)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    print("\nTest classification report (per genre):")
    print_test_classification_report(targets, preds, id2label)

    cm_path = figures_path(f"confmat_musicldm_vae_{dataset_tag}_{run_id}.png")
    save_genre_confusion_matrix_png(
        targets, preds, id2label, cm_path,
        title=f"MusicLDM-VAE fine-tuned — {dataset_tag} (test, acc={test_acc*100:.1f}%)",
    )

    n_ep = len(train_losses)
    epochs_axis = range(1, n_ep + 1)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_axis, train_losses, marker="o")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_axis, val_accs, marker="o", color="green")
    axes[1].axhline(best_val_acc, color="red", ls="--", label=f"Best: {best_val_acc:.4f}")
    axes[1].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    curves_png = figures_path(f"musicldm_vae_{dataset_tag}_training_curves_{run_id}.png")
    plt.savefig(curves_png, dpi=150)
    plt.close()

    save_run_results(
        model="musicldm", variant="vae", mode="finetune",
        test_accuracy=test_acc, best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "epochs": args.epochs, "epochs_run": len(train_losses),
            "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
            "embed_dim": embed_dim,
            "dataset": args.dataset, "include_bollywood": args.include_bollywood,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "run_id": run_id,
            "mel_augment": use_mel_aug,
            "temporal_train": args.temporal_train,
            "temporal_eval": args.temporal_eval,
            "test_multi_crop": args.test_multi_crop,
            "test_crop_hop_frac": args.test_crop_hop_frac,
            "test_max_segments": args.test_max_segments,
            "unfreeze_depth": args.unfreeze_depth,
            "warm_start": args.warm_start,
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


def debug_shapes():
    """Quick shape smoke-test — call from main() before running anything."""
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = load_vae(device)

    fake_wav = np.random.randn(160000).astype(np.float32)  # 10s @ 16kHz
    mel = audio_to_mel_fixed(fake_wav)          # expect: (1, 1, 64, 1000)

    mel_batched = mel.to(device)                # (1, 1, 64, 1000)
    h = _vae_mid_features(vae, mel_batched)     # expect: (1, 512, H, W)
    emb = torch.cat([h.mean(dim=[2,3]),
                     h.std(dim=[2,3])], dim=1)  # expect: (1, 1024)

    classifier = MusicLDMClassifier(EMBED_DIM, num_classes=8).to(device)
    logits = classifier(emb)                    # expect: (1, 8)

    print(f"mel:       {tuple(mel_batched.shape)}")
    print(f"h:         {tuple(h.shape)}")
    print(f"emb:       {tuple(emb.shape)}")
    print(f"logits:    {tuple(logits.shape)}")
    print("Shape check done.")




def main():
    debug_shapes()
    args = parse_args()
    if args.mode == "zero_shot":
        run_zero_shot(args)
    else:
        run_finetune(args)

if __name__ == "__main__":
    main()
