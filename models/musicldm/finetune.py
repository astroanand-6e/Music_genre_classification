"""
MusicLDM VAE Fine-tuning / Zero-shot Evaluation on FMA-Small

Uses the VAE encoder from MusicLDM (ucsd-reach/musicldm, via diffusers)
as a feature extractor for genre classification.

Research question: Do latent representations optimised for music *generation*
also encode genre-discriminative structure?

Preprocessing pipeline:
    raw audio (16 kHz)
      -> log-mel spectrogram (n_fft=1024, hop=160, n_mels=64)
      -> normalise to [-1, 1]
      -> add channel dim          -> (1, 1, n_mels, T)
      -> VAE encoder              -> (1, C, H, W) latent
      -> adaptive avg-pool        -> fixed 512-d vector
      -> LogisticRegression / MLP classifier

Usage
-----
    python models/musicldm/finetune.py --mode zero_shot
    python models/musicldm/finetune.py --mode finetune --epochs 20
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
    build_dataloaders, AUDIO_DIR,
    save_run_results, checkpoint_path, figures_path,
    compute_per_class_f1, get_audio_path, load_waveform,
)

# __ CLI __

def parse_args():
    p = argparse.ArgumentParser(description="MusicLDM-VAE training / evaluation")
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=2)
    return p.parse_args()

# MusicLDM operates at 16 kHz, mel-spec with 64 mel bins
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 64
EMBED_DIM = 512  # after adaptive pooling
NUM_GENRES = 8

# __ Mel-spectrogram helper __

def audio_to_mel(waveform_np, sr=TARGET_SR):
    """Convert raw waveform to a log-mel spectrogram tensor suitable for
    the MusicLDM VAE (shape ``(1, 1, n_mels, T)``)."""
    mel = librosa.feature.melspectrogram(
        y=waveform_np, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalise to [-1, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8) * 2 - 1
    return torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)

# __ VAE loader __

def load_vae(device):
    from diffusers import AutoencoderKL, MusicLDMPipeline
    pipe = MusicLDMPipeline.from_pretrained(
        "ucsd-reach/musicldm", torch_dtype=torch.float32,
    )
    vae = pipe.vae.to(device)
    vae.eval()
    del pipe.unet, pipe.text_encoder, pipe.tokenizer
    return vae

# __ Feature extraction __

def vae_encode(vae, mel_tensor, device):
    """Encode a mel-spec tensor through the VAE encoder and adaptive-pool
    to a fixed-size embedding."""
    mel_tensor = mel_tensor.to(device)
    with torch.no_grad():
        latent = vae.encode(mel_tensor).latent_dist.mean  # (1, C, H, W)
    pooled = nn.functional.adaptive_avg_pool2d(latent, (8, 16))  # (1, C, 8, 16)
    return pooled.flatten(1).squeeze(0).cpu().numpy()  # (C*8*16,) ≈ 512 if C=4

# __ FMADataset preprocess_fn (for fine-tune mode) __

def make_musicldm_preprocess():
    def fn(waveform_np, sr):
        mel = audio_to_mel(waveform_np, sr)
        return {"mel": mel.squeeze(0)}  # (1, n_mels, T)
    return fn

# __ Classifier head __

class MusicLDMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)

# __ Zero-shot (linear probe) __

def run_zero_shot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: zero_shot | Model: MusicLDM-VAE | Device: {device}")

    vae = load_vae(device)

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    def extract(split_df, desc):
        feats, labs = [], []
        for track_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            path = get_audio_path(track_id)
            try:
                wav = load_waveform(path, TARGET_SR, args.clip_secs)
                mel = audio_to_mel(wav, TARGET_SR)
                emb = vae_encode(vae, mel, device)
                feats.append(emb)
                labs.append(row["label"])
            except Exception:
                feats.append(np.zeros(EMBED_DIM))
                labs.append(row["label"])
        return np.array(feats), np.array(labs)

    X_train, y_train = extract(train_df, "Extracting train")
    X_test, y_test = extract(test_df, "Extracting test")

    # infer actual dim from extraction
    actual_dim = X_train.shape[1]
    print(f"Embedding dim: {actual_dim}")

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
        config={"clip_secs": args.clip_secs, "target_sr": TARGET_SR,
                "embed_dim": actual_dim},
        per_class_f1=f1s,
    )

# __ Fine-tune __

def run_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: finetune | Model: MusicLDM-VAE | Device: {device}")

    vae = load_vae(device)
    # freeze decoder — only encoder gradients for fine-tuning
    for p in vae.decoder.parameters():
        p.requires_grad = False
    vae.encoder.requires_grad_(True)

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    preprocess_fn = make_musicldm_preprocess()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn, target_sr=TARGET_SR,
        clip_secs=args.clip_secs, batch_size=args.batch_size,
    )

    # determine embedding dim from a dummy forward pass
    dummy = torch.randn(1, 1, N_MELS, 100).to(device)
    with torch.no_grad():
        latent = vae.encode(dummy).latent_dist.mean
    pooled = nn.functional.adaptive_avg_pool2d(latent, (8, 16))
    embed_dim = pooled.flatten(1).shape[1]
    print(f"Embedding dim: {embed_dim}")

    classifier = MusicLDMClassifier(embed_dim, NUM_GENRES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": vae.encoder.parameters(), "lr": args.lr_backbone},
        {"params": classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-8,
    )

    best_val_acc = 0.0
    ckpt_file = checkpoint_path("musicldm", "vae")
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

            latent = vae.encode(mel).latent_dist.mean
            pooled = nn.functional.adaptive_avg_pool2d(latent, (8, 16))
            embeddings = pooled.flatten(1)

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
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  ", leave=False):
                mel = batch["mel"].to(device)
                labels = batch["labels"].to(device)
                latent = vae.encode(mel).latent_dist.mean
                pooled = nn.functional.adaptive_avg_pool2d(latent, (8, 16))
                embeddings = pooled.flatten(1)
                logits = classifier(embeddings)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(targets, preds)
        val_accs.append(val_acc)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "vae_encoder_state": vae.encoder.state_dict(),
                "classifier_state": classifier.state_dict(),
            }, ckpt_file)
            flag = "  <-- best"
        print(f"Epoch {epoch+1:02d} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}{flag}")

    # test
    ckpt = torch.load(ckpt_file, weights_only=True)
    vae.encoder.load_state_dict(ckpt["vae_encoder_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    vae.encoder.eval()
    classifier.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            mel = batch["mel"].to(device)
            labels = batch["labels"].to(device)
            latent = vae.encode(mel).latent_dist.mean
            pooled = nn.functional.adaptive_avg_pool2d(latent, (8, 16))
            embeddings = pooled.flatten(1)
            logits = classifier(embeddings)
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, args.epochs + 1), train_losses, marker="o")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(range(1, args.epochs + 1), val_accs, marker="o", color="green")
    axes[1].axhline(best_val_acc, color="red", ls="--", label=f"Best: {best_val_acc:.4f}")
    axes[1].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_path("musicldm_training_curves.png"), dpi=150)
    plt.close()

    save_run_results(
        model="musicldm", variant="vae", mode="finetune",
        test_accuracy=test_acc, best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
            "embed_dim": embed_dim,
        },
        per_class_f1=f1s,
        extra={"checkpoint": ckpt_file},
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
