"""
CLAP Fine-tuning / Zero-shot Evaluation on FMA-Small

Usage
-----
    # Zero-shot (text-audio similarity)
    python models/clap/finetune.py --mode zero_shot --variant laion

    # Fine-tune audio encoder + classification head
    python models/clap/finetune.py --mode finetune --variant laion --epochs 20
    python models/clap/finetune.py --mode finetune --variant microsoft --epochs 20
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
    compute_per_class_f1, get_audio_path, load_waveform,
)

# __ CLI __

def parse_args():
    p = argparse.ArgumentParser(description="CLAP training / evaluation")
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--variant", choices=["laion", "microsoft"], default="laion")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_audio", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=2)
    return p.parse_args()

# CLAP operates at 48 kHz
TARGET_SR = 48000
NUM_GENRES = 8

GENRE_DESCRIPTIONS = {
    "Electronic": "synthesized electronic dance music with digital drums and synths",
    "Experimental": "experimental avant-garde music with unconventional sounds and structures",
    "Folk": "acoustic folk music with traditional instruments and storytelling lyrics",
    "Hip-Hop": "hip hop rap music with heavy beats and rhythmic vocals",
    "Instrumental": "instrumental music without vocals featuring live instruments",
    "International": "international world music from diverse cultural traditions",
    "Pop": "popular catchy pop music with hooks and mainstream radio appeal",
    "Rock": "rock music with electric guitars drums and intense vocals",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: zero_shot | Variant: {args.variant} | Device: {device}")

    if args.variant == "laion":
        processor, model = load_laion_clap(device)
    else:
        processor, model = load_microsoft_clap(device)
    model.eval()

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    _, _, test_df = get_splits(df)

    genre_list = sorted(GENRE_DESCRIPTIONS.keys())

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
        path = get_audio_path(track_id)
        try:
            wav = load_waveform(path, TARGET_SR, args.clip_secs)
            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))

            with torch.no_grad():
                audio_in = processor(
                    audio=wav, sampling_rate=TARGET_SR,
                    return_tensors="pt",
                ).to(device)
                audio_out = model.get_audio_features(**audio_in)
                audio_feats = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out
                audio_feats = audio_feats / (audio_feats.norm(dim=1, keepdim=True) + 1e-8)

            sims = (audio_feats @ text_feats.T)[0].cpu().numpy()
            preds.append(genre_list.index(genre_list[np.argmax(sims)]))
            targets.append(row["label"])
        except Exception:
            failed += 1

    acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nTest Accuracy: {acc*100:.2f}%  (failed: {failed})")
    save_run_results(
        model="clap", variant=args.variant, mode="zero_shot",
        test_accuracy=acc,
        config={"clip_secs": args.clip_secs, "target_sr": TARGET_SR},
        per_class_f1=f1s,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: finetune | Variant: {args.variant} | Device: {device}")

    if args.variant == "laion":
        processor, clap_model = load_laion_clap(device)
    else:
        processor, clap_model = load_microsoft_clap(device)

    classifier = CLAPClassifier(audio_dim=512, num_classes=NUM_GENRES).to(device)
    criterion = nn.CrossEntropyLoss()

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    max_len = int(TARGET_SR * args.clip_secs)
    preprocess_fn = make_clap_preprocess(max_len)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn, target_sr=TARGET_SR,
        clip_secs=args.clip_secs, batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW([
        {"params": clap_model.audio_model.parameters(), "lr": args.lr_audio},
        {"params": classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-7,
    )

    best_val_acc = 0.0
    ckpt_file = checkpoint_path("clap", args.variant)
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
                audio=waveforms.cpu().numpy(),
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
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  ", leave=False):
                waveforms = batch["waveform"].to(device)
                labels = batch["labels"].to(device)
                audio_inputs = processor(
                    audio=waveforms.cpu().numpy(),
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                ).to(device)
                audio_out = clap_model.get_audio_features(**audio_inputs)
                audio_feats = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out
                logits = classifier(audio_feats)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(targets, preds)
        val_accs.append(val_acc)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "clap_state": clap_model.state_dict(),
                "classifier_state": classifier.state_dict(),
            }, ckpt_file)
            flag = "  <-- best"
        print(f"Epoch {epoch+1:02d} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}{flag}")

    # test
    ckpt = torch.load(ckpt_file, weights_only=True)
    clap_model.load_state_dict(ckpt["clap_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    clap_model.eval()
    classifier.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            waveforms = batch["waveform"].to(device)
            labels = batch["labels"].to(device)
            audio_inputs = processor(
                audio=waveforms.cpu().numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            ).to(device)
            audio_out = clap_model.get_audio_features(**audio_inputs)
            audio_feats = audio_out.pooler_output if hasattr(audio_out, "pooler_output") else audio_out
            logits = classifier(audio_feats)
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
    plt.savefig(figures_path(f"clap_{args.variant}_training_curves.png"), dpi=150)
    plt.close()

    save_run_results(
        model="clap", variant=args.variant, mode="finetune",
        test_accuracy=test_acc, best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr_audio": args.lr_audio, "lr_head": args.lr_head,
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
