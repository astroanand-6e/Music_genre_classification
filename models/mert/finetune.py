"""
MERT Fine-tuning / Zero-shot Evaluation on FMA-Small

Usage
-----
    # Zero-shot linear probe (frozen MERT → LogisticRegression)
    python models/mert/finetune.py --mode zero_shot --model_size 95m
    python models/mert/finetune.py --mode zero_shot --model_size 330m

    # End-to-end fine-tuning
    python models/mert/finetune.py --mode finetune --model_size 330m --epochs 15
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    build_dataloaders, AUDIO_DIR, RESULTS_DIR,
    save_run_results, checkpoint_path, figures_path,
    compute_per_class_f1, get_audio_path, load_waveform,
)

# __ CLI __

def parse_args():
    p = argparse.ArgumentParser(description="MERT training / evaluation")
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--model_size", choices=["95m", "330m"], default="95m")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=4)
    return p.parse_args()

MODEL_IDS = {
    "95m":  "m-a-p/MERT-v1-95M",
    "330m": "m-a-p/MERT-v1-330M",
}
HIDDEN_DIMS = {"95m": 768, "330m": 1024}
TARGET_SR = 24000
NUM_GENRES = 8

# __ Preprocessing __

def make_mert_preprocess(processor, max_len):
    """Return a preprocess_fn compatible with FMADataset."""
    def fn(waveform_np, sr):
        inputs = processor(
            waveform_np, sampling_rate=sr,
            return_tensors="pt", padding="max_length",
            max_length=max_len, truncation=True,
        )
        return {"input_values": inputs.input_values.squeeze(0)}
    return fn

# __ Classifier head __

class MERTFineTuner(nn.Module):
    def __init__(self, base_model, hidden_dim, num_classes=8):
        super().__init__()
        self.mert = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values):
        outputs = self.mert(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# __ Zero-shot (linear probe) __

def run_zero_shot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = MODEL_IDS[args.model_size]
    hidden_dim = HIDDEN_DIMS[args.model_size]

    print(f"Mode: zero_shot | Model: {model_id} | Device: {device}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    max_len = TARGET_SR * int(args.clip_secs)

    def extract(split_df, desc):
        feats, labs = [], []
        for track_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            path = get_audio_path(track_id)
            try:
                wav = load_waveform(path, TARGET_SR, args.clip_secs)
                if len(wav) < max_len:
                    wav = np.pad(wav, (0, max_len - len(wav)))
                inputs = processor(
                    wav, sampling_rate=TARGET_SR,
                    return_tensors="pt", padding="max_length",
                    max_length=max_len, truncation=True,
                ).to(device)
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                feats.append(emb)
                labs.append(row["label"])
            except Exception:
                feats.append(np.zeros(hidden_dim))
                labs.append(row["label"])
        return np.array(feats), np.array(labs)

    X_train, y_train = extract(train_df, "Extracting train")
    X_test, y_test = extract(test_df, "Extracting test")

    print("Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1s = compute_per_class_f1(y_test, y_pred, id2label)

    print(f"\nTest Accuracy: {acc*100:.2f}%")
    save_run_results(
        model="mert", variant=args.model_size, mode="zero_shot",
        test_accuracy=acc, config={"clip_secs": args.clip_secs, "target_sr": TARGET_SR},
        per_class_f1=f1s,
    )

# __ Fine-tune (end-to-end) __

def run_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = MODEL_IDS[args.model_size]
    hidden_dim = HIDDEN_DIMS[args.model_size]

    print(f"Mode: finetune | Model: {model_id} | Device: {device}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"Eff. batch: {args.batch_size * args.grad_accum}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    model = MERTFineTuner(base_model, hidden_dim, NUM_GENRES).to(device)

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    max_len = TARGET_SR * int(args.clip_secs)
    preprocess_fn = make_mert_preprocess(processor, max_len)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn, target_sr=TARGET_SR,
        clip_secs=args.clip_secs, batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW([
        {"params": model.mert.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-7,
    )

    best_val_acc = 0.0
    ckpt_file = checkpoint_path("mert", args.model_size)
    train_losses, val_accs = [], []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [train]")

        for step, batch in enumerate(loop):
            inp = batch["input_values"].to(device)
            lab = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                logits = model(inp)
                loss = criterion(logits, lab) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            loop.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  ", leave=False):
                inp = batch["input_values"].to(device)
                lab = batch["labels"].to(device)
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(inp)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(lab.cpu().numpy())

        val_acc = accuracy_score(targets, preds)
        val_accs.append(val_acc)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_file)
            flag = "  <-- best"
        print(f"Epoch {epoch+1:02d} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}{flag}")

    # test
    model.load_state_dict(torch.load(ckpt_file, weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inp = batch["input_values"].to(device)
            lab = batch["labels"].to(device)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(inp)
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(lab.cpu().numpy())

    test_acc = accuracy_score(targets, preds)
    f1s = compute_per_class_f1(targets, preds, id2label)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")

    # training curves
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
    plt.savefig(figures_path(f"mert_{args.model_size}_training_curves.png"), dpi=150)
    plt.close()

    save_run_results(
        model="mert", variant=args.model_size, mode="finetune",
        test_accuracy=test_acc, best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
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
