"""
Audio Spectrogram Transformer (AST) Fine-tuning / Zero-shot on FMA-Small

Usage
-----
    # Zero-shot (frozen AST → CLS+dist pooled → LogisticRegression)
    python models/ast/finetune.py --mode zero_shot

    # End-to-end fine-tuning
    python models/ast/finetune.py --mode finetune --epochs 20
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ASTFeatureExtractor, ASTModel, ASTForAudioClassification
from sklearn.linear_model import LogisticRegression
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
    p = argparse.ArgumentParser(description="AST training / evaluation")
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=2)
    return p.parse_args()

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
TARGET_SR = 16000
HIDDEN_DIM = 768
AST_MAX_LEN = 1024
AST_MEL_BINS = 128
NUM_GENRES = 8

# __ Preprocessing __

def make_ast_preprocess(feature_extractor):
    """Return preprocess_fn for FMADataset — AST feature extractor handles
    raw audio → log-mel spectrogram internally."""
    def fn(waveform_np, sr):
        inputs = feature_extractor(
            waveform_np, sampling_rate=sr,
            return_tensors="pt", padding="max_length", truncation=True,
        )
        return {"input_values": inputs.input_values.squeeze(0)}
    return fn

# __ Zero-shot (linear probe) __

def run_zero_shot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: zero_shot | Model: AST | Device: {device}")

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    max_len = int(TARGET_SR * args.clip_secs)

    def extract(split_df, desc):
        feats, labs = [], []
        for track_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            path = get_audio_path(track_id)
            try:
                wav = load_waveform(path, TARGET_SR, args.clip_secs)
                if len(wav) < max_len:
                    wav = np.pad(wav, (0, max_len - len(wav)))
                inputs = feature_extractor(
                    wav, sampling_rate=TARGET_SR,
                    return_tensors="pt", padding="max_length", truncation=True,
                ).to(device)
                with torch.no_grad():
                    out = model(**inputs)
                # mean of CLS (index 0) and distillation (index 1) tokens
                emb = out.last_hidden_state[:, :2, :].mean(dim=1).squeeze().cpu().numpy()
                feats.append(emb)
                labs.append(row["label"])
            except Exception:
                feats.append(np.zeros(HIDDEN_DIM))
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
        model="ast", variant="", mode="zero_shot",
        test_accuracy=acc,
        config={"clip_secs": args.clip_secs, "target_sr": TARGET_SR},
        per_class_f1=f1s,
    )

# __ Fine-tune __

def run_finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: finetune | Model: AST | Device: {device}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"Eff. batch: {args.batch_size * args.grad_accum}")

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)

    df = load_fma_metadata()
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_GENRES,
        ignore_mismatched_sizes=True,
        id2label=id2label, label2id=label2id,
    ).to(device)

    preprocess_fn = make_ast_preprocess(feature_extractor)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn, target_sr=TARGET_SR,
        clip_secs=args.clip_secs, batch_size=args.batch_size,
    )

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-8,
    )

    best_val_acc = 0.0
    ckpt_file = checkpoint_path("ast")
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
                outputs = model(input_values=inp)
                loss = criterion(outputs.logits, lab) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    outputs = model(input_values=inp)
                preds.extend(outputs.logits.argmax(1).cpu().numpy())
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
                outputs = model(input_values=inp)
            preds.extend(outputs.logits.argmax(1).cpu().numpy())
            targets.extend(lab.cpu().numpy())

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
    plt.savefig(figures_path("ast_training_curves.png"), dpi=150)
    plt.close()

    save_run_results(
        model="ast", variant="", mode="finetune",
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
