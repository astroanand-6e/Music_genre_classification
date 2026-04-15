"""
Wav2Vec2 Conformer Fine-tuning / Zero-shot on FMA-Small or FMA-Medium

Conformer: A hybrid CNN-Transformer architecture for audio modeling.
Uses facebook/wav2vec2-conformer-rel-pos-large (pretrained on LibriSpeech 960h).

Usage
-----
    # Zero-shot (frozen Conformer → LogisticRegression)
    python models/conformer/finetune.py --mode zero_shot

    # End-to-end fine-tuning
    python models/conformer/finetune.py --mode finetune --epochs 20

    # FMA-Medium
    python models/conformer/finetune.py --mode zero_shot --dataset medium
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoFeatureExtractor, Wav2Vec2ConformerModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    build_dataloaders, fma_audio_dir,
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
    p = argparse.ArgumentParser(description="Conformer training / evaluation")
    add_temporal_cli_args(p)
    add_finetune_callback_args(p)
    add_device_arg(p)
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--dataset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to checkpoint .pt to resume from")
    p.add_argument("--start_epoch", type=int, default=1,
                   help="Epoch number to start from when resuming (for scheduler)")
    p.add_argument("--resume_best_val", type=float, default=0.0,
                   help="Best val acc from the resumed run (for early stopping)")
    return p.parse_args()

MODEL_ID = "facebook/wav2vec2-conformer-rel-pos-large"
TARGET_SR = 16000
HIDDEN_DIM = 1024

# __ Preprocessing __

def make_conformer_preprocess(feature_extractor):
    """Return preprocess_fn for FMADataset — Conformer feature extractor converts
    raw waveform to mel-spectrogram frames."""
    def fn(waveform_np, sr):
        inputs = feature_extractor(
            waveform_np, sampling_rate=sr, return_tensors="pt", padding=True
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}
    return fn


# __ Model heads __

class ConformerFineTuner(nn.Module):
    """Conformer backbone + classification head."""
    def __init__(self, backbone, hidden_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        """Input: (B, seq_len) → output logits (B, num_classes)."""
        out = self.backbone(input_values, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


# __ Zero-shot __

def run_zero_shot(args):
    """Frozen Conformer → mean pooling → LogisticRegression."""
    print("=" * 60)
    print("Conformer Zero-Shot Evaluation")
    print("=" * 60)

    device = resolve_training_device(args.device)
    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    train_df, _, test_df = get_splits(df)
    label2id, id2label = get_label_maps(df)
    audio_dir = fma_audio_dir(args.dataset)

    print(f"\nDataset: FMA-{args.dataset.upper()} ({len(label2id)} genres)")
    print(f"Train split: {len(train_df)} tracks (for linear probe)")
    print(f"Test split: {len(test_df)} tracks")

    # Load model and feature extractor
    print(f"\nLoading {MODEL_ID}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    backbone = Wav2Vec2ConformerModel.from_pretrained(MODEL_ID)
    backbone = backbone.to(device)
    backbone.eval()

    def extract_embeddings(split_df, split_name):
        """Extract frozen Conformer embeddings for a dataframe split."""
        print(f"\nExtracting embeddings from {split_name} split...")
        embs = []
        labs = []
        with torch.no_grad():
            for track_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
                audio_path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
                try:
                    y = load_waveform(audio_path, target_sr=TARGET_SR, duration=args.clip_secs)
                except Exception as e:
                    print(f"  Failed to load {track_id}: {e}")
                    continue

                inputs = feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                out = backbone(**inputs)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()

                embs.append(emb[0])
                labs.append(row["label"])
        return np.array(embs), np.array(labs)

    train_embeddings, train_labels = extract_embeddings(train_df, "train")
    test_embeddings, test_labels = extract_embeddings(test_df, "test")

    print(f"\nTrain embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # Linear probe: fit on train, predict on test
    print("\nFitting LogisticRegression on train embeddings...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings, train_labels)

    predictions = clf.predict(test_embeddings)
    test_accuracy = accuracy_score(test_labels, predictions)
    per_class_f1 = compute_per_class_f1(test_labels, predictions, id2label)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nPer-class F1:")
    for genre, f1 in sorted(per_class_f1.items()):
        print(f"  {genre:25s}: {f1:.4f}")

    config = {
        "clip_secs": args.clip_secs,
        "target_sr": TARGET_SR,
        "dataset": args.dataset,
        "include_bollywood": args.include_bollywood,
        "num_train_tracks": len(train_df),
        "num_test_tracks": len(test_df),
    }

    save_run_results(
        model="conformer",
        variant="wav2vec2",
        mode="zero_shot",
        test_accuracy=test_accuracy,
        config=config,
        per_class_f1=per_class_f1,
        dataset=args.dataset,
    )

    print(f"\n✓ Results saved to results/runs/")


# __ Fine-tuning __

def run_finetune(args):
    """End-to-end fine-tuning with early stopping."""
    print("=" * 60)
    print("Conformer Fine-Tuning")
    print("=" * 60)

    device = resolve_training_device(args.device)
    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    train_df, val_df, test_df = get_splits(df)
    label2id, id2label = get_label_maps(df)
    audio_dir = fma_audio_dir(args.dataset)
    num_classes = len(label2id)

    print(f"\nDataset: FMA-{args.dataset.upper()} ({num_classes} genres)")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load feature extractor and backbone
    print(f"\nLoading {MODEL_ID}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    backbone = Wav2Vec2ConformerModel.from_pretrained(MODEL_ID)
    backbone.gradient_checkpointing_enable()

    # Build dataloaders
    preprocess_fn = make_conformer_preprocess(feature_extractor)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        preprocess_fn=preprocess_fn,
        target_sr=TARGET_SR,
        clip_secs=args.clip_secs,
        batch_size=args.batch_size,
        audio_dir=audio_dir,
        train_temporal_sampling=args.temporal_train,
        eval_temporal_sampling=args.temporal_eval,
    )

    # Build model
    model = ConformerFineTuner(backbone, hidden_dim=HIDDEN_DIM, num_classes=num_classes)

    # Resume from checkpoint if provided
    if args.resume_ckpt:
        print(f"\nResuming from checkpoint: {args.resume_ckpt}")
        state = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Loaded. Resuming from epoch {args.start_epoch}, "
              f"best val so far = {args.resume_best_val:.4f}")

    model = model.to(device)

    # Optimizer with separate LRs
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # Training loop
    run_id = new_finetune_run_id()
    epoch_log_csv = finetune_epoch_csv_path("conformer", "wav2vec2", run_id, args.dataset)
    best_val_acc = args.resume_best_val
    epochs_since_improvement = 0

    print(f"\nRun ID: {run_id}")
    print(f"Epoch log: {epoch_log_csv}")

    # Fast-forward cosine scheduler to resume_start_epoch - 1
    if args.start_epoch > 1:
        for _ in range(args.start_epoch - 1):
            scheduler.step()
        print(f"  Scheduler fast-forwarded to epoch {args.start_epoch}")

    for epoch in range(args.start_epoch, args.epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        device_type = "cuda" if device.type == "cuda" else "cpu"

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d} train", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(train_pbar):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device_type):
                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, labels) / args.grad_accum

            scaler.scale(loss).backward()
            train_loss += loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                train_pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.3f}")

        train_pbar.close()

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d} val  ", leave=False, dynamic_ncols=True):
                input_values = batch["input_values"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels_batch = batch["labels"].to(device)

                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, labels_batch)

                val_loss += loss.item()
                val_correct += (logits.argmax(-1) == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        scheduler.step()
        max_lr = max_optimizer_lr(optimizer)

        # Log epoch
        append_finetune_epoch_log(epoch_log_csv, {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader) * args.grad_accum,
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc,
            "best_val_acc": max(best_val_acc, val_acc),
            "lr_max": max_lr,
        })

        # Early stopping
        if val_acc > best_val_acc + args.early_stop_min_delta:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            # Save checkpoint
            ckpt_path = checkpoint_path("conformer", "wav2vec2", args.dataset)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Epoch {epoch:2d} | Loss {train_loss/len(train_loader)*args.grad_accum:.3f} | "
                  f"Val {val_acc:.4f}* (saved)")
        else:
            epochs_since_improvement += 1
            print(f"Epoch {epoch:2d} | Loss {train_loss/len(train_loader)*args.grad_accum:.3f} | "
                  f"Val {val_acc:.4f}")

            if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
                print(f"\nEarly stopping (patience={args.early_stop_patience})")
                break

    # Test
    print("\n" + "=" * 60)
    print("Evaluating on test split...")
    print("=" * 60)

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            logits = model(input_values, attention_mask=attention_mask)
            preds = logits.argmax(-1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_accuracy = accuracy_score(test_labels, test_preds)
    per_class_f1 = compute_per_class_f1(test_labels, test_preds, id2label)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nPer-class F1:")
    for genre, f1 in sorted(per_class_f1.items()):
        print(f"  {genre:25s}: {f1:.4f}")

    # Confusion matrix and curves
    cm_path = figures_path("confmat", "conformer", "wav2vec2", run_id, args.dataset)
    save_genre_confusion_matrix_png(test_labels, test_preds, id2label, cm_path)
    print(f"\nConfusion matrix saved: {cm_path}")

    print_test_classification_report(test_labels, test_preds, id2label)

    # Save results
    config = {
        "clip_secs": args.clip_secs,
        "target_sr": TARGET_SR,
        "epochs": args.epochs,
        "epochs_run": epoch,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "grad_accum": args.grad_accum,
        "dataset": args.dataset,
        "include_bollywood": args.include_bollywood,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "run_id": run_id,
        "temporal_train": args.temporal_train,
        "temporal_eval": args.temporal_eval,
    }

    ckpt_path = checkpoint_path("conformer", "wav2vec2", args.dataset)
    save_run_results(
        model="conformer",
        variant="wav2vec2",
        mode="finetune",
        test_accuracy=test_accuracy,
        best_val_accuracy=best_val_acc,
        config=config,
        per_class_f1=per_class_f1,
        extra={
            "checkpoint": ckpt_path,
            "training_log_csv": epoch_log_csv,
            "confusion_matrix_png": cm_path,
        },
        dataset=args.dataset,
    )

    print(f"\n✓ Results saved to results/runs/")


def main():
    args = parse_args()

    if args.mode == "zero_shot":
        run_zero_shot(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    main()
