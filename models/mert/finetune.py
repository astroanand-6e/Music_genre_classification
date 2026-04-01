"""
MERT Fine-tuning / Zero-shot Evaluation on FMA-Small or FMA-Medium

Usage
-----
    # Zero-shot linear probe (frozen MERT → LogisticRegression)
    python models/mert/finetune.py --mode zero_shot --model_size 95m
    python models/mert/finetune.py --mode zero_shot --model_size 330m

    # End-to-end fine-tuning
    python models/mert/finetune.py --mode finetune --model_size 330m --epochs 15

    # FMA-Medium + Bollywood
    python models/mert/finetune.py --mode zero_shot --model_size 330m \\
        --dataset medium --include_bollywood

    # Save embeddings for multimodal fusion
    python models/mert/finetune.py --mode zero_shot --model_size 330m --save_embeddings
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
    p = argparse.ArgumentParser(description="MERT training / evaluation")
    add_temporal_cli_args(p)
    add_finetune_callback_args(p)
    add_device_arg(p)
    p.add_argument("--mode", choices=["zero_shot", "finetune"], required=True)
    p.add_argument("--model_size", choices=["95m", "330m"], default="95m")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--clip_secs", type=float, default=10)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--dataset", choices=["small", "medium"], default="small",
                   help="FMA subset to use")
    p.add_argument("--include_bollywood", action="store_true",
                   help="Add Bollywood tracks as an extra genre class")
    p.add_argument("--save_embeddings", action="store_true",
                   help="Save zero-shot embeddings to disk for multimodal fusion")
    return p.parse_args()

MODEL_IDS = {
    "95m":  "m-a-p/MERT-v1-95M",
    "330m": "m-a-p/MERT-v1-330M",
}
HIDDEN_DIMS = {"95m": 768, "330m": 1024}
TARGET_SR = 24000

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
    model_id = MODEL_IDS[args.model_size]
    hidden_dim = HIDDEN_DIMS[args.model_size]
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")

    device = resolve_training_device(args.device)
    print(f"Mode: zero_shot | Model: {model_id} | Dataset: {dataset_tag}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    label2id, id2label = get_label_maps(df)
    train_df, val_df, test_df = get_splits(df)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )
    max_len = TARGET_SR * int(args.clip_secs)
    split_mode = {
        "Extracting train": ("train", args.temporal_train),
        "Extracting test": ("test", args.temporal_eval),
    }

    def extract(split_df, desc):
        feats, labs, ids = [], [], []
        stag, tmode = split_mode.get(desc, ("train", args.temporal_train))
        for track_id, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            path = resolve_audio_path(track_id, row, audio_dir=audio_dir)
            try:
                off = segment_offset_for_mode(
                    path, track_id, args.clip_secs, stag, tmode,
                )
                wav = load_waveform(path, TARGET_SR, args.clip_secs, offset=off)
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
                ids.append(track_id)
            except Exception:
                feats.append(np.zeros(hidden_dim))
                labs.append(row["label"])
                ids.append(track_id)
        return np.array(feats), np.array(labs), ids

    X_train, y_train, train_ids = extract(train_df, "Extracting train")
    X_test, y_test, test_ids = extract(test_df, "Extracting test")

    if args.save_embeddings:
        feat_dir = os.path.join(RESULTS_DIR, "features")
        os.makedirs(feat_dir, exist_ok=True)
        feat_path = os.path.join(feat_dir, f"mert_{args.model_size}_{dataset_tag}.npz")
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
        model="mert", variant=args.model_size, mode="zero_shot",
        test_accuracy=acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "dataset": args.dataset, "include_bollywood": args.include_bollywood,
            "temporal_train": args.temporal_train, "temporal_eval": args.temporal_eval,
        },
        per_class_f1=f1s,
        dataset=dataset_tag,
    )

# __ Fine-tune (end-to-end) __

def run_finetune(args):
    model_id = MODEL_IDS[args.model_size]
    hidden_dim = HIDDEN_DIMS[args.model_size]
    dataset_tag = f"fma_{args.dataset}" + ("_bollywood" if args.include_bollywood else "")
    device = resolve_training_device(args.device)

    print(f"Mode: finetune | Model: {model_id}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"Eff. batch: {args.batch_size * args.grad_accum}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    df = load_fma_metadata(subset=args.dataset, include_bollywood=args.include_bollywood)
    label2id, id2label = get_label_maps(df)
    num_classes = len(id2label)

    model = MERTFineTuner(base_model, hidden_dim, num_classes).to(device)

    train_df, val_df, test_df = get_splits(df)

    max_len = TARGET_SR * int(args.clip_secs)
    preprocess_fn = make_mert_preprocess(processor, max_len)

    audio_dir = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.dataset}",
    )
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
    epoch_log_csv = finetune_epoch_csv_path("mert", args.model_size, dataset_tag, run_id)
    print(f"Training log CSV -> {epoch_log_csv}  (run_id={run_id})")

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
    epochs_no_improve = 0
    ckpt_file = checkpoint_path("mert", args.model_size, dataset=dataset_tag)
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
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  ", leave=False):
                inp = batch["input_values"].to(device)
                lab = batch["labels"].to(device)
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(inp)
                total_val_loss += criterion(logits, lab).item()
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(lab.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(targets, preds)
        val_accs.append(val_acc)

        flag = ""
        if val_acc > best_val_acc + args.early_stop_min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_file)
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
    model.load_state_dict(torch.load(ckpt_file, weights_only=True))
    model.eval()
    preds, targets = [], []
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
                        pv = processor(
                            wav, sampling_rate=TARGET_SR,
                            return_tensors="pt", padding="max_length",
                            max_length=max_len, truncation=True,
                        ).input_values.to(device)
                        with torch.amp.autocast(device_type="cuda"):
                            lg = model(pv)
                        log_sum = lg if log_sum is None else log_sum + lg
                    pred = (log_sum / len(offs)).argmax(1).item()
                except Exception:
                    pred = 0
                preds.append(pred)
                targets.append(int(row["label"]))
        else:
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
    print("\nTest classification report (per genre):")
    print_test_classification_report(targets, preds, id2label)

    cm_path = figures_path(f"confmat_mert_{args.model_size}_{dataset_tag}_{run_id}.png")
    save_genre_confusion_matrix_png(
        targets, preds, id2label, cm_path,
        title=f"MERT-{args.model_size} fine-tuned — {dataset_tag} (test, acc={test_acc*100:.1f}%)",
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
    curves_png = figures_path(f"mert_{args.model_size}_{dataset_tag}_training_curves_{run_id}.png")
    plt.savefig(curves_png, dpi=150)
    plt.close()

    save_run_results(
        model="mert", variant=args.model_size, mode="finetune",
        test_accuracy=test_acc, best_val_accuracy=best_val_acc,
        config={
            "clip_secs": args.clip_secs, "target_sr": TARGET_SR,
            "epochs": args.epochs, "epochs_run": len(train_losses),
            "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone, "lr_head": args.lr_head,
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
