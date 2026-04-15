"""
CALM Zero-Shot Evaluation
=========================
Loads a trained CALM checkpoint and evaluates on FMA-Small or FMA-Medium test set.
No training — just inference + metrics.

Usage:
    # Evaluate on FMA-Medium (default):
    python models/calm/zero_shot.py --dataset medium \
        --calm_ckpt results/checkpoints/calm/calm_no_tags_medium.pt

    # Evaluate on FMA-Small:
    python models/calm/zero_shot.py --dataset small \
        --calm_ckpt results/checkpoints/calm/calm_no_tags_medium.pt \
        --text_cache data/text_cache_small_no_tags.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

_here = Path(__file__).resolve().parent
_project_root = _here.parent.parent
sys.path.insert(0, str(_project_root))

from models.calm.finetune import (
    CALM, AudioTextDataset, collate_fn,
    load_conformer_backbone, freeze_module,
    build_text_embeddings, get_audio_feature_extractor,
    GEMMA_MODEL_NAME, TEXT_D,
)
from data.data_utils import (
    load_fma_metadata, get_splits, get_label_maps,
    fma_audio_dir, save_run_results, compute_per_class_f1,
)


def main():
    parser = argparse.ArgumentParser(description="CALM zero-shot evaluation")
    parser.add_argument("--dataset", default="medium", choices=["small", "medium"])
    parser.add_argument("--calm_ckpt", default=None, help="Path to CALM checkpoint (.pt)")
    parser.add_argument("--no_ckpt", action="store_true",
                        help="Skip loading checkpoint — evaluate with random fusion layers (true zero-shot)")
    parser.add_argument("--text_cache", default=None,
                        help="Text cache JSON (auto-detected from dataset if not set)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--clip_secs", type=float, default=5.0)
    parser.add_argument("--n_cross_layers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Auto-detect text cache
    if args.text_cache is None:
        args.text_cache = f"data/text_cache_{args.dataset}_no_tags.json"
    text_cache_path = Path(args.text_cache)
    if not text_cache_path.is_absolute():
        text_cache_path = _project_root / text_cache_path

    # ── Data ──
    print(f"\nLoading FMA-{args.dataset.upper()} metadata...")
    df = load_fma_metadata(subset=args.dataset)
    _, _, test_df = get_splits(df)
    label2id, id2label = get_label_maps(df)
    num_classes = len(label2id)
    audio_dir = fma_audio_dir(args.dataset)
    print(f"Test set: {len(test_df)} tracks, {num_classes} genres")

    # ── Text embeddings ──
    print(f"Loading text cache: {text_cache_path.name}")
    with open(text_cache_path) as f:
        text_cache = {int(k): v for k, v in json.load(f).items()}

    emb_cache_path = text_cache_path.with_suffix(".embeddings.npy")
    emb_ids_path   = text_cache_path.with_suffix(".embeddings_ids.npy")

    if emb_cache_path.exists() and emb_ids_path.exists():
        print("Loading cached text embeddings...")
        raw_embs = np.load(str(emb_cache_path))
        raw_ids  = np.load(str(emb_ids_path))
        text_embeddings = {int(tid): emb for tid, emb in zip(raw_ids, raw_embs)}

        # Verify no NaN
        nan_count = sum(1 for emb in text_embeddings.values() if np.isnan(emb).any())
        if nan_count > 0:
            print(f"  WARNING: {nan_count} embeddings have NaN — re-encoding...")
            os.remove(str(emb_cache_path))
            os.remove(str(emb_ids_path))
            text_embeddings = None
    else:
        text_embeddings = None

    if text_embeddings is None:
        text_embeddings = build_text_embeddings(
            text_cache,
            device="cuda" if device.type == "cuda" else "cpu",
        )
        ids_arr  = np.array(list(text_embeddings.keys()), dtype=np.int64)
        embs_arr = np.stack(list(text_embeddings.values()))
        np.save(str(emb_cache_path), embs_arr)
        np.save(str(emb_ids_path),   ids_arr)
        print(f"  Saved embeddings cache → {emb_cache_path.name}")

    # ── Dataset ──
    test_ds = AudioTextDataset(
        test_df, text_cache, text_embeddings,
        audio_dir=audio_dir, clip_secs=args.clip_secs, split="test",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # ── Model ──
    print("\nBuilding CALM model...")
    model = CALM(num_classes=num_classes, n_cross_layers=args.n_cross_layers)

    backbone = load_conformer_backbone()
    freeze_module(backbone)
    model.audio_encoder = backbone
    model = model.to(device)

    # Load checkpoint (or skip for true zero-shot)
    if args.no_ckpt:
        print("*** TRUE ZERO-SHOT: no checkpoint loaded — random fusion layers ***")
    elif args.calm_ckpt:
        print(f"Loading checkpoint: {args.calm_ckpt}")
        state = torch.load(args.calm_ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        parser.error("Either --calm_ckpt or --no_ckpt is required")
    model.eval()

    # ── Evaluate ──
    print("\nRunning evaluation...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval", dynamic_ncols=True):
            mel   = batch["input_features"].to(device)
            mask  = batch["attention_mask"].to(device)
            text  = batch["text_emb"].to(device)

            logits = model(mel, text, attention_mask=mask)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    # ── Metrics ──
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"CALM Zero-Shot Evaluation — FMA-{args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Checkpoint : {args.calm_ckpt}")
    print(f"Text cache : {text_cache_path.name}")
    print(f"Test acc   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}")

    genre_names = [id2label[i] for i in range(num_classes)]
    print(f"\n{classification_report(all_labels, all_preds, target_names=genre_names, zero_division=0)}")

    # Save results
    from sklearn.metrics import f1_score
    f1s = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = {name: round(float(f), 4) for name, f in zip(genre_names, f1s)}
    mode_tag = "zero_shot_untrained" if args.no_ckpt else "zero_shot_eval"
    save_run_results(
        model="calm",
        variant=f"no_tags{'_untrained' if args.no_ckpt else ''}",
        mode=mode_tag,
        test_accuracy=round(acc, 4),
        config={
            "checkpoint": args.calm_ckpt or "none (random init)",
            "text_cache": str(text_cache_path.name),
            "no_ckpt": args.no_ckpt,
            "dataset": args.dataset,
        },
        per_class_f1=per_class_f1,
        dataset=args.dataset,
    )
    print("Results saved to results/runs/")


if __name__ == "__main__":
    main()
