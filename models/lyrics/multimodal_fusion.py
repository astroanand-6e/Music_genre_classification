"""
Multimodal Audio + Lyrics Genre Classifier

Fuses pre-computed **audio** embeddings (MERT, CLAP, MusicLDM-VAE, …) with
**lyrics** embeddings (from ``lyrics_embedder.py``) and trains a combined
classifier.

How to combine modalities (design space)
----------------------------------------

1. **Early fusion (implemented)** — Stack or project audio + lyrics, then MLP.
   ``concat``: ``[z_audio ; z_lyrics]``. ``gated``: learned mixture of projected
   modalities (useful when lyrics are missing or noisy).

2. **Shared CLAP space** — CLAP text encoder for lyrics + CLAP audio encoder for
   waveforms puts both in one geometry (good for similarity and smaller fusion
   heads). Use ``lyrics_embedder.py --backend clap`` and CLAP audio features.

3. **Missing lyrics** — Placeholder text (e.g. ``[no lyrics]``) → fixed
   embedding; or future: learned unk token / audio-only skip connection.

4. **Late fusion (future)** — Separate classifiers + meta-learner; strong when
   audio vs lyrics disagree (genre fusion).

Align examples on ``track_ids`` in ``.npz`` files from ``--save_embeddings`` and
from ``lyrics_embedder.py``.

Comparison
----------
  audio-only  → logistic regression on audio embeddings
  lyrics-only → logistic regression on lyrics embeddings
  audio+lyrics → concat LR + multimodal MLP (concat or gated)

Usage
-----
    python models/lyrics/multimodal_fusion.py \\
        --audio_embed results/features/mert_330m_fma_small.npz \\
        --lyrics_embed results/lyrics_embeddings/lyrics_st_small.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_label_maps, RESULTS_DIR, figures_path, save_run_results,
)

FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
EMBED_CACHE_DIR = os.path.join(RESULTS_DIR, "lyrics_embeddings")


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Audio + Lyrics classifier")
    p.add_argument("--audio_embed", required=True,
                   help="Path to .npz with keys: embeddings, labels, track_ids")
    p.add_argument("--lyrics_embed", required=True,
                   help="Path to .npz from lyrics_embedder.py")
    p.add_argument("--fusion", choices=["concat", "gated"], default="concat",
                   help="How to combine audio and lyrics embeddings")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    return p.parse_args()


# __ Gated Fusion Module __

class GatedFusion(nn.Module):
    """Learns a per-feature gate to weight audio vs lyrics contributions."""
    def __init__(self, audio_dim: int, lyrics_dim: int, out_dim: int):
        super().__init__()
        self.proj_audio = nn.Linear(audio_dim, out_dim)
        self.proj_lyrics = nn.Linear(lyrics_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(audio_dim + lyrics_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, audio_emb, lyrics_emb):
        pa = self.proj_audio(audio_emb)
        pl = self.proj_lyrics(lyrics_emb)
        g = self.gate(torch.cat([audio_emb, lyrics_emb], dim=-1))
        return g * pa + (1 - g) * pl


class MultimodalMLP(nn.Module):
    def __init__(self, audio_dim: int, lyrics_dim: int, num_classes: int,
                 hidden: int, fusion: str = "concat"):
        super().__init__()
        self.fusion_type = fusion
        if fusion == "concat":
            fused_dim = audio_dim + lyrics_dim
        else:
            fused_dim = hidden
            self.fusion_layer = GatedFusion(audio_dim, lyrics_dim, fused_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, audio_emb, lyrics_emb):
        if self.fusion_type == "concat":
            fused = torch.cat([audio_emb, lyrics_emb], dim=-1)
        else:
            fused = self.fusion_layer(audio_emb, lyrics_emb)
        return self.classifier(fused)


# __ Align embeddings by track_id __

def align_embeddings(audio_data: dict, lyrics_data: dict):
    """Align audio and lyrics embeddings on shared track_ids."""
    audio_ids = {str(tid): i for i, tid in enumerate(audio_data["track_ids"])}
    lyrics_ids = {str(tid): i for i, tid in enumerate(lyrics_data["track_ids"])}
    shared = sorted(set(audio_ids) & set(lyrics_ids))

    if not shared:
        raise RuntimeError("No shared track_ids between audio and lyrics embeddings.")

    a_idx = [audio_ids[tid] for tid in shared]
    l_idx = [lyrics_ids[tid] for tid in shared]

    audio_emb = audio_data["embeddings"][a_idx]
    lyrics_emb = lyrics_data["embeddings"][l_idx]
    labels = audio_data["labels"][a_idx]

    print(f"Shared tracks: {len(shared)} "
          f"(audio: {len(audio_ids)}, lyrics: {len(lyrics_ids)})")
    return audio_emb.astype(np.float32), lyrics_emb.astype(np.float32), labels.astype(np.int32)


# __ Training __

def train_multimodal(X_audio_tr, X_lyrics_tr, y_tr,
                     X_audio_val, X_lyrics_val, y_val,
                     X_audio_te, X_lyrics_te, y_te,
                     audio_dim, lyrics_dim, num_classes, args) -> tuple[float, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalMLP(audio_dim, lyrics_dim, num_classes, args.hidden,
                          args.fusion).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def to_tensor(x):
        return torch.from_numpy(x).float().to(device)

    a_tr, l_tr, y_tr_ = to_tensor(X_audio_tr), to_tensor(X_lyrics_tr), torch.from_numpy(y_tr).long().to(device)
    a_v, l_v = to_tensor(X_audio_val), to_tensor(X_lyrics_val)
    a_te, l_te = to_tensor(X_audio_te), to_tensor(X_lyrics_te)

    best_val = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(a_tr, l_tr)
        loss = crit(logits, y_tr_)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(a_v, l_v).argmax(1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d}/{args.epochs} | val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(a_te, l_te).argmax(1).cpu().numpy()
    return accuracy_score(y_te, test_preds), test_preds


# __ Main __

def main():
    args = parse_args()

    print(f"Loading audio embeddings: {args.audio_embed}")
    audio_data = np.load(args.audio_embed, allow_pickle=True)

    print(f"Loading lyrics embeddings: {args.lyrics_embed}")
    lyrics_data = np.load(args.lyrics_embed, allow_pickle=True)

    audio_emb, lyrics_emb, labels = align_embeddings(audio_data, lyrics_data)
    audio_dim = audio_emb.shape[1]
    lyrics_dim = lyrics_emb.shape[1]

    df = load_fma_metadata(subset=args.subset, include_bollywood=args.include_bollywood)
    _, id2label = get_label_maps(df)
    num_classes = len(id2label)

    # Standardize each modality independently
    a_scaler = StandardScaler()
    l_scaler = StandardScaler()
    audio_emb = a_scaler.fit_transform(audio_emb)
    lyrics_emb = l_scaler.fit_transform(lyrics_emb)

    # Splits
    idx = np.arange(len(labels))
    idx_trval, idx_test = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    idx_train, idx_val = train_test_split(idx_trval, test_size=0.2,
                                          stratify=labels[idx_trval], random_state=42)

    A_tr, A_val, A_te = audio_emb[idx_train], audio_emb[idx_val], audio_emb[idx_test]
    L_tr, L_val, L_te = lyrics_emb[idx_train], lyrics_emb[idx_val], lyrics_emb[idx_test]
    y_tr, y_val, y_te = labels[idx_train], labels[idx_val], labels[idx_test]

    print(f"\nAudio dim: {audio_dim} | Lyrics dim: {lyrics_dim} | Classes: {num_classes}")
    print(f"Train: {len(A_tr)} | Val: {len(A_val)} | Test: {len(A_te)}")

    # __ Audio-only baseline (logistic regression) __
    print("\n--- Audio-only (LogReg baseline) ---")
    lr_audio = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    lr_audio.fit(A_tr, y_tr)
    audio_preds = lr_audio.predict(A_te)
    audio_acc = accuracy_score(y_te, audio_preds)
    print(f"Test Accuracy: {audio_acc*100:.2f}%")

    # __ Lyrics-only baseline (logistic regression) __
    print("\n--- Lyrics-only (LogReg baseline) ---")
    lr_lyrics = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    lr_lyrics.fit(L_tr, y_tr)
    lyrics_preds = lr_lyrics.predict(L_te)
    lyrics_acc = accuracy_score(y_te, lyrics_preds)
    print(f"Test Accuracy: {lyrics_acc*100:.2f}%")

    # __ Concat baseline (logistic regression) __
    print("\n--- Audio+Lyrics Concat (LogReg) ---")
    lr_concat = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    lr_concat.fit(np.concatenate([A_tr, L_tr], axis=1), y_tr)
    concat_preds = lr_concat.predict(np.concatenate([A_te, L_te], axis=1))
    concat_acc = accuracy_score(y_te, concat_preds)
    print(f"Test Accuracy: {concat_acc*100:.2f}%")

    # __ Multimodal MLP __
    print(f"\n--- Multimodal MLP ({args.fusion} fusion) ---")
    mm_acc, mm_preds = train_multimodal(
        A_tr, L_tr, y_tr, A_val, L_val, y_val, A_te, L_te, y_te,
        audio_dim, lyrics_dim, num_classes, args,
    )
    print(f"Test Accuracy: {mm_acc*100:.2f}%")
    print(classification_report(
        y_te, mm_preds,
        target_names=[id2label[i] for i in sorted(id2label)],
        zero_division=0,
    ))

    # __ Summary table __
    print("\n=== Comparison Summary ===")
    print(f"Audio-only  (LR):         {audio_acc*100:.2f}%")
    print(f"Lyrics-only (LR):         {lyrics_acc*100:.2f}%")
    print(f"Audio+Lyrics concat (LR): {concat_acc*100:.2f}%")
    print(f"Multimodal MLP ({args.fusion}): {mm_acc*100:.2f}%")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Audio-only\n(LR)", "Lyrics-only\n(LR)", "Audio+Lyrics\nConcat (LR)",
             f"Multimodal MLP\n({args.fusion})"]
    accs = [audio_acc * 100, lyrics_acc * 100, concat_acc * 100, mm_acc * 100]
    colors = ["steelblue", "mediumseagreen", "darkorange", "crimson"]
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(100 / num_classes, ls="--", color="gray", alpha=0.7,
               label=f"Chance ({100/num_classes:.1f}%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Audio vs Lyrics vs Multimodal Fusion")
    ax.legend()
    ax.set_ylim(0, max(accs) + 10)
    plt.tight_layout()

    backend_tag = "st" if "st_" in args.lyrics_embed else "clap"
    out = figures_path(f"multimodal_comparison_{args.fusion}_{backend_tag}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nComparison chart -> {out}")

    # Save run results
    save_run_results(
        model="multimodal_mlp", variant=f"{args.fusion}_{backend_tag}",
        mode="finetune", test_accuracy=mm_acc,
        config={"audio_dim": audio_dim, "lyrics_dim": lyrics_dim,
                "fusion": args.fusion, "hidden": args.hidden, "epochs": args.epochs,
                "subset": args.subset},
        dataset=f"fma_{args.subset}",
    )


if __name__ == "__main__":
    main()
