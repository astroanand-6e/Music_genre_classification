"""
Lyrics-Only Genre Classifier

Trains and evaluates a genre classifier on lyrics embeddings (from
``lyrics_embedder.py``) using logistic regression and an MLP.

Covers:
  - Lyrics-only logistic regression baseline
  - Lyrics-only MLP classifier
  - Genre fusion detection via embedding similarity
  - t-SNE / UMAP visualisation of lyrics embedding space

Usage
-----
    # Lyrics-only classification (requires embeddings computed first)
    python models/lyrics/lyrics_classifier.py \\
        --embed_path results/lyrics_embeddings/lyrics_st_small.npz

    # Skip tracks without real lyrics
    python models/lyrics/lyrics_classifier.py \\
        --embed_path results/lyrics_embeddings/lyrics_st_small.npz \\
        --lyrics_only_tracks
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_label_maps, RESULTS_DIR, figures_path, save_run_results,
)

EMBED_CACHE_DIR = os.path.join(RESULTS_DIR, "lyrics_embeddings")


def parse_args():
    p = argparse.ArgumentParser(description="Lyrics-only genre classifier + fusion analysis")
    p.add_argument("--embed_path", required=True,
                   help="Path to .npz file from lyrics_embedder.py")
    p.add_argument("--lyrics_only_tracks", action="store_true",
                   help="Exclude tracks with no lyrics (placeholder embedding)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fusion_topk", type=int, default=10,
                   help="Top-K nearest neighbours per track for fusion analysis")
    p.add_argument("--viz", action="store_true", default=True,
                   help="Generate t-SNE / UMAP visualisations")
    return p.parse_args()


# __ MLP classifier __

class LyricsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
              input_dim, num_classes, args) -> tuple[float, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LyricsMLP(input_dim, args.hidden, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val).long().to(device)
    X_te = torch.from_numpy(X_test).float().to(device)

    best_val = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr)
        loss = crit(logits, y_tr)
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v).argmax(1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_te).argmax(1).cpu().numpy()
    return accuracy_score(y_test, test_preds), test_preds


# __ Genre fusion detection __

def detect_genre_fusion(embeddings: np.ndarray, genres: np.ndarray,
                        track_ids, topk: int, id2label: dict) -> list[dict]:
    """For each track, find its topk nearest neighbours.  If they span
    multiple genres the track is a genre-fusion candidate."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -1)  # exclude self

    fusion_candidates = []
    for i, (tid, genre) in enumerate(zip(track_ids, genres)):
        top_indices = np.argsort(sim[i])[::-1][:topk]
        neighbour_genres = set(genres[j] for j in top_indices)
        if len(neighbour_genres) > 1 and genre in neighbour_genres:
            avg_cross_sim = float(np.mean([sim[i, j] for j in top_indices
                                           if genres[j] != genre]))
            fusion_candidates.append({
                "track_id": str(tid),
                "genre": genre,
                "neighbour_genres": sorted(neighbour_genres - {genre}),
                "avg_cross_genre_sim": round(avg_cross_sim, 4),
            })
    return sorted(fusion_candidates, key=lambda x: -x["avg_cross_genre_sim"])


# __ Visualisation __

def visualise_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                         id2label: dict, method: str = "umap", title: str = ""):
    """Generate a 2-D projection of the lyrics embedding space."""
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                                min_dist=0.1)
        except ImportError:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    proj = reducer.fit_transform(embeddings)

    genre_names = [id2label[i] for i in sorted(id2label)]
    colors = plt.cm.tab20(np.linspace(0, 1, len(genre_names)))

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, gname in enumerate(genre_names):
        mask = labels == i
        ax.scatter(proj[mask, 0], proj[mask, 1], label=gname,
                   color=colors[i], alpha=0.5, s=10)

    ax.set_title(f"Lyrics Embeddings — {method.upper()} | {title}")
    ax.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    plt.tight_layout()

    out = figures_path(f"lyrics_{method}_{title.replace(' ', '_')}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualisation saved -> {out}")
    return out


# __ Main __

def main():
    args = parse_args()

    print(f"Loading embeddings: {args.embed_path}")
    data = np.load(args.embed_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    track_ids = data["track_ids"]
    labels = data["labels"].astype(np.int32)
    genres = data["genres"]
    has_lyrics = data["has_lyrics"]

    backend = "st" if "st_" in args.embed_path else "clap"
    subset = "small" if "small" in args.embed_path else "medium"

    # Load metadata to get id2label
    df = load_fma_metadata(
        subset=subset,
        include_bollywood=("bollywood" in args.embed_path.lower()),
    )
    _, id2label = get_label_maps(df)
    num_classes = len(id2label)

    if args.lyrics_only_tracks:
        mask = has_lyrics
        embeddings = embeddings[mask]
        labels = labels[mask]
        genres = genres[mask]
        track_ids = track_ids[mask]
        print(f"Filtered to tracks with lyrics: {mask.sum()} / {len(mask)}")

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Train / val / test split
    X_trval, X_test, y_trval, y_test = train_test_split(
        embeddings_scaled, labels, test_size=0.2, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.2, stratify=y_trval, random_state=42)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Embed dim: {embeddings.shape[1]} | Classes: {num_classes}")

    # __ 1. Logistic regression __
    print("\n--- Logistic Regression (lyrics-only) ---")
    lr_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    lr_clf.fit(X_train, y_train)
    lr_preds = lr_clf.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    print(f"Test Accuracy: {lr_acc*100:.2f}%")
    print(classification_report(y_test, lr_preds,
                                target_names=[id2label[i] for i in sorted(id2label)],
                                zero_division=0))

    save_run_results(
        model="lyrics_lr", variant=backend, mode="zero_shot",
        test_accuracy=lr_acc,
        config={"embed_dim": embeddings.shape[1], "backend": backend,
                "subset": subset, "lyrics_only": args.lyrics_only_tracks},
        dataset=f"fma_{subset}",
    )

    # __ 2. MLP __
    print("\n--- MLP Classifier (lyrics-only) ---")
    mlp_acc, mlp_preds = train_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test,
        embeddings.shape[1], num_classes, args,
    )
    print(f"Test Accuracy: {mlp_acc*100:.2f}%")
    print(classification_report(y_test, mlp_preds,
                                target_names=[id2label[i] for i in sorted(id2label)],
                                zero_division=0))

    save_run_results(
        model="lyrics_mlp", variant=backend, mode="finetune",
        test_accuracy=mlp_acc,
        config={"embed_dim": embeddings.shape[1], "backend": backend,
                "subset": subset, "lyrics_only": args.lyrics_only_tracks,
                "hidden": args.hidden, "epochs": args.epochs},
        dataset=f"fma_{subset}",
    )

    # __ 3. Genre fusion analysis __
    print(f"\n--- Genre Fusion Detection (top-{args.fusion_topk} neighbours) ---")
    fusion_candidates = detect_genre_fusion(
        embeddings_scaled, genres, track_ids, args.fusion_topk, id2label,
    )
    print(f"Fusion candidates: {len(fusion_candidates)}")
    for fc in fusion_candidates[:15]:
        other = ", ".join(fc["neighbour_genres"])
        print(f"  Track {fc['track_id']} ({fc['genre']}) ↔ [{other}]  "
              f"sim={fc['avg_cross_genre_sim']:.3f}")

    fusion_path = os.path.join(RESULTS_DIR, f"fusion_candidates_{backend}_{subset}.json")
    with open(fusion_path, "w") as f:
        json.dump(fusion_candidates, f, indent=2)
    print(f"Fusion candidates saved -> {fusion_path}")

    # __ 4. Visualisation __
    if args.viz:
        print("\n--- Visualisation ---")
        try:
            visualise_embeddings(embeddings_scaled, labels, id2label,
                                 method="umap", title=f"{backend}_{subset}")
        except Exception as e:
            print(f"UMAP failed ({e}), trying t-SNE...")
            try:
                visualise_embeddings(embeddings_scaled, labels, id2label,
                                     method="tsne", title=f"{backend}_{subset}")
            except Exception as e2:
                print(f"t-SNE also failed: {e2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
