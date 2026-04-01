"""
Lyrics Embedding Module

Generates fixed-size vector representations from song lyrics text using one of
three backends:

    A. SentenceTransformer  — fast, high-quality, no GPU needed
    B. CLAP text encoder    — aligns with audio embedding space (already in project)
    C. (stub) Large LLM     — placeholder for future extension

The embeddings are cached to disk to avoid re-encoding on every run.

Usage
-----
    python models/lyrics/lyrics_embedder.py --backend sentence_transformer
    python models/lyrics/lyrics_embedder.py --backend clap

    # After Genius fetch (see lyrics_genius.py):
    python models/lyrics/lyrics_embedder.py --backend sentence_transformer \\
        --lyrics_cache results/lyrics_cache_genius.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import load_fma_metadata, RESULTS_DIR

LYRICS_CACHE = os.path.join(RESULTS_DIR, "lyrics_cache.json")
EMBED_CACHE_DIR = os.path.join(RESULTS_DIR, "lyrics_embeddings")

SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLAP_MODEL_ID = "laion/clap-htsat-fused"

NO_LYRICS_PLACEHOLDER = "[no lyrics]"
MIN_LYRICS_LEN = 20


def parse_args():
    p = argparse.ArgumentParser(description="Embed lyrics with a text model")
    p.add_argument("--backend", choices=["sentence_transformer", "clap"],
                   default="sentence_transformer")
    p.add_argument("--subset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--lyrics_cache",
        default=None,
        help="Path to lyrics JSON (default: results/lyrics_cache.json). "
        "Use results/lyrics_cache_genius.json after Genius fetch.",
    )
    return p.parse_args()


def load_lyrics_cache(path: str | None = None) -> dict:
    p = path or LYRICS_CACHE
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def get_text_for_track(track_id, cache: dict) -> str:
    entry = cache.get(str(track_id), {})
    text = entry.get("text", "").strip()
    if len(text) < MIN_LYRICS_LEN:
        return NO_LYRICS_PLACEHOLDER
    return text


# __ Backend A: SentenceTransformer __

def embed_sentence_transformer(texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENTENCE_MODEL_ID)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
    )
    return embeddings  # (N, 384)


# __ Backend B: CLAP text encoder __

def embed_clap(texts: list[str], batch_size: int) -> np.ndarray:
    from transformers import ClapProcessor, ClapModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLAP from {CLAP_MODEL_ID} on {device}...")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
    model.eval()

    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="CLAP text encoding"):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeds.append(text_features.cpu().numpy())
    return np.vstack(all_embeds)  # (N, 512)


# __ Main __

def main():
    args = parse_args()
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

    df = load_fma_metadata(subset=args.subset, include_bollywood=args.include_bollywood)
    cache_path = args.lyrics_cache or LYRICS_CACHE
    cache = load_lyrics_cache(cache_path)
    print(f"Lyrics cache: {cache_path} ({len(cache)} entries)")

    track_ids = df.index.tolist()
    labels = df["label"].tolist()
    genres = df["genre"].tolist()
    texts = [get_text_for_track(tid, cache) for tid in track_ids]

    n_with_lyrics = sum(1 for t in texts if t != NO_LYRICS_PLACEHOLDER)
    print(f"Tracks: {len(track_ids)} | With lyrics: {n_with_lyrics} | "
          f"Backend: {args.backend}")

    if args.backend == "sentence_transformer":
        embeddings = embed_sentence_transformer(texts, args.batch_size)
        embed_dim = embeddings.shape[1]
        out_name = f"lyrics_st_{args.subset}.npz"
    elif args.backend == "clap":
        embeddings = embed_clap(texts, args.batch_size)
        embed_dim = embeddings.shape[1]
        out_name = f"lyrics_clap_{args.subset}.npz"
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    out_path = os.path.join(EMBED_CACHE_DIR, out_name)
    np.savez(
        out_path,
        embeddings=embeddings,
        track_ids=np.array(track_ids, dtype=object),
        labels=np.array(labels, dtype=np.int32),
        genres=np.array(genres, dtype=object),
        has_lyrics=np.array([t != NO_LYRICS_PLACEHOLDER for t in texts]),
    )
    print(f"Embeddings saved -> {out_path}  (shape: {embeddings.shape})")
    return out_path


if __name__ == "__main__":
    main()
