"""
Build combined text caches for CALM ablation study.

Generates:
  - text_cache_medium_audio_only.json   (empty strings → audio-only baseline)
  - text_cache_medium_combined.json     (metadata + lyrics concatenated)

Usage:
    python data/build_combined_cache.py --dataset medium
"""

import json
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="medium", choices=["small", "medium"])
    args = parser.parse_args()

    data_dir = _root / "data"

    # Load existing caches
    no_tags_path = data_dir / f"text_cache_{args.dataset}_no_tags.json"
    lyrics_path  = data_dir / f"text_cache_{args.dataset}_lyrics.json"

    with open(no_tags_path) as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {len(metadata)} entries")

    # ── Audio-only cache (empty strings) ──
    audio_only = {k: "" for k in metadata}
    out = data_dir / f"text_cache_{args.dataset}_audio_only.json"
    with open(out, "w") as f:
        json.dump(audio_only, f)
    print(f"Written: {out.name} ({len(audio_only)} entries)")

    # ── Combined cache (metadata + lyrics) ──
    if lyrics_path.exists():
        with open(lyrics_path) as f:
            lyrics = json.load(f)
        print(f"Loaded lyrics: {len(lyrics)} entries")

        combined = {}
        n_both = 0
        for tid, meta_text in metadata.items():
            lyric_text = lyrics.get(tid, "")
            if lyric_text.startswith("Lyrics:"):
                # Has real lyrics — append to metadata (skip bio to save tokens)
                artist_line = meta_text.split("Bio:")[0].strip()  # keep artist+title
                combined[tid] = f"{artist_line} {lyric_text}"
                n_both += 1
            else:
                combined[tid] = meta_text  # fallback to metadata only

        out = data_dir / f"text_cache_{args.dataset}_combined.json"
        with open(out, "w", ensure_ascii=False) as f:
            json.dump(combined, f, ensure_ascii=False)
        print(f"Written: {out.name} ({len(combined)} entries, {n_both} with lyrics)")
    else:
        print(f"Lyrics cache not found: {lyrics_path.name} — skipping combined cache")
        print(f"Run: python data/build_lyrics_cache.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
