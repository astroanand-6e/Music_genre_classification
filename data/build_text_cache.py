"""
Build and cache text representations for FMA tracks from metadata.

Constructs a natural-language description per track from:
  - Track title         (100% coverage)
  - Artist name         (100% coverage)
  - Artist biography    (70% coverage, HTML-stripped)
  - Artist tags         (98% coverage — ABLATION ONLY, see --include_tags)

Output: JSON file mapping track_id (int) -> text string
  data/text_cache_<subset>[_with_tags].json

Usage
-----
# Default (bio + name + title, NO tags):
python data/build_text_cache.py --dataset medium

# Ablation variant (adds artist tags):
python data/build_text_cache.py --dataset medium --include_tags

# Both variants in one shot:
python data/build_text_cache.py --dataset medium --both
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

_here = Path(__file__).resolve().parent
_project_root = _here.parent
sys.path.insert(0, str(_project_root))

from data.data_utils import META_DIR, BASE_DIR


# ── HTML stripping ────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    if not text or pd.isna(text):
        return ""
    text = _TAG_RE.sub(" ", str(text))
    text = _SPACE_RE.sub(" ", text).strip()
    # Remove common FMA boilerplate
    for phrase in [
        "This file is licensed under the Creative Commons",
        "Attribution-NonCommercial",
        "freemusicarchive.org",
    ]:
        if phrase in text:
            text = text[:text.index(phrase)].strip()
    return text


def parse_tags(raw) -> list[str]:
    """Parse tag string like \"['rock', 'indie']\" into a clean list."""
    if not raw or pd.isna(raw):
        return []
    s = str(raw).strip()
    if s in ("[]", "nan", "None", ""):
        return []
    try:
        import ast
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(t).strip() for t in val if str(t).strip()]
    except Exception:
        pass
    # Fallback: strip brackets, split on comma
    s = s.strip("[]").replace("'", "").replace('"', "")
    return [t.strip() for t in s.split(",") if t.strip()]


# ── Text builder ──────────────────────────────────────────────────────────────

def build_text(row, include_tags: bool = False) -> str:
    """
    Construct a natural-language text description for a single track.

    Structure (fields omitted when empty):
        Artist: <name>. Bio: <bio>. Track: <title>. [Tags: <tags>.]
    """
    parts = []

    # Artist name
    artist_name = str(row.get("artist_name", "")).strip()
    if artist_name and artist_name not in ("nan", "None"):
        parts.append(f"Artist: {artist_name}.")

    # Artist biography (HTML-stripped)
    bio = strip_html(row.get("artist_bio", ""))
    if bio:
        # Trim very long bios to ~300 chars to avoid token overflow
        if len(bio) > 300:
            bio = bio[:300].rsplit(" ", 1)[0] + "..."
        parts.append(f"Bio: {bio}.")

    # Track title
    title = str(row.get("track_title", "")).strip()
    if title and title not in ("nan", "None"):
        parts.append(f"Track: {title}.")

    # Artist tags — ABLATION ONLY
    if include_tags:
        tags = parse_tags(row.get("artist_tags", ""))
        if tags:
            parts.append(f"Tags: {', '.join(tags)}.")

    return " ".join(parts) if parts else "Unknown artist."


# ── Main ──────────────────────────────────────────────────────────────────────

def load_metadata(subset: str) -> pd.DataFrame:
    tracks = pd.read_csv(
        os.path.join(META_DIR, "tracks.csv"),
        index_col=0, header=[0, 1], low_memory=False,
    )

    subsets = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            "category", categories=subsets, ordered=True)
    except (ValueError, TypeError):
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=subsets, ordered=True))

    if subset == "medium":
        df = tracks[tracks[("set", "subset")] <= "medium"]
    else:
        df = tracks[tracks[("set", "subset")] == subset]

    # Flatten to a simple DataFrame
    flat = pd.DataFrame(index=df.index)
    flat["track_title"]  = df[("track", "title")]
    flat["artist_name"]  = df[("artist", "name")]
    flat["artist_bio"]   = df[("artist", "bio")]
    flat["artist_tags"]  = df[("artist", "tags")]
    flat["genre"]        = df[("track", "genre_top")]
    return flat


def build_cache(subset: str, include_tags: bool) -> dict:
    print(f"Loading FMA-{subset.upper()} metadata...")
    df = load_metadata(subset)
    print(f"  Tracks: {len(df)}")

    cache = {}
    empty_bio = 0
    for track_id, row in df.iterrows():
        text = build_text(row, include_tags=include_tags)
        cache[int(track_id)] = text
        if "Bio:" not in text:
            empty_bio += 1

    print(f"  Tracks with bio   : {len(df) - empty_bio}/{len(df)} ({100*(len(df)-empty_bio)/len(df):.1f}%)")
    print(f"  Tracks without bio: {empty_bio}/{len(df)} (will use name + title only)")
    return cache


def save_cache(cache: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved → {path}  ({size_kb:.0f} KB, {len(cache)} entries)")


def preview(cache: dict, n: int = 5):
    print(f"\n  {'─'*70}")
    print("  Sample entries:")
    for i, (tid, text) in enumerate(list(cache.items())[:n]):
        print(f"  [{tid}] {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"  {'─'*70}")


def run(args):
    data_dir = _project_root / "data"
    variants = []

    if args.both:
        variants = [(False, "no_tags"), (True, "with_tags")]
    elif args.include_tags:
        variants = [(True, "with_tags")]
    else:
        variants = [(False, "no_tags")]

    for include_tags, suffix in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {suffix}  |  Dataset: FMA-{args.dataset.upper()}")
        print(f"{'='*60}")

        cache = build_cache(args.dataset, include_tags=include_tags)

        fname = f"text_cache_{args.dataset}_{suffix}.json"
        save_cache(cache, data_dir / fname)
        preview(cache)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Build FMA text metadata cache for audio-text fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="medium", choices=["small", "medium"])
    parser.add_argument(
        "--include_tags", action="store_true",
        help="Include artist tags (ablation variant — may leak genre signal)",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Build both no_tags and with_tags variants",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
