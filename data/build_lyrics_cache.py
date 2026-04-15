"""
Build and cache song lyrics for FMA / Bollywood tracks.

Uses free lyrics APIs (no auth needed):
  1. lyrics.ovh  — free, no auth, good Bollywood coverage
  2. LRCLIB       — free, open-source, synced lyrics

Accepts any script (romanized, Devanagari, etc.) — Gemma 4 E2B handles all.

Usage
-----
python data/build_lyrics_cache.py --dataset bollywood
python data/build_lyrics_cache.py --dataset medium --resume
python data/build_lyrics_cache.py --dataset bollywood --limit 5 --debug
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError

import pandas as pd

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root))

from data.data_utils import META_DIR, BASE_DIR

# ── Config ────────────────────────────────────────────────────────────────────

REQUEST_DELAY    = 0.3
LYRICS_MAX_CHARS = 600
BOLLYWOOD_META_CSV = _root / "data" / "bollywood_metadata.csv"
BOLLYWOOD_OVERRIDES = _root / "data" / "bollywood_lyrics_overrides.json"

# ── Text helpers ──────────────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r"\[.*?\]")
_SPACE_RE   = re.compile(r"\s+")


def clean_lyrics(raw: str) -> str:
    text = _BRACKET_RE.sub("", raw)
    text = _SPACE_RE.sub(" ", text).strip()
    for phrase in ["Embed", "You might also like"]:
        idx = text.rfind(phrase)
        if idx > len(text) * 0.8:
            text = text[:idx].strip()
    return text


def strip_html(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()


# ── API 1: lyrics.ovh (free, no auth) ────────────────────────────────────────

def fetch_lyrics_ovh(artist: str, title: str, debug: bool = False) -> str:
    url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            lyrics = data.get("lyrics", "")
            if lyrics:
                if debug:
                    print(f"    [lyrics.ovh] HIT ({len(lyrics)} chars)")
                return clean_lyrics(lyrics)
            return ""
    except (HTTPError, URLError, Exception) as e:
        if debug:
            print(f"    [lyrics.ovh] miss ({e})")
        return ""


# ── API 2: LRCLIB (free, open-source) ────────────────────────────────────────

def fetch_lyrics_lrclib(artist: str, title: str, debug: bool = False) -> str:
    params = urlencode({"artist_name": artist, "track_name": title})
    url = f"https://lrclib.net/api/search?{params}"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as r:
            results = json.loads(r.read())
        if not results:
            if debug:
                print(f"    [lrclib] no results")
            return ""
        lyrics = results[0].get("plainLyrics") or results[0].get("syncedLyrics") or ""
        if lyrics:
            if debug:
                print(f"    [lrclib] HIT ({len(lyrics)} chars)")
            return clean_lyrics(lyrics)
        return ""
    except Exception as e:
        if debug:
            print(f"    [lrclib] miss ({e})")
        return ""


# ── Combined fetcher ─────────────────────────────────────────────────────────

def fetch_lyrics(artist: str, title: str,
                 delay: float = REQUEST_DELAY,
                 debug: bool = False) -> str:
    """Try each lyrics API in order. Return first match."""
    if debug:
        print(f"\n  [{title}] by [{artist}]")

    # 1. lyrics.ovh
    lyrics = fetch_lyrics_ovh(artist, title, debug)
    if lyrics:
        return lyrics
    time.sleep(delay)

    # 2. LRCLIB
    lyrics = fetch_lyrics_lrclib(artist, title, debug)
    if lyrics:
        return lyrics

    if debug:
        print(f"    → all APIs missed")
    return ""


# ── Fallback text ────────────────────────────────────────────────────────────

def metadata_text(artist: str, title: str, bio: str = "") -> str:
    parts = []
    if artist:
        parts.append(f"Artist: {artist}.")
    if bio:
        bio = bio[:300].rsplit(" ", 1)[0] + "..." if len(bio) > 300 else bio
        parts.append(f"Bio: {bio}.")
    if title:
        parts.append(f"Track: {title}.")
    return " ".join(parts) if parts else "Unknown artist."


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_fma_tracks(subset: str) -> pd.DataFrame:
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

    df = tracks[tracks[("set", "subset")] <= subset] if subset == "medium" \
         else tracks[tracks[("set", "subset")] == subset]

    flat = pd.DataFrame(index=df.index)
    flat["title"]  = df[("track",  "title")].fillna("").astype(str)
    flat["artist"] = df[("artist", "name")].fillna("").astype(str)
    flat["bio"]    = df[("artist", "bio")].fillna("").astype(str)
    return flat


def load_bollywood_tracks() -> pd.DataFrame:
    if not BOLLYWOOD_META_CSV.exists():
        print(f"ERROR: {BOLLYWOOD_META_CSV} not found. "
              "Run data/build_bollywood_metadata.py first.")
        sys.exit(1)
    df = pd.read_csv(BOLLYWOOD_META_CSV).set_index("filename")
    out = pd.DataFrame(index=df.index)
    out["title"]  = df["song_name"]
    out["artist"] = df["artist_name"]
    out["bio"]    = ""
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build lyrics cache (lyrics.ovh + LRCLIB, no auth needed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="medium",
                        choices=["small", "medium", "bollywood"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fallback_metadata", action="store_true", default=True)
    args = parser.parse_args()

    out_path = _root / "data" / f"text_cache_{args.dataset}_lyrics.json"

    # Load manual overrides for bollywood
    overrides = {}
    if args.dataset == "bollywood" and BOLLYWOOD_OVERRIDES.exists():
        with open(BOLLYWOOD_OVERRIDES) as f:
            overrides = json.load(f)
        print(f"  Overrides loaded: {len(overrides)} tracks")

    print(f"Loading {args.dataset} tracks...")
    df = load_bollywood_tracks() if args.dataset == "bollywood" else load_fma_tracks(args.dataset)
    print(f"  Tracks: {len(df)}")
    print(f"  APIs: lyrics.ovh → LRCLIB")

    cache = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            cache = json.load(f)
        print(f"  Resuming — already cached: {len(cache)}")

    if args.dry_run:
        print("\n[DRY RUN] First 10 queries:")
        for i, (tid, row) in enumerate(df.iterrows()):
            if i >= 10:
                break
            print(f"  '{row['artist']}' — '{row['title']}'")
        return

    n_lyrics = n_fallback = n_empty = processed = 0
    track_ids = list(df.index)
    if args.limit:
        track_ids = track_ids[:args.limit]
    total = len(track_ids)

    print(f"\nFetching lyrics for {total} tracks...\n")

    try:
        from tqdm import tqdm
        pbar = tqdm(track_ids, desc="Lyrics", unit="track")
    except ImportError:
        pbar = track_ids

    for tid in pbar:
        str_tid = str(tid)
        if str_tid in cache:
            processed += 1
            continue

        row    = df.loc[tid]
        # Use manual override if available (cleaner artist/title for API)
        if str_tid in overrides:
            artist = overrides[str_tid]["artist"]
            title  = overrides[str_tid]["title"]
        else:
            artist = str(row["artist"]).strip()
            title  = str(row["title"]).strip()
        bio    = str(row.get("bio", "")).strip()

        if not title or title in ("nan", "None"):
            cache[str_tid] = ""
            n_empty += 1
            processed += 1
            continue

        lyrics = fetch_lyrics(artist, title, delay=args.delay, debug=args.debug)

        if lyrics:
            if len(lyrics) > LYRICS_MAX_CHARS:
                lyrics = lyrics[:LYRICS_MAX_CHARS].rsplit(" ", 1)[0] + "..."
            text = f"Lyrics: {lyrics}"
            n_lyrics += 1
        elif args.fallback_metadata:
            bio_clean = strip_html(bio)
            text = metadata_text(artist, title, bio_clean)
            n_fallback += 1
        else:
            text = ""
            n_empty += 1

        cache[str_tid] = text
        processed += 1

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(lyrics=n_lyrics, fallback=n_fallback, refresh=False)

        if processed % 25 == 0:
            with open(out_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)

    with open(out_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Total processed  : {processed}")
    print(f"  Lyrics found     : {n_lyrics}  ({100*n_lyrics/max(processed,1):.1f}%)")
    print(f"  Metadata fallback: {n_fallback}")
    print(f"  Empty            : {n_empty}")
    print(f"  Output           : {out_path}  ({size_kb:.0f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
