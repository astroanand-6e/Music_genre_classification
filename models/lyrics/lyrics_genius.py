"""
Fetch lyrics from Genius for FMA tracks (artist + title from ``tracks.csv``).

Requires a Genius API access token (free at https://genius.com/api-clients).

Environment
-----------
    export GENIUS_ACCESS_TOKEN="your_token_here"

Output
------
Writes ``results/lyrics_cache_genius.json`` with the same per-track shape as
``lyrics_extractor.py`` (``text``, ``source``, etc.) so ``lyrics_embedder.py``
can consume it via ``--lyrics_cache``.

Usage
-----
    # Smoke test (first 20 tracks)
    python models/lyrics/lyrics_genius.py --subset small --limit 20

    # Full FMA-Small (~8k API calls — respect rate limits; ~1h+ with default sleep)
    python models/lyrics/lyrics_genius.py --subset small

    # Merge Genius hits into the main cache used by default embedder
    python models/lyrics/lyrics_genius.py --subset small --limit 50 --merge_into_main

Notes
-----
* Genius coverage on obscure FMA / indie tracks is partial; misses are stored
  with empty ``text`` for later Whisper backfill.
* The API is rate-limited; use ``--sleep`` (seconds between requests).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata,
    META_DIR,
    RESULTS_DIR,
    BOLLYWOOD_META,
)

CACHE_GENIUS = os.path.join(RESULTS_DIR, "lyrics_cache_genius.json")
CACHE_MAIN = os.path.join(RESULTS_DIR, "lyrics_cache.json")

INSTRUMENTAL_GENRES = {"Instrumental"}
MIN_LYRICS_LEN = 20


def parse_args():
    p = argparse.ArgumentParser(description="Fetch lyrics from Genius API for FMA tracks")
    p.add_argument("--subset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Max tracks to query (for testing). Default: all.")
    p.add_argument("--sleep", type=float, default=0.35,
                   help="Seconds between Genius API calls (rate limiting).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-fetch even if track already in genius cache.")
    p.add_argument(
        "--include_instrumental",
        action="store_true",
        help="Also query Genius for Instrumental genre (default: skip).",
    )
    p.add_argument("--merge_into_main", action="store_true",
                   help="After run, merge genius cache into lyrics_cache.json (non-empty text wins).")
    p.add_argument("--out", default=None, help="Output JSON path (default: results/lyrics_cache_genius.json)")
    return p.parse_args()


def _load_tracks_raw() -> pd.DataFrame:
    path = os.path.join(META_DIR, "tracks.csv")
    return pd.read_csv(path, index_col=0, header=[0, 1])


def load_artist_title_for_metadata(
    df_labels: pd.DataFrame,
    include_bollywood: bool = False,
) -> pd.DataFrame:
    """Artist + title aligned to ``df_labels.index`` (FMA from tracks.csv; Bollywood from metadata CSV)."""
    tracks = _load_tracks_raw()
    bollywood_df = None
    if include_bollywood and os.path.isfile(BOLLYWOOD_META):
        bollywood_df = pd.read_csv(BOLLYWOOD_META, index_col="track_id")

    artists, titles = [], []
    for tid in df_labels.index:
        if tid in tracks.index:
            an = tracks.loc[tid, ("artist", "name")]
            tt = tracks.loc[tid, ("track", "title")]
            artists.append(str(an).strip() if pd.notna(an) else "")
            titles.append(str(tt).strip() if pd.notna(tt) else "")
        elif bollywood_df is not None and tid in bollywood_df.index:
            row = bollywood_df.loc[tid]
            artists.append(str(row.get("artist", "") or "").strip())
            titles.append(str(row.get("title", "") or "").strip())
        else:
            artists.append("")
            titles.append("")

    return pd.DataFrame({"artist": artists, "title": titles}, index=df_labels.index)


def _normalize_lyrics(text: str) -> str:
    if not text:
        return ""
    # lyricsgenius sometimes appends embed markers
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("Embed")]
    return "\n".join(lines).strip()


def fetch_genius_client():
    token = os.environ.get("GENIUS_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Set GENIUS_ACCESS_TOKEN in the environment (https://genius.com/api-clients)."
        )
    import lyricsgenius

    return lyricsgenius.Genius(
        token,
        verbose=False,
        remove_section_headers=True,
        skip_non_songs=True,
    )


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    out_path = args.out or CACHE_GENIUS

    df = load_fma_metadata(subset=args.subset, include_bollywood=args.include_bollywood)
    meta = load_artist_title_for_metadata(df, include_bollywood=args.include_bollywood)

    genius = fetch_genius_client()
    cache = load_json(out_path) if not args.overwrite else {}

    rows = list(meta.iterrows())
    if args.limit is not None:
        rows = rows[: args.limit]

    n_ok = n_miss = n_skip = 0
    for track_id, row in tqdm(rows, desc="Genius"):
        key = str(track_id)
        if key in cache and not args.overwrite:
            if len(cache[key].get("text", "").strip()) >= MIN_LYRICS_LEN:
                continue

        genre = df.loc[track_id, "genre"] if track_id in df.index else ""
        if (not args.include_instrumental) and genre in INSTRUMENTAL_GENRES:
            cache[key] = {
                "text": "",
                "source": "genius_skipped",
                "note": "instrumental_genre",
            }
            n_skip += 1
            continue

        artist, title = row["artist"], row["title"]
        if not title:
            cache[key] = {"text": "", "source": "genius", "note": "missing_title"}
            n_miss += 1
            continue

        try:
            song = genius.search_song(title, artist)
            time.sleep(args.sleep)
        except Exception as e:
            cache[key] = {"text": "", "source": "genius", "error": str(e)[:200]}
            n_miss += 1
            time.sleep(args.sleep)
            continue

        if song and song.lyrics:
            text = _normalize_lyrics(song.lyrics)
            cache[key] = {
                "text": text,
                "source": "genius",
                "genius_url": getattr(song, "url", "") or "",
                "artist": artist,
                "title": title,
            }
            if len(text) >= MIN_LYRICS_LEN:
                n_ok += 1
            else:
                n_miss += 1
        else:
            cache[key] = {
                "text": "",
                "source": "genius",
                "note": "not_found",
                "artist": artist,
                "title": title,
            }
            n_miss += 1

        if len(cache) % 100 == 0:
            save_json(out_path, cache)

    save_json(out_path, cache)
    print(f"\nSaved -> {out_path}")
    print(f"Lyrics found (len>={MIN_LYRICS_LEN}): {n_ok} | misses/empty: {n_miss} | skipped instrumental: {n_skip}")

    if args.merge_into_main:
        main_c = load_json(CACHE_MAIN)
        for k, v in cache.items():
            text = (v.get("text") or "").strip()
            if len(text) < MIN_LYRICS_LEN:
                continue
            prev = (main_c.get(k) or {}).get("text", "").strip()
            if len(prev) < MIN_LYRICS_LEN or args.overwrite:
                main_c[k] = {**main_c.get(k, {}), **v, "source": "genius"}
        save_json(CACHE_MAIN, main_c)
        print(f"Merged into -> {CACHE_MAIN}")


if __name__ == "__main__":
    main()
