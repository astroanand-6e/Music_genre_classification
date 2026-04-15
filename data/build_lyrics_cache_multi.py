"""
Build lyrics cache using multiple APIs with fallback chain.

Priority order:
  1. LRCLIB      — free, open-source, best for popular tracks
  2. lyrics.ovh  — free, no auth, good romanized Hindi
  3. Genius       — needs GENIUS_TOKEN in .env (optional)
  4. AZLyrics     — scraper fallback, no auth

Extra metadata used to improve search:
  - album/title       → disambiguates same-name tracks
  - track/language_code → helps non-English search
  - track/genre_top   → marks Instrumental/Electronic as "[Instrumental]"

Fully resumable: saves every 200 tracks. Pass --resume to skip already-cached IDs.

Requirements:
    pip install lyricsgenius beautifulsoup4 pandas tqdm

Usage:
    # FMA-Medium (full run, ~2-4 hours):
    python data/build_lyrics_cache_multi.py --dataset medium --resume

    # FMA-Small:
    python data/build_lyrics_cache_multi.py --dataset small

    # Bollywood:
    python data/build_lyrics_cache_multi.py --dataset bollywood

    # Test with 20 tracks:
    python data/build_lyrics_cache_multi.py --dataset medium --limit 20 --debug
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
from tqdm import tqdm

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root))

from data.data_utils import META_DIR

# ── Load .env ─────────────────────────────────────────────────────────────────

_env_path = _root / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ────────────────────────────────────────────────────────────────────

GENIUS_TOKEN       = os.environ.get("GENIUS_TOKEN", "")
LYRICS_MAX_CHARS   = 600
DELAY              = 0.3
BOLLYWOOD_META_CSV = _root / "data" / "bollywood_metadata.csv"

# Genres where lyrics are not expected
INSTRUMENTAL_GENRES = {"Instrumental"}
LIKELY_NO_LYRICS    = {"Instrumental", "Electronic"}

# ── Text helpers ──────────────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r"\[.*?\]")
_SPACE_RE   = re.compile(r"\s+")


def clean_lyrics(raw: str) -> str:
    text = _BRACKET_RE.sub("", raw)
    text = _SPACE_RE.sub(" ", text).strip()
    for phrase in ["Embed", "You might also like", "Lyrics Licensed"]:
        idx = text.rfind(phrase)
        if idx > len(text) * 0.75:
            text = text[:idx].strip()
    return text


def strip_html(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()


def truncate(text: str, max_chars: int = LYRICS_MAX_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


# ── API 1: LRCLIB ────────────────────────────────────────────────────────────

def fetch_lrclib(artist: str, title: str, album: str = "",
                 debug: bool = False) -> str:
    """LRCLIB search — tries exact match first, then fuzzy search."""
    # Try exact match (faster, more accurate)
    params = {"artist_name": artist, "track_name": title}
    if album:
        params["album_name"] = album
    url = f"https://lrclib.net/api/search?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "CALM-MusicGenre/1.0"})
        with urlopen(req, timeout=10) as r:
            results = json.loads(r.read())
        if results:
            lyrics = results[0].get("plainLyrics") or ""
            if lyrics:
                if debug:
                    print(f"    [lrclib] HIT ({len(lyrics)} chars)")
                return clean_lyrics(lyrics)
    except Exception as e:
        if debug:
            print(f"    [lrclib] error: {e}")

    # Retry without album if first attempt had album
    if album:
        params2 = {"artist_name": artist, "track_name": title}
        url2 = f"https://lrclib.net/api/search?{urlencode(params2)}"
        try:
            req = Request(url2, headers={"User-Agent": "CALM-MusicGenre/1.0"})
            with urlopen(req, timeout=10) as r:
                results = json.loads(r.read())
            if results:
                lyrics = results[0].get("plainLyrics") or ""
                if lyrics:
                    if debug:
                        print(f"    [lrclib] HIT on retry ({len(lyrics)} chars)")
                    return clean_lyrics(lyrics)
        except Exception:
            pass

    if debug:
        print(f"    [lrclib] miss")
    return ""


# ── API 2: lyrics.ovh ────────────────────────────────────────────────────────

def fetch_ovh(artist: str, title: str, debug: bool = False) -> str:
    url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(title)}"
    try:
        req = Request(url, headers={"User-Agent": "CALM-MusicGenre/1.0"})
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            lyrics = data.get("lyrics", "")
            if lyrics:
                if debug:
                    print(f"    [lyrics.ovh] HIT ({len(lyrics)} chars)")
                return clean_lyrics(lyrics)
    except Exception as e:
        if debug:
            print(f"    [lyrics.ovh] miss ({e})")
    return ""


# ── API 3: Genius (optional) ─────────────────────────────────────────────────

_genius_client = None


def _get_genius():
    global _genius_client
    if _genius_client is None and GENIUS_TOKEN:
        try:
            import lyricsgenius
            _genius_client = lyricsgenius.Genius(
                GENIUS_TOKEN, timeout=15, retries=1,
                remove_section_headers=False, skip_non_songs=True,
            )
        except ImportError:
            pass
    return _genius_client


def _normalize(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def fetch_genius(artist: str, title: str, debug: bool = False) -> str:
    genius = _get_genius()
    if not genius:
        return ""
    try:
        song = genius.search_song(title, artist, get_full_info=False)
    except Exception as e:
        if debug:
            print(f"    [genius] error: {e}")
        return ""

    if not song or not song.lyrics:
        if debug:
            print(f"    [genius] miss")
        return ""

    # Verify title match to avoid wrong songs
    q = _normalize(title)
    g = _normalize(song.title)
    g = re.sub(r"\s*\(romanized\)|\s*\(translation\)", "", g)
    if q not in g and g not in q:
        if debug:
            print(f"    [genius] REJECTED: '{title}' vs '{song.title}'")
        return ""

    lyrics = clean_lyrics(song.lyrics)
    if debug:
        print(f"    [genius] HIT ({len(lyrics)} chars)")
    return lyrics


# ── API 4: AZLyrics scraper ──────────────────────────────────────────────────

def _azlyrics_url(artist: str, title: str) -> str:
    """Build AZLyrics URL from artist and title."""
    a = re.sub(r"[^a-z0-9]", "", artist.lower())
    if a.startswith("the"):
        a = a[3:]
    t = re.sub(r"[^a-z0-9]", "", title.lower())
    return f"https://www.azlyrics.com/lyrics/{a}/{t}.html"


def fetch_azlyrics(artist: str, title: str, debug: bool = False) -> str:
    url = _azlyrics_url(artist, title)
    try:
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        with urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # AZLyrics stores lyrics in a div with no class/id after the ringtone div
        divs = soup.find_all("div", class_=False, id_=False)
        for div in divs:
            text = div.get_text(strip=True)
            if len(text) > 100 and not text.startswith("if"):
                if debug:
                    print(f"    [azlyrics] HIT ({len(text)} chars)")
                return clean_lyrics(text)
    except Exception as e:
        if debug:
            print(f"    [azlyrics] miss ({e})")
    return ""


# ── Combined fetcher ──────────────────────────────────────────────────────────

def fetch_lyrics(artist: str, title: str, album: str = "",
                 genre: str = "", debug: bool = False) -> str:
    """
    Try all APIs in order. Returns lyrics string or empty string.
    For Instrumental genre, returns "[Instrumental]" if no lyrics found.
    """
    if debug:
        print(f"\n  [{title}] by [{artist}] album=[{album}] genre=[{genre}]")

    # 1. LRCLIB (best quality, includes album matching)
    lyrics = fetch_lrclib(artist, title, album, debug)
    if lyrics:
        return lyrics
    time.sleep(DELAY)

    # 2. lyrics.ovh
    lyrics = fetch_ovh(artist, title, debug)
    if lyrics:
        return lyrics
    time.sleep(DELAY)

    # 3. Genius (if token available)
    if GENIUS_TOKEN:
        lyrics = fetch_genius(artist, title, debug)
        if lyrics:
            return lyrics
        time.sleep(DELAY)

    # 4. AZLyrics (last resort scraper)
    lyrics = fetch_azlyrics(artist, title, debug)
    if lyrics:
        return lyrics
    time.sleep(DELAY)

    # No lyrics found — check if instrumental
    if genre in INSTRUMENTAL_GENRES:
        if debug:
            print(f"    → all missed, genre={genre} → [Instrumental]")
        return "[Instrumental]"

    if debug:
        print(f"    → all missed")
    return ""


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_fma_tracks(subset: str) -> pd.DataFrame:
    tracks = pd.read_csv(
        os.path.join(META_DIR, "tracks.csv"),
        index_col=0, header=[0, 1], low_memory=False,
    )
    subsets = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=subsets, ordered=True))
    except (ValueError, TypeError):
        pass

    df = tracks[tracks[("set", "subset")] <= subset] if subset == "medium" \
         else tracks[tracks[("set", "subset")] == subset]

    flat = pd.DataFrame(index=df.index)
    flat["title"]    = df[("track",  "title")].fillna("").astype(str)
    flat["artist"]   = df[("artist", "name")].fillna("").astype(str)
    flat["album"]    = df[("album",  "title")].fillna("").astype(str)
    flat["genre"]    = df[("track",  "genre_top")].fillna("").astype(str)
    flat["bio"]      = df[("artist", "bio")].fillna("").astype(str)
    flat["language"]  = df[("track",  "language_code")].fillna("").astype(str)
    return flat


def load_bollywood_tracks() -> pd.DataFrame:
    if not BOLLYWOOD_META_CSV.exists():
        print(f"ERROR: {BOLLYWOOD_META_CSV} not found.")
        sys.exit(1)
    df = pd.read_csv(BOLLYWOOD_META_CSV).set_index("filename")
    out = pd.DataFrame(index=df.index)
    out["title"]    = df["song_name"]
    out["artist"]   = df["artist_name"]
    out["album"]    = ""
    out["genre"]    = "Bollywood"
    out["bio"]      = ""
    out["language"]  = ""
    return out


# ── Fallback text ────────────────────────────────────────────────────────────

def metadata_text(artist: str, title: str, bio: str = "") -> str:
    parts = []
    if artist and artist not in ("nan", ""):
        parts.append(f"Artist: {artist}.")
    if bio:
        bio = strip_html(bio)
        if bio:
            bio = bio[:300].rsplit(" ", 1)[0] + "..." if len(bio) > 300 else bio
            parts.append(f"Bio: {bio}.")
    if title and title not in ("nan", ""):
        parts.append(f"Track: {title}.")
    return " ".join(parts) if parts else "Unknown artist."


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build lyrics cache using LRCLIB → lyrics.ovh → Genius → AZLyrics",
    )
    parser.add_argument("--dataset", default="medium",
                        choices=["small", "medium", "bollywood"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    out_path = _root / "data" / f"text_cache_{args.dataset}_lyrics.json"

    # Load tracks
    print(f"Loading {args.dataset} tracks...")
    if args.dataset == "bollywood":
        df = load_bollywood_tracks()
    else:
        df = load_fma_tracks(args.dataset)
    print(f"  Total tracks: {len(df)}")

    # Genre breakdown
    if "genre" in df.columns:
        genre_counts = df["genre"].value_counts()
        instrumental = genre_counts.get("Instrumental", 0)
        electronic = genre_counts.get("Electronic", 0)
        print(f"  Instrumental: {instrumental} | Electronic: {electronic}")

    # APIs available
    apis = ["LRCLIB", "lyrics.ovh"]
    if GENIUS_TOKEN:
        apis.append("Genius")
    apis.append("AZLyrics")
    print(f"  APIs: {' → '.join(apis)}")

    # Resume
    cache = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            cache = json.load(f)
        print(f"  Resuming — already cached: {len(cache)}")

    # Track IDs to process
    track_ids = list(df.index)
    if args.limit:
        track_ids = track_ids[:args.limit]
    total = len(track_ids)

    # Stats
    n_lyrics      = 0
    n_instrumental = 0
    n_fallback    = 0
    n_skipped     = 0
    api_hits      = {"lrclib": 0, "ovh": 0, "genius": 0, "azlyrics": 0}

    print(f"\nProcessing {total} tracks...\n")
    pbar = tqdm(track_ids, desc="Lyrics", unit="track")

    for tid in pbar:
        str_tid = str(tid)

        # Skip if already cached (and has real lyrics or is tagged)
        if str_tid in cache:
            existing = cache[str_tid]
            if existing.startswith("Lyrics:") or existing == "[Instrumental]":
                n_skipped += 1
                pbar.set_postfix(lyrics=n_lyrics, skip=n_skipped, fb=n_fallback)
                continue

        row    = df.loc[tid]
        artist = str(row["artist"]).strip()
        title  = str(row["title"]).strip()
        album  = str(row.get("album", "")).strip()
        genre  = str(row.get("genre", "")).strip()
        bio    = str(row.get("bio", "")).strip()

        if not title or title in ("nan", "None"):
            cache[str_tid] = ""
            n_fallback += 1
            continue

        # Skip known problematic values
        if artist in ("nan", "None", ""):
            artist = ""

        lyrics = fetch_lyrics(artist, title, album, genre, debug=args.debug)

        if lyrics == "[Instrumental]":
            cache[str_tid] = "[Instrumental]"
            n_instrumental += 1
        elif lyrics:
            cache[str_tid] = f"Lyrics: {truncate(lyrics)}"
            n_lyrics += 1
        else:
            bio_clean = strip_html(bio) if bio and bio != "nan" else ""
            cache[str_tid] = metadata_text(artist, title, bio_clean)
            n_fallback += 1

        pbar.set_postfix(
            lyrics=n_lyrics, inst=n_instrumental, fb=n_fallback
        )

        # Periodic save
        processed = n_lyrics + n_instrumental + n_fallback + n_skipped
        if processed % args.save_every == 0 and processed > 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    total_processed = n_lyrics + n_instrumental + n_fallback
    print(f"\n{'='*60}")
    print(f"Done — {args.dataset.upper()}")
    print(f"  Processed         : {total_processed}")
    print(f"  Skipped (cached)  : {n_skipped}")
    print(f"  Lyrics found      : {n_lyrics}  ({100*n_lyrics/max(total_processed,1):.1f}%)")
    print(f"  [Instrumental]    : {n_instrumental}")
    print(f"  Metadata fallback : {n_fallback}")
    print(f"  Output            : {out_path}  ({size_kb:.0f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
