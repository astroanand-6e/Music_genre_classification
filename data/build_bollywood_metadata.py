"""
Build Bollywood dataset metadata CSV
=====================================
Reads ID3 tags from all MP3s in bollywood/Extra Unsorted/,
filters out non-Bollywood (Western/international) tracks,
and writes data/bollywood_metadata.csv.

Usage:
    python data/build_bollywood_metadata.py [--dry_run]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root))

try:
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.mp3 import MP3
except ImportError:
    print("mutagen not installed. Run: pip install mutagen")
    sys.exit(1)

BOLLYWOOD_DIR = _root / "bollywood" / "Extra Unsorted"
OUT_CSV       = _root / "data" / "bollywood_metadata.csv"

# ── Non-Bollywood track filenames (identified by artist metadata + filename) ──
# These are Western/international artists confirmed via ID3 tags.
NON_BOLLYWOOD_FILES = {
    "10. DJ Snake - Magenta Riddim.mp3",                            # DJ Snake (French)
    "DJ Snake Bipolar Sunshine - Middle.mp3",                       # DJ Snake (French)
    "Dj Khaled Ft Justin Bieber - I m the One.mp3",                 # DJ Khaled (US)
    "G-Eazy & Kehlani - Good Life - MP3 320.mp3",                   # G-Eazy (US Hip-Hop)
    "Hailee Steinfeld feat. Zedd - Starving.mp3",                   # Hailee Steinfeld (US)
    "INNA Ft. Marian Hill - Diggy Down.mp3",                        # INNA (Romanian)
    "Jonas Blue ft. William Singe - Mama - MP3 320.mp3",            # Jonas Blue (British)
    "jay_sean_feat._lil_wayne_-_down.mp3",                          # Jay Sean ft Lil Wayne (Western pop)
    "Array - Not Afraid (Explicit Version)(music.naij.com).mp3",    # Eminem (US)
    "68. Logic - 1-800-273-8255.mp3",                               # Logic (US)
    "33526-apologize-timbaland-feat-one-republic--1411574209.mp3",  # Timbaland / One Republic (US)
    "Abdel Kader.mp3",                                              # Algerian Mix (North African)
    "danza kaduro.mp3",                                             # Don Omar (Latin)
}

# Sub-genre mapping based on ID3 genre tags and filenames (best-effort)
# For the report we just want a broad "bollywood" label, but sub-genres are nice to have.
# ── Title / artist cleaners ───────────────────────────────────────────────────

_SITE_RE = re.compile(
    r"\s*[-–]\s*(DJMaza\.\w+|DownloadMing\.\w+|PagalWorld[\.\w]*|SongsMp3\.\w+|"
    r"MixMusic\.\w+|DjPaji\.\w+|Mr-Jatt\.\w+|Mr-Song\.\w+|FreshMaza\.\w+|"
    r"MyMp3Singer\.\w+|Songspkred\.\w+|HeroMaza\.\w+|songsfarm\.\w+|"
    r"www\.\S+|BossMp3\.\w+|SongsLover\.\w+|IndiaMp3\.\w+|WapKing\.\w+|"
    r"SongsPkred\.\w+|RoyalJatt\.\w+|PagalSongs\.\w+)"
    r"|\s*[-–]?\s*\d{2,3}[Kk]bps|\s*\(\d{2,3}\s*[Kk]bps\)"
    r"|\s*\[\d{2,3}[Kk]bps\]",
    re.IGNORECASE,
)
_PARENS_SITE_RE = re.compile(r"\s*\((?:www\.|http)\S+\)", re.IGNORECASE)
_TRACK_NUM_RE   = re.compile(r"^\d{1,3}[\s.\-]+")   # leading "01 - ", "02. " etc.


def clean_song_name(raw_title: str, filename_stem: str) -> str:
    """Return a clean song name suitable for Genius search."""
    # Prefer ID3 title if it looks clean (short, no site suffix)
    t = raw_title.strip()
    if not t or t.lower() in ("nan", "none"):
        t = filename_stem  # fall back to filename
    t = _SITE_RE.sub("", t)
    t = _PARENS_SITE_RE.sub("", t)
    t = _TRACK_NUM_RE.sub("", t)      # strip leading track numbers
    t = t.strip(" -–[]().")
    return t


def clean_artist_name(raw_artist: str) -> str:
    """Return primary artist name suitable for Genius search."""
    a = raw_artist.strip()
    if not a or a.lower() in ("nan", "none"):
        return ""
    # Take only first credited artist
    a = re.split(r"\s*[,&]\s*|\s+ft\.?\s+|\s+feat\.?\s+", a, maxsplit=1)[0]
    # Strip site suffixes that sometimes leak into artist tags
    a = re.sub(r"\s*[-–]\s*(PagalWorld|DJMaza|www\.\S+)\S*", "", a,
               flags=re.IGNORECASE)
    a = a.strip(" -–[]().")
    return a


def infer_subgenre(title: str, artist: str, id3_genre: str, filename: str) -> str:
    combined = (title + artist + id3_genre + filename).lower()
    if any(k in combined for k in ["ghazal", "jagjit", "mehdi hassan", "pankaj udhas"]):
        return "ghazal"
    if any(k in combined for k in ["sufi", "qawwali", "rahat fateh", "nusrat"]):
        return "sufi/qawwali"
    if any(k in combined for k in ["devotional", "bhajan", "aarti", "shiva", "krishna",
                                    "ganesh", "bahubali", "jiyo re"]):
        return "devotional/folk"
    if any(k in combined for k in ["remix", "mashup", "dj ", "party mix", "bend party"]):
        return "remix/mashup"
    if any(k in combined for k in ["nucleya", "edm", "electronic"]):
        return "indian_edm"
    if any(k in combined for k in ["honey singh", "badshah", "raftaar", "dj waley",
                                    "birthday bash", "bom diggy", "mercy", "daru",
                                    "daaru", "thug ranjha", "chhote chhote peg"]):
        return "hindi_hiphop"
    if any(k in combined for k in ["punjabi", "dil chori", "suit suit", "ban ja rani",
                                    "daru badnaam", "prada", "dholebaaziyan", "kala chashma",
                                    "aa soni"]):
        return "punjabi_pop"
    if any(k in combined for k in ["gujarati", "garba", "chogada", "dholida", "navratri"]):
        return "gujarati/garba"
    return "bollywood_pop"


def read_tags(path: Path) -> dict:
    try:
        tags = ID3(path)
        title  = str(tags.get("TIT2", "")).strip()
        artist = str(tags.get("TPE1", "")).strip()
        album  = str(tags.get("TALB", "")).strip()
        genre  = str(tags.get("TCON", "")).strip()
    except (ID3NoHeaderError, Exception):
        title = artist = album = genre = ""

    try:
        audio = MP3(path)
        duration = round(audio.info.length, 1)
    except Exception:
        duration = 0.0

    return {"title": title, "artist": artist, "album": album,
            "id3_genre": genre, "duration_s": duration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print results without writing CSV")
    args = parser.parse_args()

    if not BOLLYWOOD_DIR.exists():
        print(f"ERROR: Bollywood dir not found: {BOLLYWOOD_DIR}")
        sys.exit(1)

    all_mp3s = sorted(BOLLYWOOD_DIR.glob("*.mp3"))
    print(f"Total MP3s found : {len(all_mp3s)}")

    rows = []
    skipped = []

    for mp3 in all_mp3s:
        if mp3.name in NON_BOLLYWOOD_FILES:
            skipped.append(mp3.name)
            continue

        tags = read_tags(mp3)
        song_name   = clean_song_name(tags["title"], mp3.stem)
        artist_name = clean_artist_name(tags["artist"])
        subgenre    = infer_subgenre(
            tags["title"], tags["artist"], tags["id3_genre"], mp3.stem
        )

        rows.append({
            "filename":    mp3.name,
            "filepath":    str(mp3),
            "song_name":   song_name,       # clean — ready for Genius API
            "artist_name": artist_name,     # primary artist only
            "title":       tags["title"],   # raw ID3 title (kept for reference)
            "artist":      tags["artist"],  # raw ID3 artist (kept for reference)
            "album":       tags["album"],
            "id3_genre":   tags["id3_genre"],
            "subgenre":    subgenre,
            "label":       "bollywood",
            "duration_s":  tags["duration_s"],
        })

    print(f"Bollywood tracks : {len(rows)}")
    print(f"Filtered out     : {len(skipped)}")
    print(f"\nFiltered (non-Bollywood):")
    for s in skipped:
        print(f"  - {s}")

    print(f"\nSub-genre breakdown:")
    from collections import Counter
    counts = Counter(r["subgenre"] for r in rows)
    for sg, n in counts.most_common():
        print(f"  {sg:<25} {n}")

    if args.dry_run:
        print("\n[dry_run] No file written.")
        return

    fields = ["filename", "filepath", "song_name", "artist_name",
              "title", "artist", "album", "id3_genre", "subgenre",
              "label", "duration_s"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
