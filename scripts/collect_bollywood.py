"""
Bollywood Dataset Collection Script

Downloads ~30 Bollywood songs from YouTube using yt-dlp, extracts a 30-second
clip from each, and creates a metadata CSV compatible with the FMA data pipeline.

If you **already have** audio files (MP3/WAV), skip yt-dlp: place clips under
``bollywood/audio/``, then create ``bollywood/metadata.csv`` with columns
``track_id,title,audio_path`` (and optional ``artist`` for Genius). ``audio_path``
must be an absolute or project-relative path to each file.

Usage
-----
    python scripts/collect_bollywood.py
    python scripts/collect_bollywood.py --output_dir /path/to/bollywood --clip_sec 30 --clip_start 30

The output directory will contain:
    bollywood/
        audio/          # 30-second MP3 clips
        metadata.csv    # track_id, title, audio_path, genre
"""

import os
import sys
import csv
import argparse
import subprocess
import tempfile
import shutil
import json

import librosa
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

# Curated list of 30 iconic Bollywood songs with YouTube IDs.
# Covers a range of eras (1990s–2020s) and sub-styles (romantic, dance, folk fusion).
BOLLYWOOD_TRACKS = [
    # (youtube_id, title)
    ("3ySPAANzkr4", "Tum Hi Ho - Aashiqui 2"),
    ("YQHsXMglC9A", "Jai Ho - Slumdog Millionaire"),
    ("_VxhDqH_N2s", "Balam Pichkari - Yeh Jawaani Hai Deewani"),
    ("kz5mttQGnN4", "Dil Dhadakne Do Title Track"),
    ("q68wsVJLQuk", "Ghagra - Yeh Jawaani Hai Deewani"),
    ("_PCaP7eAWU8", "Badtameez Dil - Yeh Jawaani Hai Deewani"),
    ("QvKyBqp5UYM", "London Thumakda - Queen"),
    ("N7Xm5eomaSg", "Senorita - Zindagi Na Milegi Dobara"),
    ("b6ATN-kMVGo", "Nagada Sang Dhol - Goliyon Ki Raasleela"),
    ("7nh7xFHD65A", "Malhari - Bajirao Mastani"),
    ("mOt4JiD-LhA", "Galliyan - Ek Villain"),
    ("k4oua9TgPM4", "Abhi Mujh Mein Kahin - Agneepath"),
    ("0UDKjn9wDO4", "Yeh Ishq Hai - Jab We Met"),
    ("b7AVXCN_h7Y", "Dard-E-Disco - Om Shanti Om"),
    ("i2EUJjb-wKM", "Rang De Basanti Title"),
    ("OkQ2Q9IVKX0", "Zinda - Bhaag Milkha Bhaag"),
    ("SYt-lECBMD4", "Dil Chahta Hai Title"),
    ("J4WXxBv7hIc", "Kal Ho Naa Ho - Title Song"),
    ("1R4kXM84X44", "Tujhe Bhula Diya - Anjaana Anjaani"),
    ("hJhJvWgHM_U", "Ainvayi Ainvayi - Band Baaja Baaraat"),
    ("4E_kz_vBiVI", "Rowdy Rathore - Chinta Ta Ta"),
    ("g1eAtEVP-XM", "Gerua - Dilwale"),
    ("2OaKVVpCRVs", "Mast Magan - 2 States"),
    ("zCsG6F-NCWU", "Kabira - Yeh Jawaani Hai Deewani"),
    ("Ir18HvSuVXA", "Nashe Si Chadh Gayi - Befikre"),
    ("U4Y6FeSsTe0", "Lungi Dance - Chennai Express"),
    ("l_MyUGq7pgs", "Radha - SOTY"),
    ("U62W-kufM7U", "Banjaara - Ek Villain"),
    ("r3wGlW9OMOU", "Subhanallah - Yeh Jawaani Hai Deewani"),
    ("jz5HOCNMWRE", "Pareshaan - Ishaqzaade"),
]

TRACK_ID_OFFSET = 900000  # use IDs starting from 900000 to avoid FMA collision


def parse_args():
    p = argparse.ArgumentParser(description="Collect Bollywood audio dataset")
    p.add_argument("--output_dir", default=os.path.join(BASE_DIR, "bollywood"))
    p.add_argument("--clip_sec", type=float, default=30.0,
                   help="Length of extracted clip in seconds")
    p.add_argument("--clip_start", type=float, default=30.0,
                   help="Start offset (seconds) for clip extraction — skip intros")
    p.add_argument("--target_sr", type=int, default=22050)
    return p.parse_args()


def download_audio(youtube_id: str, out_path: str) -> bool:
    """Download audio from YouTube using yt-dlp. Returns True on success."""
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={youtube_id}",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",      # ~128kbps, good enough for classification
        "-o", out_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0


def extract_clip(src_path: str, dst_path: str, start: float, duration: float, sr: int) -> bool:
    """Load audio, extract a fixed-length clip, save as MP3. Returns True on success."""
    try:
        y, orig_sr = librosa.load(src_path, sr=sr, mono=True)
        start_sample = int(start * sr)
        end_sample = int((start + duration) * sr)

        if start_sample >= len(y):
            start_sample = 0
            end_sample = int(duration * sr)

        clip = y[start_sample:end_sample]
        if len(clip) < int(duration * sr):
            clip = np.pad(clip, (0, int(duration * sr) - len(clip)))

        sf.write(dst_path.replace(".mp3", ".wav"), clip, sr, format="WAV")
        # Convert WAV → MP3 with ffmpeg (lighter than installing pydub)
        subprocess.run(
            ["ffmpeg", "-i", dst_path.replace(".mp3", ".wav"),
             "-q:a", "5", dst_path, "-y", "-loglevel", "quiet"],
            check=True,
        )
        os.remove(dst_path.replace(".mp3", ".wav"))
        return True
    except Exception as e:
        print(f"  Clip extraction failed: {e}")
        return False


def main():
    args = parse_args()
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    meta_path = os.path.join(args.output_dir, "metadata.csv")
    rows = []

    print(f"Collecting {len(BOLLYWOOD_TRACKS)} Bollywood tracks...")
    print(f"Output dir : {args.output_dir}")
    print(f"Clip       : {args.clip_start}s → {args.clip_start + args.clip_sec}s "
          f"({args.clip_sec}s)")
    print()

    for i, (yt_id, title) in enumerate(BOLLYWOOD_TRACKS):
        track_id = TRACK_ID_OFFSET + i
        clip_path = os.path.join(audio_dir, f"{track_id}.mp3")

        if os.path.exists(clip_path):
            print(f"[{i+1:02d}/{len(BOLLYWOOD_TRACKS)}] SKIP (exists): {title}")
            rows.append({
                "track_id": track_id,
                "title": title,
                "youtube_id": yt_id,
                "audio_path": clip_path,
                "genre": "Bollywood",
            })
            continue

        print(f"[{i+1:02d}/{len(BOLLYWOOD_TRACKS)}] Downloading: {title}")

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, f"{yt_id}.mp3")
            success = download_audio(yt_id, raw_path)

            if not success or not os.path.exists(raw_path):
                print(f"  FAILED to download {yt_id} — skipping")
                continue

            ok = extract_clip(raw_path, clip_path, args.clip_start, args.clip_sec, args.target_sr)
            if ok:
                print(f"  Saved clip -> {clip_path}")
                rows.append({
                    "track_id": track_id,
                    "title": title,
                    "youtube_id": yt_id,
                    "audio_path": clip_path,
                    "genre": "Bollywood",
                })
            else:
                print(f"  FAILED clip extraction for {title}")

    if rows:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "title", "youtube_id",
                                                    "audio_path", "genre"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nMetadata saved -> {meta_path}")
        print(f"Successfully collected {len(rows)}/{len(BOLLYWOOD_TRACKS)} tracks.")
    else:
        print("\nNo tracks collected successfully.")


if __name__ == "__main__":
    main()
