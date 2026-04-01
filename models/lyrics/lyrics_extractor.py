"""
Lyrics Extraction Pipeline — Whisper ASR

Transcribes song lyrics from audio using OpenAI Whisper.

For **Genius API** lyrics (recommended first for a quick test on FMA-Small), use
``lyrics_genius.py`` instead — it writes ``results/lyrics_cache_genius.json`` in
the same format this module uses for ``text`` fields.

Output (Whisper)
----------------
``results/lyrics_cache.json`` mapping track_id → ``{ "text": ... }``.

Usage
-----
    # Transcribe FMA-Small (subset="small")
    python models/lyrics/lyrics_extractor.py --subset small

    # Transcribe FMA-Small + Bollywood
    python models/lyrics/lyrics_extractor.py --subset small --include_bollywood

    # Use larger Whisper model for better accuracy
    python models/lyrics/lyrics_extractor.py --subset small --whisper_model medium
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import librosa
import torch
import whisper
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.data_utils import (
    load_fma_metadata, get_splits, get_audio_path,
    load_waveform, RESULTS_DIR, BOLLYWOOD_DIR,
    resolve_audio_path,
)

CACHE_PATH = os.path.join(RESULTS_DIR, "lyrics_cache.json")
WHISPER_SR = 16000  # Whisper always expects 16 kHz

# Genres that are predominantly instrumental — transcription likely to fail
INSTRUMENTAL_GENRES = {"Instrumental"}

# Minimum non-whitespace characters to consider a transcription meaningful
MIN_LYRICS_LEN = 20


def parse_args():
    p = argparse.ArgumentParser(description="Extract lyrics via Whisper ASR")
    p.add_argument("--subset", choices=["small", "medium"], default="small")
    p.add_argument("--include_bollywood", action="store_true")
    p.add_argument("--whisper_model", default="base",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--language", default=None,
                   help="Force language (e.g. 'en', 'hi'). None = auto-detect.")
    p.add_argument("--clip_secs", type=float, default=30.0,
                   help="Seconds of audio to feed Whisper (from the middle of the track)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-transcribe even if cached")
    return p.parse_args()


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def transcribe_audio(model, audio_path: str, clip_secs: float, language=None) -> dict:
    """Load audio, pass to Whisper, return result dict with 'text' and 'language'."""
    try:
        # Load from the middle of the track to avoid silent intros
        duration = librosa.get_duration(path=audio_path)
        start = max(0, duration / 2 - clip_secs / 2)
        y = load_waveform(audio_path, target_sr=WHISPER_SR, duration=clip_secs)
        # Whisper expects a float32 array at 16 kHz
        audio = y.astype(np.float32)
        options = {"language": language} if language else {}
        result = model.transcribe(audio, fp16=torch.cuda.is_available(), **options)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "confidence": float(np.mean([s.get("avg_logprob", 0.0)
                                         for s in result.get("segments", [{"avg_logprob": 0.0}])])),
        }
    except Exception as e:
        return {"text": "", "language": "unknown", "confidence": -99.0, "error": str(e)}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{args.whisper_model}' on {device}...")
    model = whisper.load_model(args.whisper_model, device=device)

    df = load_fma_metadata(subset=args.subset, include_bollywood=args.include_bollywood)
    cache = load_cache()

    # Determine Bollywood audio directory override
    audio_dir_fma = os.path.join(
        os.environ.get("FMA_BASE_DIR", "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI"),
        f"fma_{args.subset}",
    )

    print(f"Dataset: FMA-{args.subset.capitalize()} | "
          f"Tracks: {len(df)} | Bollywood: {args.include_bollywood}")
    print(f"Cache: {CACHE_PATH}")

    processed = 0
    skipped_instrumental = 0
    failed = 0

    for track_id, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        key = str(track_id)
        if key in cache and not args.overwrite:
            continue

        genre = row.get("genre", "")
        if genre in INSTRUMENTAL_GENRES:
            cache[key] = {
                "text": "",
                "language": "none",
                "confidence": 0.0,
                "note": "instrumental_genre_skipped",
            }
            skipped_instrumental += 1
            continue

        audio_path = resolve_audio_path(track_id, row, audio_dir=audio_dir_fma)
        if not os.path.exists(audio_path):
            cache[key] = {"text": "", "language": "unknown", "confidence": -99.0,
                          "error": "file_not_found"}
            failed += 1
            continue

        result = transcribe_audio(model, audio_path, args.clip_secs,
                                  language=args.language)
        cache[key] = result
        processed += 1

        # Periodic save to avoid losing progress
        if processed % 50 == 0:
            save_cache(cache)

    save_cache(cache)

    total = len(df)
    has_lyrics = sum(
        1 for v in cache.values()
        if len(v.get("text", "").strip()) >= MIN_LYRICS_LEN
    )
    print(f"\nDone. Processed: {processed} | Skipped (instrumental): {skipped_instrumental} "
          f"| Failed: {failed}")
    print(f"Tracks with meaningful lyrics (>={MIN_LYRICS_LEN} chars): "
          f"{has_lyrics} / {total}")
    print(f"Cache saved -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
