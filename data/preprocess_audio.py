"""
Audio pre-processing pipeline for FMA (Small/Medium) and Bollywood datasets.

Converts every MP3 track to a 16 kHz mono WAV file, mirroring the original
directory structure under a new root (e.g. fma_small_wav/, fma_medium_wav/,
bollywood_wav/).  Subsequent training and zero-shot scripts can load WAVs with
soundfile (no libmpg123 fallback, no decoding errors, faster I/O).

Usage
-----
# Convert FMA-Small (default):
python data/preprocess_audio.py --dataset small

# Convert FMA-Medium (takes ~1–2 hours, output ~24 GB):
python data/preprocess_audio.py --dataset medium --workers 8

# Convert Bollywood songs:
python data/preprocess_audio.py --dataset bollywood

# Convert everything:
python data/preprocess_audio.py --dataset all --workers 8

# Dry run — lists what would be converted without writing any files:
python data/preprocess_audio.py --dataset small --dry_run

Output
------
Each converted WAV is saved alongside its source MP3 (same dir, .wav extension).
A manifest CSV is written to data/manifest_<dataset>_wav.csv with columns:
    track_id, src_path, wav_path, status   (status: ok | error | skipped)

Environment variables
---------------------
FMA_BASE_DIR       — project root (default: repo root)
FMA_AUDIO_DIR      — FMA-Small MP3 directory
FMA_MEDIUM_AUDIO_DIR — FMA-Medium MP3 directory
BOLLYWOOD_DIR      — Bollywood MP3 directory
"""

import os
import sys
import csv
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_project_root = _here.parent
sys.path.insert(0, str(_project_root))

from data.data_utils import (
    fma_audio_dir,
    load_fma_metadata,
    resolve_audio_path,
    BOLLYWOOD_DIR,
    BOLLYWOOD_META,
    BASE_DIR,
)

TARGET_SR = 16_000  # Hz — matches most models (AST, Conformer, MusicLDM, Gemma)
# CLAP LAION uses 48 kHz; CLAP Microsoft uses 44.1 kHz — those models resample
# from 16 kHz internally, so 16 kHz is a safe universal intermediate.


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _convert_one(args):
    """Convert a single MP3 → WAV.  Returns (track_id, src, wav, status)."""
    track_id, src_path, wav_path, clip_secs, overwrite = args

    if not overwrite and Path(wav_path).exists():
        return (track_id, src_path, wav_path, "skipped")

    try:
        dur = clip_secs if clip_secs > 0 else None
        waveform, _ = librosa.load(src_path, sr=TARGET_SR, mono=True, duration=dur)

        if clip_secs > 0:
            target_len = int(clip_secs * TARGET_SR)
            if len(waveform) < target_len:
                waveform = np.pad(waveform, (0, target_len - len(waveform)))
            else:
                waveform = waveform[:target_len]

        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, waveform, TARGET_SR, subtype="PCM_16")
        return (track_id, src_path, wav_path, "ok")

    except Exception as e:
        return (track_id, src_path, wav_path, f"error: {e}")


# ── Collection helpers ────────────────────────────────────────────────────────

def collect_fma_jobs(subset: str, clip_secs: float, overwrite: bool):
    """Return list of (track_id, src_mp3, dst_wav, clip_secs, overwrite) for FMA."""
    audio_dir = Path(fma_audio_dir(subset))
    df = load_fma_metadata(subset=subset)

    jobs = []
    missing = 0
    for track_id, row in df.iterrows():
        src = resolve_audio_path(track_id, row, audio_dir=str(audio_dir))
        if not Path(src).exists():
            missing += 1
            continue
        wav = str(Path(src).with_suffix(".wav"))
        jobs.append((track_id, src, wav, clip_secs, overwrite))

    if missing:
        print(f"  [{subset}] {missing} tracks not found on disk — skipped.")
    return jobs


def collect_bollywood_jobs(clip_secs: float, overwrite: bool):
    """Return jobs for all audio files under BOLLYWOOD_DIR."""
    bdir = Path(BOLLYWOOD_DIR)
    if not bdir.exists():
        print(f"  Bollywood dir not found: {bdir}")
        return []

    jobs = []
    for src in sorted(bdir.rglob("*.mp3")):
        wav = src.with_suffix(".wav")
        # Use filename stem as track_id
        jobs.append((src.stem, str(src), str(wav), clip_secs, overwrite))

    # Also pick up .m4a and .flac in case user has mixed formats
    for ext in ("*.m4a", "*.flac", "*.ogg"):
        for src in sorted(bdir.rglob(ext)):
            wav = src.with_suffix(".wav")
            jobs.append((src.stem, str(src), str(wav), clip_secs, overwrite))

    return jobs


# ── Manifest writer ───────────────────────────────────────────────────────────

def write_manifest(records, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "src_path", "wav_path", "status"])
        writer.writerows(records)
    print(f"  Manifest saved → {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    datasets = (
        ["small", "medium", "bollywood"] if args.dataset == "all"
        else [args.dataset]
    )

    data_dir = _project_root / "data"
    all_records = []

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*60}")

        if ds in ("small", "medium"):
            jobs = collect_fma_jobs(ds, args.clip_secs, args.overwrite)
        else:
            jobs = collect_bollywood_jobs(args.clip_secs, args.overwrite)

        if not jobs:
            print("  No jobs found.")
            continue

        already = sum(1 for j in jobs if Path(j[2]).exists() and not args.overwrite)
        todo = len(jobs) - already
        print(f"  Total tracks : {len(jobs)}")
        print(f"  Already done : {already} (use --overwrite to redo)")
        print(f"  To convert   : {todo}")
        print(f"  Workers      : {args.workers}")
        print(f"  Clip secs    : {'full track' if args.clip_secs <= 0 else f'{args.clip_secs}s'}")
        print(f"  Target SR    : {TARGET_SR} Hz")

        if args.dry_run:
            print("  [DRY RUN] No files written.")
            continue

        if todo == 0:
            print("  Nothing to do.")
            records = [(j[0], j[1], j[2], "skipped") for j in jobs]
        elif args.workers == 1:
            records = []
            for job in tqdm(jobs, desc=f"Converting {ds}", dynamic_ncols=True):
                records.append(_convert_one(job))
        else:
            with Pool(processes=args.workers) as pool:
                records = list(tqdm(
                    pool.imap_unordered(_convert_one, jobs, chunksize=16),
                    total=len(jobs),
                    desc=f"Converting {ds}",
                    dynamic_ncols=True,
                ))

        # Summary
        ok = sum(1 for r in records if r[3] == "ok")
        skipped = sum(1 for r in records if r[3] == "skipped")
        errors = [r for r in records if r[3].startswith("error")]
        print(f"\n  Converted : {ok}")
        print(f"  Skipped   : {skipped}")
        print(f"  Errors    : {len(errors)}")
        if errors:
            print("  First 10 errors:")
            for r in errors[:10]:
                print(f"    [{r[0]}] {r[1]}: {r[3]}")

        manifest_path = data_dir / f"manifest_{ds}_wav.csv"
        write_manifest(records, manifest_path)
        all_records.extend(records)

    print(f"\n{'='*60}")
    print("Done.")
    total_ok = sum(1 for r in all_records if r[3] == "ok")
    total_err = sum(1 for r in all_records if r[3].startswith("error"))
    print(f"Total converted: {total_ok}  |  Errors: {total_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process FMA / Bollywood audio to 16 kHz mono WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="small",
        choices=["small", "medium", "bollywood", "all"],
        help="Which dataset to process",
    )
    parser.add_argument(
        "--clip_secs", type=float, default=30.0,
        help="Seconds to keep per track (0 = full track, no clipping)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, cpu_count() - 1),
        help="Parallel worker processes",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-convert files that already have a .wav",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List what would be done without writing any files",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
