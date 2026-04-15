"""
Shared data utilities for FMA genre classification experiments.

Supports FMA-Small (8 genres, 8k tracks), FMA-Medium (16 genres, 25k tracks),
and a custom Bollywood dataset that can be merged in as an additional genre.

Single source of truth for metadata loading, audio I/O, dataset construction,
train/val/test splits, and run-result persistence.
"""

import os
import json
import ast
import csv
import glob
import zlib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# __ Default paths (override via environment variables) __
BASE_DIR = os.environ.get(
    "FMA_BASE_DIR",
    "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI",
)
META_DIR = os.environ.get("FMA_META_DIR", os.path.join(BASE_DIR, "fma_metadata"))
# FMA-Small audio tree (default for legacy scripts)
AUDIO_DIR = os.environ.get("FMA_AUDIO_DIR", os.path.join(BASE_DIR, "fma_small"))
# FMA-Medium audio tree (~25k tracks); override with FMA_MEDIUM_AUDIO_DIR if needed
AUDIO_DIR_MEDIUM = os.environ.get(
    "FMA_MEDIUM_AUDIO_DIR", os.path.join(BASE_DIR, "fma_medium")
)
BOLLYWOOD_DIR = os.environ.get("BOLLYWOOD_DIR", os.path.join(BASE_DIR, "bollywood"))
BOLLYWOOD_META = os.environ.get(
    "BOLLYWOOD_META", os.path.join(BASE_DIR, "bollywood", "metadata.csv")
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# FMA-Small has 8 genres; FMA-Medium has 16 (+ optional Bollywood as extra class).
# Use ``len(get_label_maps(load_fma_metadata(subset=...))[0])`` for the active count.
NUM_GENRES = 8
GENRE_NAMES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]


def fma_audio_dir(subset: str = "small") -> str:
    """Return the root directory for FMA audio files.

    Parameters
    ----------
    subset
        ``"small"`` → ``fma_small/`` (or ``FMA_AUDIO_DIR``).
        ``"medium"`` → ``fma_medium/`` (or ``FMA_MEDIUM_AUDIO_DIR``).

    After downloading FMA-Medium, expect ``.../fma_medium/000/000001.mp3`` etc.
    """
    s = subset.lower()
    if s == "medium":
        return AUDIO_DIR_MEDIUM
    return AUDIO_DIR


# __ Metadata __

def load_fma_metadata(
    meta_dir: str = META_DIR,
    subset: str = "small",
    include_bollywood: bool = False,
) -> pd.DataFrame:
    """Load FMA tracks.csv, filter to *subset* (``"small"`` or ``"medium"``),
    and optionally merge Bollywood tracks.

    Returns a DataFrame with columns ``genre`` (str), ``label`` (int), and
    optionally ``audio_path`` (str) for Bollywood rows.  Index = track_id.

    For FMA tracks the audio path is derived from ``get_audio_path``; for
    Bollywood tracks an explicit ``audio_path`` column is set from the
    Bollywood metadata CSV.
    """
    tracks = pd.read_csv(os.path.join(meta_dir, "tracks.csv"),
                         index_col=0, header=[0, 1])

    subsets = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            "category", categories=subsets, ordered=True)
    except (ValueError, TypeError):
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=subsets, ordered=True))

    # For "medium" we include small tracks too (medium ⊇ small in FMA)
    if subset == "medium":
        fma_df = tracks[tracks[("set", "subset")] <= "medium"]
    else:
        fma_df = tracks[tracks[("set", "subset")] == subset]

    df = fma_df[[("track", "genre_top")]].copy()
    df.columns = ["genre"]
    df = df.dropna()
    df["audio_path"] = None  # filled at access time for FMA tracks

    if include_bollywood and os.path.exists(BOLLYWOOD_META):
        bdf = _load_bollywood_metadata()
        df = pd.concat([df, bdf], axis=0)

    genre_names = sorted(df["genre"].unique())
    label2id = {g: i for i, g in enumerate(genre_names)}
    df["label"] = df["genre"].map(label2id)
    return df


def _load_bollywood_metadata() -> pd.DataFrame:
    """Load the Bollywood metadata CSV and return in the same format as FMA df.

    Expected CSV columns: track_id, title, audio_path (absolute path to .mp3)
    Genre is always ``"Bollywood"``.
    """
    bdf = pd.read_csv(BOLLYWOOD_META, index_col="track_id")
    bdf["genre"] = "Bollywood"
    # keep only the columns we need
    keep = ["genre", "audio_path"] if "audio_path" in bdf.columns else ["genre"]
    bdf = bdf[keep].copy()
    if "audio_path" not in bdf.columns:
        bdf["audio_path"] = None
    return bdf


def get_label_maps(df: pd.DataFrame):
    """Return (label2id, id2label) dicts from a metadata DataFrame."""
    genre_names = sorted(df["genre"].unique())
    label2id = {g: i for i, g in enumerate(genre_names)}
    id2label = {i: g for g, i in label2id.items()}
    return label2id, id2label


# __ Splits __

def get_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train / val / test split.  Same random_state everywhere
    guarantees identical partitions across all model scripts.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state,
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label"],
        random_state=random_state,
    )
    return train_df, val_df, test_df


# __ Audio I/O __

def get_audio_path(track_id: int, audio_dir: str = AUDIO_DIR) -> str:
    """Derive the FMA audio file path from a numeric track_id."""
    tid = f"{track_id:06d}"
    return os.path.join(audio_dir, tid[:3], f"{tid}.mp3")


def resolve_audio_path(track_id, row: pd.Series, audio_dir: str = AUDIO_DIR) -> str:
    """Return the audio path for a row, handling both FMA and Bollywood tracks.

    For Bollywood tracks the path is stored in ``row["audio_path"]``.
    For FMA tracks it is derived from the numeric track_id.

    Prefers a pre-converted .wav file (same path, .wav extension) over the
    original .mp3 when it exists — avoids libmpg123 decoding errors and is
    faster to load (no decompression overhead).
    """
    if pd.notna(row.get("audio_path")):
        return row["audio_path"]
    mp3_path = get_audio_path(int(track_id), audio_dir)
    wav_path = mp3_path[:-4] + ".wav"
    if os.path.exists(wav_path):
        return wav_path
    return mp3_path


def load_waveform(
    path: str,
    target_sr: int = 16000,
    duration: float = 10.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Load an audio file and return a mono numpy array at *target_sr*.

    Uses librosa for robust MP3 decoding.  Returns shape ``(samples,)``.
    ``offset`` is the start time in seconds (for random temporal crops).
    Raises ``FileNotFoundError`` / ``RuntimeError`` on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, _ = librosa.load(
        path, sr=target_sr, offset=offset, duration=duration, mono=True,
    )
    return y


# __ Temporal sampling (train vs val/test, multi-crop eval) __

def temporal_offset_seed(track_id, split_tag: str) -> int:
    """Stable 32-bit seed from track id and split name (not Python's salted ``hash``)."""
    key = f"{track_id}:{split_tag}".encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


def segment_offset_for_mode(
    path: str,
    track_id,
    clip_secs: float,
    split_tag: str,
    mode: str,
) -> float:
    """Start offset in seconds for one ``clip_secs`` window from *path*.

    *mode*
        ``"start"`` — always from t=0.
        ``"random"`` — uniform random in [0, max(0, duration - clip_secs)].
        ``"deterministic"`` — reproducible offset from ``track_id`` + ``split_tag``
        so train / val / test usually see different parts of the same file.
    """
    if mode == "start":
        return 0.0
    try:
        file_dur = librosa.get_duration(path=path)
    except Exception:
        return 0.0
    max_off = max(0.0, file_dur - clip_secs)
    if max_off <= 0:
        return 0.0
    if mode == "random":
        return float(np.random.uniform(0.0, max_off))
    if mode == "deterministic":
        rng = np.random.RandomState(temporal_offset_seed(track_id, split_tag))
        return float(rng.uniform(0.0, max_off))
    return 0.0


def tiled_segment_offsets(
    path: str,
    clip_secs: float,
    hop_frac: float = 0.5,
    max_segments: int = 24,
) -> list[float]:
    """List of start offsets (seconds) covering the file with overlapping clips.

    Used for full-track test evaluation: average predictions over windows.
    ``hop_frac`` is hop length as a fraction of ``clip_secs`` (minimum 0.25 s hop).
    """
    try:
        file_dur = librosa.get_duration(path=path)
    except Exception:
        file_dur = 0.0
    if file_dur <= clip_secs + 1e-6:
        return [0.0]
    hop_secs = max(clip_secs * float(hop_frac), 0.25)
    out: list[float] = []
    t = 0.0
    while t + clip_secs <= file_dur + 1e-6 and len(out) < max_segments:
        out.append(t)
        t += hop_secs
    last = max(0.0, file_dur - clip_secs)
    if len(out) < max_segments and (not out or abs(out[-1] - last) > 1e-2):
        out.append(last)
    return out if out else [0.0]


def add_temporal_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register shared flags for clip selection and optional full-track test eval."""
    parser.add_argument(
        "--temporal_train",
        choices=("start", "random", "deterministic"),
        default="random",
        help="Train clip: file start, random window each epoch, or stable hash-based window.",
    )
    parser.add_argument(
        "--temporal_eval",
        choices=("start", "random", "deterministic"),
        default="deterministic",
        help="Val/test clip (default: deterministic, different from train and val vs test).",
    )
    parser.add_argument(
        "--test_multi_crop",
        action="store_true",
        help="Final test: average logits over tiled windows spanning each full track.",
    )
    parser.add_argument(
        "--test_crop_hop_frac",
        type=float,
        default=0.5,
        help="Multi-crop hop as fraction of clip_secs (with --test_multi_crop).",
    )
    parser.add_argument(
        "--test_max_segments",
        type=int,
        default=24,
        help="Max windows per track for --test_multi_crop.",
    )
    return parser


def add_finetune_callback_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Early stopping and related fine-tuning controls (use on all finetune CLIs)."""
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many epochs without val accuracy improvement (0 = disabled).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum val accuracy increase to count as improvement for early stopping.",
    )
    return parser


def add_device_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Train on GPU when possible, or force / require a backend."""
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: CUDA if available else CPU; cuda: exit if no GPU; cpu: force CPU.",
    )
    return parser


def resolve_training_device(device_choice: str):
    """Return a ``torch.device`` and print which backend is used (with hints if CPU-only)."""
    import torch

    choice = (device_choice or "auto").lower()
    if choice == "cpu":
        dev = torch.device("cpu")
        print(f"Device: {dev} (forced via --device cpu)")
        return dev

    if choice == "cuda":
        if not torch.cuda.is_available():
            _print_cuda_troubleshoot(torch)
            raise SystemExit(
                "Exiting: --device cuda but torch.cuda.is_available() is False."
            )
        dev = torch.device("cuda")
        idx = torch.cuda.current_device()
        print(f"Device: cuda:{idx} — {torch.cuda.get_device_name(idx)}")
        return dev

    # auto
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        idx = torch.cuda.current_device()
        print(f"Device: cuda:{idx} — {torch.cuda.get_device_name(idx)}")
        return dev

    print("Device: cpu (CUDA not available — training will be very slow).")
    _print_cuda_troubleshoot(torch, brief=True)
    return torch.device("cpu")


def _print_cuda_troubleshoot(torch_mod, brief: bool = False) -> None:
    print("\n--- CUDA check ---")
    print(f"torch.__version__={torch_mod.__version__}")
    cv = getattr(torch_mod.version, "cuda", None)
    print(f"torch.version.cuda={cv!r}  (None usually means CPU-only PyTorch build)")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    print(f"CUDA_VISIBLE_DEVICES={vis}")
    if brief:
        print("Fix: install a CUDA build from https://pytorch.org/get-started/locally/")
        print("     e.g. pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("     Run: nvidia-smi   (driver + visible GPUs)")
    else:
        print("Run `nvidia-smi`. If it fails, fix the NVIDIA driver or use a GPU machine.")
        print("If `nvidia-smi` works but PyTorch shows cuda=None, reinstall torch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("Do not set CUDA_VISIBLE_DEVICES to an empty string (hides all GPUs).")
    print("------------------\n")


def new_finetune_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def finetune_epoch_csv_path(
    model: str, variant: str, dataset_tag: str, run_id: str,
) -> str:
    """Per-run CSV path under ``results/logs/finetune/`` for epoch metrics."""
    log_dir = os.path.join(RESULTS_DIR, "logs", "finetune")
    _ensure_dir(log_dir)
    v = f"_{variant}" if variant else ""
    return os.path.join(log_dir, f"{model}{v}_{dataset_tag}_{run_id}.csv")


_FINITEPOCH_FIELDS = (
    "run_id", "epoch", "train_loss", "val_loss", "val_acc", "best_val_acc", "lr_max",
)


def append_finetune_epoch_log(csv_path: str, row: dict) -> None:
    """Append one epoch row to CSV (writes header on first line)."""
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FINITEPOCH_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _FINITEPOCH_FIELDS})


def max_optimizer_lr(optimizer) -> float:
    return max(float(g["lr"]) for g in optimizer.param_groups)


def print_test_classification_report(y_true, y_pred, id2label: dict) -> None:
    names = [id2label[i] for i in sorted(id2label)]
    print(classification_report(y_true, y_pred, target_names=names, digits=3, zero_division=0))


def save_genre_confusion_matrix_png(
    y_true,
    y_pred,
    id2label: dict,
    save_path: str,
    title: str | None = None,
) -> None:
    """Save row-normalised (%) genre confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    genre_names = [id2label[i] for i in sorted(id2label)]
    cm = confusion_matrix(y_true, y_pred)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(genre_names)), max(8, len(genre_names) * 0.6)))
    try:
        import seaborn as sns
        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=genre_names,
            yticklabels=genre_names,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Row %"},
            ax=ax,
        )
    except ImportError:
        im = ax.imshow(cm_pct, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(genre_names)))
        ax.set_yticks(range(len(genre_names)))
        ax.set_xticklabels(genre_names, rotation=45, ha="right")
        ax.set_yticklabels(genre_names)
        fig.colorbar(im, ax=ax, label="Row %")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row %)")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")


# __ Dataset __

class FMADataset(Dataset):
    """Generic FMA / Bollywood dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``label`` column and track_id as index.
        May optionally have ``audio_path`` column for non-FMA tracks.
    audio_dir : str
        Root of fma_small / fma_medium audio tree (ignored for rows that have
        an explicit ``audio_path``).
    preprocess_fn : callable(waveform_np, sr) -> dict
        Model-specific preprocessing.  Must return a dict whose values are
        tensors (the dict is collated by the default DataLoader collator).
    target_sr : int
    clip_secs : float
    temporal_sampling : str
        One of ``"start"``, ``"random"``, ``"deterministic"`` — see
        ``segment_offset_for_mode``.
    split_tag : str
        ``"train"``, ``"val"``, or ``"test"`` — used with ``deterministic`` mode.
    """

    def __init__(
        self,
        df,
        audio_dir,
        preprocess_fn,
        target_sr,
        clip_secs,
        temporal_sampling: str = "start",
        split_tag: str = "train",
    ):
        self.track_ids = df.index.tolist()
        self.labels = df["label"].tolist()
        self.audio_paths = (
            df["audio_path"].tolist()
            if "audio_path" in df.columns
            else [None] * len(df)
        )
        self.audio_dir = audio_dir
        self.preprocess_fn = preprocess_fn
        self.target_sr = target_sr
        self.clip_secs = clip_secs
        self.temporal_sampling = temporal_sampling
        self.split_tag = split_tag

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        label = self.labels[idx]
        explicit_path = self.audio_paths[idx]

        if pd.notna(explicit_path) if explicit_path is not None else False:
            path = explicit_path
        else:
            mp3_path = get_audio_path(int(track_id), self.audio_dir)
            wav_path = mp3_path[:-4] + ".wav"
            path = wav_path if os.path.exists(wav_path) else mp3_path

        try:
            offset = segment_offset_for_mode(
                path, track_id, self.clip_secs, self.split_tag, self.temporal_sampling,
            )
            waveform = load_waveform(
                path, self.target_sr, self.clip_secs, offset=offset,
            )
            max_len = int(self.target_sr * self.clip_secs)
            if len(waveform) > max_len:
                waveform = waveform[:max_len]
            elif len(waveform) < max_len:
                waveform = np.pad(waveform, (0, max_len - len(waveform)))
            result = self.preprocess_fn(waveform, self.target_sr)
        except Exception:
            result = self.preprocess_fn(
                np.zeros(int(self.target_sr * self.clip_secs), dtype=np.float32),
                self.target_sr,
            )

        result["labels"] = torch.tensor(label, dtype=torch.long)
        return result


def build_dataloaders(
    train_df, val_df, test_df,
    preprocess_fn,
    target_sr: int,
    clip_secs: float,
    batch_size: int = 16,
    num_workers: int = 4,
    audio_dir: str = AUDIO_DIR,
    train_temporal_sampling: str = "random",
    eval_temporal_sampling: str = "deterministic",
):
    """Build train / val / test DataLoaders from split DataFrames.

    ``train_temporal_sampling`` / ``eval_temporal_sampling`` control how each
    clip is chosen (see ``segment_offset_for_mode``). Val and test both use
    *eval_temporal_sampling* but different ``split_tag`` so deterministic
    offsets differ between splits.
    """
    kw = dict(audio_dir=audio_dir, preprocess_fn=preprocess_fn,
              target_sr=target_sr, clip_secs=clip_secs)

    train_ds = FMADataset(
        train_df, **kw,
        temporal_sampling=train_temporal_sampling, split_tag="train",
    )
    val_ds = FMADataset(
        val_df, **kw,
        temporal_sampling=eval_temporal_sampling, split_tag="val",
    )
    test_ds = FMADataset(
        test_df, **kw,
        temporal_sampling=eval_temporal_sampling, split_tag="test",
    )

    loader_kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader


# __ Run-result persistence __

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_run_results(
    *,
    model: str,
    variant: str,
    mode: str,
    test_accuracy: float,
    best_val_accuracy: float | None = None,
    config: dict | None = None,
    per_class_f1: dict | None = None,
    extra: dict | None = None,
    dataset: str = "fma_small",
):
    """Write a structured JSON to ``results/runs/``."""
    now = datetime.now()
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_iso = now.isoformat()

    slug = f"{model}_{variant}_{mode}_{ts_file}" if variant else f"{model}_{mode}_{ts_file}"
    if dataset != "fma_small":
        slug = f"{slug}_{dataset}"
    runs_dir = os.path.join(RESULTS_DIR, "runs")
    _ensure_dir(runs_dir)

    record = {
        "model": model,
        "variant": variant,
        "mode": mode,
        "dataset": dataset,
        "timestamp": ts_iso,
        "test_accuracy": round(test_accuracy, 4),
        "best_val_accuracy": round(best_val_accuracy, 4) if best_val_accuracy is not None else None,
        "config": config or {},
        "per_class_f1": per_class_f1 or {},
    }
    if extra:
        record.update(extra)

    path = os.path.join(runs_dir, f"{slug}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Run results saved -> {path}")
    return path


def load_all_run_results() -> pd.DataFrame:
    """Load every JSON in ``results/runs/`` into a DataFrame."""
    runs_dir = os.path.join(RESULTS_DIR, "runs")
    if not os.path.isdir(runs_dir):
        return pd.DataFrame()
    records = []
    for fp in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(fp) as f:
            records.append(json.load(f))
    return pd.DataFrame(records)


def checkpoint_path(model: str, variant: str = "", dataset: str = "fma_small") -> str:
    """Return a timestamped checkpoint file path and ensure the dir exists."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{model}_{variant}_{ts}" if variant else f"{model}_{ts}"
    if dataset != "fma_small":
        slug = f"{slug}_{dataset}"
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints", model)
    _ensure_dir(ckpt_dir)
    return os.path.join(ckpt_dir, f"{slug}.pt")


def figures_path(filename: str) -> str:
    """Return a path inside ``results/figures/`` and ensure the dir exists."""
    d = os.path.join(RESULTS_DIR, "figures")
    _ensure_dir(d)
    return os.path.join(d, filename)


def compute_per_class_f1(y_true, y_pred, id2label: dict) -> dict:
    """Return {genre_name: f1} dict."""
    target_names = [id2label[i] for i in sorted(id2label)]
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0,
    )
    return {name: round(report[name]["f1-score"], 4) for name in target_names}
