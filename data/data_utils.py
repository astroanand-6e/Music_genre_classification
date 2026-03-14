"""
Shared data utilities for FMA-Small genre classification experiments.

Single source of truth for metadata loading, audio I/O, dataset construction,
train/val/test splits, and run-result persistence.
"""

import os
import json
import ast
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# __ Default paths (override via environment variables) __
BASE_DIR = os.environ.get(
    "FMA_BASE_DIR",
    "/home/anand_dev/STUDY/NU/spring26/CS5100_FAI",
)
META_DIR = os.environ.get("FMA_META_DIR", os.path.join(BASE_DIR, "fma_metadata"))
AUDIO_DIR = os.environ.get("FMA_AUDIO_DIR", os.path.join(BASE_DIR, "fma_small"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

NUM_GENRES = 8
GENRE_NAMES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]

# __ Metadata __

def load_fma_metadata(meta_dir: str = META_DIR) -> pd.DataFrame:
    """Load FMA tracks.csv, filter to the *small* subset, return a tidy
    DataFrame with columns ``genre`` (str) and ``label`` (int).

    Index = track_id.
    """
    tracks = pd.read_csv(os.path.join(meta_dir, "tracks.csv"),
                         index_col=0, header=[0, 1])

    # Categorical dtype for subset column
    subsets = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            "category", categories=subsets, ordered=True)
    except (ValueError, TypeError):
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=subsets, ordered=True))

    small = tracks[tracks[("set", "subset")] == "small"]
    df = small[[("track", "genre_top")]].copy()
    df.columns = ["genre"]
    df = df.dropna()

    genre_names = sorted(df["genre"].unique())
    label2id = {g: i for i, g in enumerate(genre_names)}
    df["label"] = df["genre"].map(label2id)
    return df


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
    tid = f"{track_id:06d}"
    return os.path.join(audio_dir, tid[:3], f"{tid}.mp3")


def load_waveform(
    path: str,
    target_sr: int = 16000,
    duration: float = 10.0,
) -> np.ndarray:
    """Load an audio file and return a mono numpy array at *target_sr*.

    Uses librosa for robust MP3 decoding.  Returns shape ``(samples,)``.
    Raises ``FileNotFoundError`` / ``RuntimeError`` on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, _ = librosa.load(path, sr=target_sr, duration=duration, mono=True)
    return y


# __ Dataset __

class FMADataset(Dataset):
    """Generic FMA-Small dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``label`` column and track_id as index.
    audio_dir : str
        Root of fma_small audio tree.
    preprocess_fn : callable(waveform_np, sr) -> dict
        Model-specific preprocessing.  Must return a dict whose values are
        tensors (the dict is collated by the default DataLoader collator).
    target_sr : int
    clip_secs : float
    """

    def __init__(self, df, audio_dir, preprocess_fn, target_sr, clip_secs):
        self.track_ids = df.index.tolist()
        self.labels = df["label"].tolist()
        self.audio_dir = audio_dir
        self.preprocess_fn = preprocess_fn
        self.target_sr = target_sr
        self.clip_secs = clip_secs

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        label = self.labels[idx]
        path = get_audio_path(track_id, self.audio_dir)

        try:
            waveform = load_waveform(path, self.target_sr, self.clip_secs)
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
):
    """Build train / val / test DataLoaders from split DataFrames."""
    kw = dict(audio_dir=audio_dir, preprocess_fn=preprocess_fn,
              target_sr=target_sr, clip_secs=clip_secs)

    train_ds = FMADataset(train_df, **kw)
    val_ds = FMADataset(val_df, **kw)
    test_ds = FMADataset(test_df, **kw)

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
):
    """Write a structured JSON to ``results/runs/``."""
    now = datetime.now()
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    ts_iso = now.isoformat()

    slug = f"{model}_{variant}_{mode}_{ts_file}" if variant else f"{model}_{mode}_{ts_file}"
    runs_dir = os.path.join(RESULTS_DIR, "runs")
    _ensure_dir(runs_dir)

    record = {
        "model": model,
        "variant": variant,
        "mode": mode,
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


def checkpoint_path(model: str, variant: str = "") -> str:
    """Return a timestamped checkpoint file path and ensure the dir exists."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"{model}_{variant}_{ts}" if variant else f"{model}_{ts}"
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
