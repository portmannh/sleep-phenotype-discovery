"""
Convert Sleep-EDF PSG and hypnogram EDF files into FIF recordings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

EPOCH_DURATION = 30.0
data_path = "../../data"
DEFAULT_DATA_ROOTS = (
    Path(data_path),
    Path(data_path) / "sleep-edfx" / "1.0.0",
    Path("sleep-edf-database-expanded-1.0.0"),
    Path("sleep-edfx"),
)
DATASET_DIRS = {"sc": "sleep-cassette", "st": "sleep-telemetry"}

mne.set_log_level("WARNING")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Sleep-EDF PSG/Hypnogram EDF pairs to preprocessed FIF files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root folder containing sleep-cassette/ and sleep-telemetry/.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DIRS),
        default=sorted(DATASET_DIRS),
        help="Datasets to preprocess: sc, st, or both.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of recordings per dataset.",
    )
    parser.add_argument(
        "--no-trim-wake-edges",
        action="store_true",
        help="Keep wake epochs before sleep onset and after the final sleep epoch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FIF files.",
    )
    return parser.parse_args()


def raw_file_count(data_root: Path) -> int:
    return sum(
        len(list((data_root / dataset_dir).glob("*PSG.edf")))
        for dataset_dir in DATASET_DIRS.values()
    )


def resolve_data_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        if not explicit_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {explicit_root}")
        return explicit_root

    candidates = []
    for data_root in DEFAULT_DATA_ROOTS:
        if data_root.exists():
            candidates.append((raw_file_count(data_root), data_root))

    if not candidates:
        searched = ", ".join(str(path) for path in DEFAULT_DATA_ROOTS)
        raise FileNotFoundError(
            "Could not find a dataset root automatically. "
            f"Searched: {searched}. Use --data-root to point to your data."
        )

    best_count, best_root = max(candidates, key=lambda item: item[0])
    if best_count == 0:
        raise FileNotFoundError(f"No PSG EDF files found below {best_root}.")
    return best_root


def recording_identifier(path: Path) -> str:
    return path.name.split("-")[0][:-2]


def get_data_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    psg_files = sorted(dataset_dir.glob("*PSG.edf"))
    hypnogram_by_identifier = {
        recording_identifier(path): path
        for path in sorted(dataset_dir.glob("*Hypnogram.edf"))
    }

    pairs: list[tuple[Path, Path]] = []
    for psg_path in psg_files:
        hypnogram_path = hypnogram_by_identifier.get(recording_identifier(psg_path))
        if hypnogram_path is not None:
            pairs.append((psg_path, hypnogram_path))
    return pairs


def map_sleep_stage_to_label(stage: str) -> int:
    stage_mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }
    return stage_mapping.get(stage, -1)


def load_data(
    psg_path: Path,
    hypnogram_path: Path,
    epoch_duration: float = EPOCH_DURATION,
) -> tuple[np.ndarray, np.ndarray, float, list[str], list[str]]:
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    psg_data = raw.get_data()
    hypnogram = mne.read_annotations(hypnogram_path)

    sfreq = raw.info["sfreq"]
    ch_names = raw.info["ch_names"]
    ch_types = raw.get_channel_types()

    samples_per_epoch = int(round(sfreq * epoch_duration))
    total_epochs = psg_data.shape[1] // samples_per_epoch
    sleep_stage_labels = np.full(total_epochs, -1, dtype=np.int32)

    for row in hypnogram:
        stage_label = map_sleep_stage_to_label(row["description"])
        start_epoch = int(row["onset"] // epoch_duration)
        end_epoch = start_epoch + int(row["duration"] // epoch_duration)
        sleep_stage_labels[start_epoch:min(end_epoch, total_epochs)] = stage_label

    return psg_data, sleep_stage_labels, sfreq, ch_names, ch_types


def remove_wake(
    signals: np.ndarray,
    labels: np.ndarray,
    sampling_rate: float,
    epoch_duration: float = EPOCH_DURATION,
) -> tuple[np.ndarray, np.ndarray]:
    samples_per_epoch = int(round(sampling_rate * epoch_duration))
    non_wake_indices = np.where((labels != 0) & (labels != -1))[0]
    if len(non_wake_indices) == 0:
        raise ValueError("No non-wake sleep stages found in the data.")

    first_non_wake_epoch = non_wake_indices[0]
    last_non_wake_epoch = non_wake_indices[-1]

    trimmed_labels = labels[first_non_wake_epoch : last_non_wake_epoch + 1]
    start_index = first_non_wake_epoch * samples_per_epoch
    end_index = start_index + len(trimmed_labels) * samples_per_epoch
    trimmed_signals = signals[:, start_index:end_index]
    return trimmed_signals, trimmed_labels


def hypnogram_to_annotations(
    labels: np.ndarray,
    epoch_duration: float = EPOCH_DURATION,
) -> mne.Annotations:
    onsets = np.arange(len(labels)) * epoch_duration
    durations = np.full(len(labels), epoch_duration)
    descriptions = labels.astype(str)
    return mne.Annotations(onsets, durations, descriptions)


def preprocess_dataset(
    dataset_dir: Path,
    trim_wake_edges: bool,
    overwrite: bool,
    limit: int | None,
) -> tuple[int, int]:
    pairs = get_data_pairs(dataset_dir)
    if limit is not None:
        pairs = pairs[:limit]

    out_dir = dataset_dir / "preprocessed"
    out_dir.mkdir(exist_ok=True)

    written = 0
    skipped = 0

    for psg_path, hypnogram_path in tqdm(pairs, desc=f"Processing {dataset_dir.name}"):
        out_path = out_dir / psg_path.name.replace("PSG.edf", "raw.fif")
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        signal, labels, sfreq, ch_names, ch_types = load_data(psg_path, hypnogram_path)
        if trim_wake_edges:
            signal, labels = remove_wake(signal, labels, sampling_rate=sfreq)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw_new = mne.io.RawArray(signal, info)
        raw_new.set_annotations(hypnogram_to_annotations(labels))
        raw_new.save(out_path, overwrite=overwrite)
        written += 1

    return written, skipped


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    trim_wake_edges = not args.no_trim_wake_edges

    for dataset_key in args.datasets:
        dataset_dir = data_root / DATASET_DIRS[dataset_key]
        if not dataset_dir.exists():
            print(f"Skipping missing dataset directory: {dataset_dir}")
            continue

        written, skipped = preprocess_dataset(
            dataset_dir=dataset_dir,
            trim_wake_edges=trim_wake_edges,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(
            f"{dataset_dir.name}: wrote {written} FIF files"
            + (f", skipped {skipped} existing files" if skipped else "")
        )


if __name__ == "__main__":
    main()
