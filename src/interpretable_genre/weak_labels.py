from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import load_metadata
from .features import extract_measure_features
from .utils import clamp


def _minmax(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


def _dataset_stats(metadata_path: str) -> Dict[str, Tuple[float, float]]:
    df = load_metadata(metadata_path)
    return _dataset_stats_from_paths(df["path"].tolist())


def build_stats_for_paths(paths: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    density_vals: List[float] = []
    entropy_vals: List[float] = []
    for path in paths:
        try:
            measures, feature_names = extract_measure_features(path)
        except Exception:
            continue
        if not measures:
            continue
        idx_density = feature_names.index("note_density")
        idx_entropy = feature_names.index("pitch_entropy")
        for measure in measures:
            density_vals.append(float(measure.vector[idx_density]))
            entropy_vals.append(float(measure.vector[idx_entropy]))
    if not density_vals or not entropy_vals:
        return {
            "note_density": (0.0, 1.0),
            "pitch_entropy": (0.0, 1.0),
        }
    density_p10, density_p90 = np.percentile(density_vals, [10, 90])
    entropy_p10, entropy_p90 = np.percentile(entropy_vals, [10, 90])
    return {
        "note_density": (float(density_p10), float(density_p90)),
        "pitch_entropy": (float(entropy_p10), float(entropy_p90)),
    }


def _concept_scores(feature_names: List[str], vector: np.ndarray, stats: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    idx = {name: i for i, name in enumerate(feature_names)}

    triad_strength = max(vector[idx["triad_major_strength"]], vector[idx["triad_minor_strength"]])
    extended_ratio = vector[idx["extended_pitch_ratio"]]
    syncopation = vector[idx["syncopation"]]
    key_clarity = vector[idx["key_clarity"]]
    pitch_entropy = vector[idx["pitch_entropy"]]
    note_density = vector[idx["note_density"]]

    density_norm = _minmax(note_density, *stats["note_density"])
    entropy_norm = _minmax(pitch_entropy, *stats["pitch_entropy"])

    harmonic_complexity = clamp(0.6 * extended_ratio + 0.4 * (1.0 - triad_strength), 0.0, 1.0)
    tonal_stability = clamp(0.6 * key_clarity + 0.4 * (1.0 - entropy_norm), 0.0, 1.0)

    return {
        "harmonic_complexity": float(harmonic_complexity),
        "syncopation": float(syncopation),
        "tonal_stability": float(tonal_stability),
        "note_density": float(density_norm),
    }


def generate_weak_labels(metadata_path: str) -> List[Dict[str, object]]:
    df = load_metadata(metadata_path)
    stats = build_stats_for_paths(df["path"].tolist())
    outputs: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        outputs.append(generate_weak_label_for_path(row["path"], stats, row["genre"]))
    return outputs


def generate_weak_labels_from_paths(paths: List[str], genre_map: Dict[str, str] | None = None) -> List[Dict[str, object]]:
    stats = build_stats_for_paths(paths)
    outputs: List[Dict[str, object]] = []
    for path in paths:
        outputs.append(generate_weak_label_for_path(path, stats, genre_map.get(path, "") if genre_map else ""))
    return outputs


def generate_weak_label_for_path(path: str, stats: Dict[str, Tuple[float, float]], genre: str = "") -> Dict[str, object]:
    try:
        measures, feature_names = extract_measure_features(path)
    except Exception:
        measures, feature_names = [], [
            "triad_major_strength",
            "triad_minor_strength",
            "extended_pitch_ratio",
            "interval_mean",
            "interval_small",
            "interval_thirds",
            "interval_fifths",
            "interval_octaves",
            "syncopation",
            "note_density",
            "key_clarity",
            "pitch_entropy",
        ]
    measure_labels = []
    for measure in measures:
        concepts = _concept_scores(feature_names, measure.vector, stats)
        measure_labels.append(
            {
                "measure_index": measure.info.index,
                "start_tick": measure.info.start_tick,
                "end_tick": measure.info.end_tick,
                "concepts": concepts,
            }
        )
    return {
        "path": path,
        "genre": genre,
        "measures": measure_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-measure weak concept labels.")
    parser.add_argument("--metadata", required=True, help="CSV with path,genre columns")
    parser.add_argument("--out_path", required=True, help="Output JSON path")
    args = parser.parse_args()

    labels = generate_weak_labels(args.metadata)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":
    main()
