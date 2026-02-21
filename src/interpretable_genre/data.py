from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .features import MeasureFeatures, extract_measure_features


@dataclass
class SongFeatures:
    path: str
    label: str
    aggregate: np.ndarray
    measures: List[MeasureFeatures]


def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "path" not in df.columns or "genre" not in df.columns:
        raise ValueError("metadata must include 'path' and 'genre' columns")
    return df


def build_song_measure_vectors(midi_path: str) -> Tuple[np.ndarray, List[MeasureFeatures], List[str]]:
    measures, feature_names = extract_measure_features(midi_path)
    if not measures:
        return np.zeros((0, len(feature_names)), dtype=float), measures, feature_names
    return np.vstack([m.vector for m in measures]), measures, feature_names


def project_measures(measure_vectors: np.ndarray, projector) -> np.ndarray:
    if measure_vectors.size == 0:
        return np.zeros((0, projector.n_components_), dtype=float)
    return projector.transform(measure_vectors)


def aggregate_concepts(concept_vectors: np.ndarray) -> np.ndarray:
    if concept_vectors.size == 0:
        return np.zeros(concept_vectors.shape[1], dtype=float)
    return np.sum(concept_vectors, axis=0)


def dataset_from_metadata(metadata_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    df = load_metadata(metadata_path)
    labels = sorted(df["genre"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    features = []
    targets = []
    feature_names: List[str] = []
    for _, row in df.iterrows():
        measure_vectors, _, names = build_song_measure_vectors(row["path"])
        feature_names = names
        features.append(measure_vectors)
        targets.append(label_map[row["genre"]])
    return np.array(features, dtype=object), np.array(targets), label_map, feature_names
