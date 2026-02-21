from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .utils import softmax


@dataclass
class TrainedModel:
    model: LogisticRegression
    projector: PCA
    concept_names: List[str]


def train_projector(measure_vectors: np.ndarray, concept_dim: int) -> PCA:
    projector = PCA(n_components=concept_dim, random_state=0)
    projector.fit(measure_vectors)
    return projector


def train_classifier(concept_features: np.ndarray, targets: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(concept_features, targets)
    return clf


def measure_scores(model: LogisticRegression, concept_vectors: np.ndarray) -> np.ndarray:
    weights = model.coef_
    return concept_vectors @ weights.T


def song_scores(model: LogisticRegression, concept_vectors: np.ndarray) -> np.ndarray:
    scores = measure_scores(model, concept_vectors)
    total = scores.sum(axis=0)
    total += model.intercept_
    return total


def predict_proba_from_scores(scores: np.ndarray) -> np.ndarray:
    return softmax(scores)


def feature_contributions(
    model: LogisticRegression, song_vector: np.ndarray, concept_names: List[str]
) -> List[Tuple[str, float]]:
    weights = model.coef_
    if weights.shape[0] == 1:
        contrib = song_vector * weights[0]
    else:
        predicted = int(np.argmax(weights @ song_vector + model.intercept_))
        contrib = song_vector * weights[predicted]
    return list(zip(concept_names, contrib))


def class_index_to_label(label_map: Dict[str, int]) -> Dict[int, str]:
    return {idx: label for label, idx in label_map.items()}
