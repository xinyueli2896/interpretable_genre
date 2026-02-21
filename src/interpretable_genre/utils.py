from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class MeasureInfo:
    index: int
    start_tick: int
    end_tick: int


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def pitch_class_histogram(pitches: Iterable[int]) -> np.ndarray:
    hist = np.zeros(12, dtype=float)
    for pitch in pitches:
        hist[pitch % 12] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def entropy(prob: np.ndarray) -> float:
    prob = prob[prob > 0]
    if prob.size == 0:
        return 0.0
    return float(-np.sum(prob * np.log2(prob)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def save_label_map(path: str, label_map: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)


def top_k(items: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    return sorted(items, key=lambda x: abs(x[1]), reverse=True)[:k]


def softmax(logits: np.ndarray) -> np.ndarray:
    shift = logits - np.max(logits)
    exp = np.exp(shift)
    return exp / exp.sum()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clean_nan(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value
