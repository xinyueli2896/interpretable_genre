from __future__ import annotations

import argparse
import json
from typing import Dict, List

import joblib
import numpy as np

from .data import aggregate_concepts, dataset_from_metadata, project_measures
from .model import class_index_to_label, predict_proba_from_scores
from .utils import load_label_map


def _genre_profiles(
    per_song_measures: np.ndarray, targets: np.ndarray, label_map: Dict[str, int], projector
) -> Dict[str, np.ndarray]:
    profiles: Dict[str, List[np.ndarray]] = {label: [] for label in label_map}
    inv = class_index_to_label(label_map)
    for idx, measure_vectors in enumerate(per_song_measures):
        label = inv[int(targets[idx])]
        concepts = project_measures(measure_vectors, projector)
        aggregate = aggregate_concepts(concepts)
        profiles[label].append(aggregate)
    return {label: np.mean(vectors, axis=0) for label, vectors in profiles.items() if vectors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate between two genre profiles.")
    parser.add_argument("--metadata", required=True, help="CSV with path,genre columns")
    parser.add_argument("--model", required=True, help="Path to joblib model")
    parser.add_argument("--label_map", required=True, help="Path to label map json")
    parser.add_argument("--genre_a", required=True, help="Start genre")
    parser.add_argument("--genre_b", required=True, help="End genre")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--out_path", help="Optional output JSON path")
    args = parser.parse_args()

    payload = joblib.load(args.model)
    model = payload["model"]
    projector = payload["projector"]

    per_song_measures, targets, label_map, _ = dataset_from_metadata(args.metadata)
    profiles = _genre_profiles(per_song_measures, targets, label_map, projector)

    if args.genre_a not in profiles or args.genre_b not in profiles:
        raise ValueError("requested genres not found in metadata")

    vec_a = profiles[args.genre_a]
    vec_b = profiles[args.genre_b]

    results = []
    for step in range(args.steps):
        alpha = step / max(1, args.steps - 1)
        blend = (1 - alpha) * vec_a + alpha * vec_b
        scores = model.coef_ @ blend + model.intercept_
        probs = predict_proba_from_scores(scores)
        predicted = int(np.argmax(probs))
        label_lookup = class_index_to_label(label_map)
        results.append(
            {
                "step": step,
                "alpha": alpha,
                "predicted_genre": label_lookup[predicted],
                "confidence": float(probs[predicted]),
            }
        )

    for item in results:
        print(f"step {item['step']}: alpha={item['alpha']:.2f} -> {item['predicted_genre']} ({item['confidence']:.3f})")

    if args.out_path:
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
