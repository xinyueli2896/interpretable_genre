from __future__ import annotations

import argparse
import joblib
import numpy as np

from .data import aggregate_concepts, dataset_from_metadata, project_measures
from .model import train_classifier, train_projector
from .utils import save_label_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Train interpretable genre classifier.")
    parser.add_argument("--metadata", required=True, help="CSV with path,genre columns")
    parser.add_argument("--model_out", required=True, help="Output path for joblib model")
    parser.add_argument("--label_map_out", required=True, help="Output path for label map json")
    parser.add_argument("--concept_dim", type=int, default=8, help="Concept bottleneck dimension")
    args = parser.parse_args()

    per_song_measures, targets, label_map, _ = dataset_from_metadata(args.metadata)

    all_measures = []
    for song_measures in per_song_measures:
        if song_measures.size > 0:
            all_measures.append(song_measures)
    if not all_measures:
        raise ValueError("no measure vectors found in metadata")
    stacked_measures = np.vstack(all_measures)

    projector = train_projector(stacked_measures, args.concept_dim)

    concept_features = []
    for song_measures in per_song_measures:
        concepts = project_measures(song_measures, projector)
        aggregate = aggregate_concepts(concepts)
        concept_features.append(aggregate)
    concept_features = np.vstack(concept_features)

    model = train_classifier(concept_features, targets)

    payload = {
        "model": model,
        "projector": projector,
        "concept_names": [f"concept_{i}" for i in range(args.concept_dim)],
    }
    joblib.dump(payload, args.model_out)
    save_label_map(args.label_map_out, label_map)


if __name__ == "__main__":
    main()
