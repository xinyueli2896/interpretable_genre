from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .data import aggregate_concepts, build_song_measure_vectors, project_measures
from .model import class_index_to_label, measure_scores, predict_proba_from_scores, song_scores
from .utils import MeasureInfo, clean_nan, top_k


@dataclass
class MeasureEvidence:
    measure: MeasureInfo
    score: float


def build_explanation(model, projector, concept_names: List[str], label_map: Dict[str, int], midi_path: str) -> Dict[str, object]:
    measure_vectors, measures, _ = build_song_measure_vectors(midi_path)
    if not measures:
        raise ValueError("no measures found in MIDI")

    concept_vectors = project_measures(measure_vectors, projector)
    aggregate = aggregate_concepts(concept_vectors)
    scores = song_scores(model, concept_vectors)
    probs = predict_proba_from_scores(scores)

    label_lookup = class_index_to_label(label_map)
    predicted_idx = int(np.argmax(probs))
    predicted_label = label_lookup[predicted_idx]
    confidence = float(probs[predicted_idx])

    per_measure_scores = measure_scores(model, concept_vectors)
    measure_evidence = [
        MeasureEvidence(measure=m.info, score=float(per_measure_scores[idx, predicted_idx]))
        for idx, m in enumerate(measures)
    ]
    measure_evidence_sorted = sorted(measure_evidence, key=lambda x: x.score, reverse=True)
    top_measures = [
        {
            "measure_index": ev.measure.index,
            "start_tick": ev.measure.start_tick,
            "end_tick": ev.measure.end_tick,
            "score": clean_nan(ev.score),
        }
        for ev in measure_evidence_sorted[:5]
    ]

    weights = model.coef_[predicted_idx]
    contributions = list(zip(concept_names, (aggregate * weights).tolist()))
    top_features = [
        {"feature": name, "contribution": clean_nan(value)}
        for name, value in top_k(contributions, k=6)
    ]

    occlusion_deltas = []
    total_score = scores[predicted_idx]
    for idx, m in enumerate(measures):
        occluded_score = total_score - per_measure_scores[idx, predicted_idx]
        delta = total_score - occluded_score
        occlusion_deltas.append(delta)

    occlusion_measures = [
        {
            "measure_index": measures[i].info.index,
            "delta": clean_nan(float(occlusion_deltas[i])),
        }
        for i in np.argsort(occlusion_deltas)[::-1][:5]
    ]

    if top_measures:
        measure_indices = sorted(item["measure_index"] for item in top_measures[:2])
        if len(measure_indices) == 1:
            measure_span = f"measure {measure_indices[0]}"
        else:
            measure_span = f"measures {measure_indices[0]}-{measure_indices[-1]}"
    else:
        measure_span = "key measures"

    top_concepts = [item["feature"] for item in top_features[:2]]
    concept_phrase = ", ".join(top_concepts) if top_concepts else "key concepts"
    summary = f"The genre is {predicted_label} because {measure_span} show strong {concept_phrase}."

    return {
        "midi_path": midi_path,
        "predicted_genre": predicted_label,
        "confidence": clean_nan(confidence),
        "top_features": top_features,
        "top_measures": top_measures,
        "occlusion": occlusion_measures,
        "summary": summary,
    }
