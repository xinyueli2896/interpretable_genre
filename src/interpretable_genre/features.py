from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import mido

from .utils import MeasureInfo, clamp, cosine_similarity, entropy, pitch_class_histogram, safe_div


@dataclass
class MeasureFeatures:
    info: MeasureInfo
    vector: np.ndarray
    names: List[str]


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _get_time_signature(midi: mido.MidiFile) -> Tuple[int, int]:
    for track in midi.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def _collect_note_onsets(midi: mido.MidiFile) -> List[Tuple[int, int]]:
    events: List[Tuple[int, int]] = []
    abs_tick = 0
    for msg in mido.merge_tracks(midi.tracks):
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append((abs_tick, msg.note))
    return events


def _measure_boundaries(total_ticks: int, ticks_per_beat: int, numerator: int, denominator: int) -> List[MeasureInfo]:
    beat_unit = 4 / denominator
    ticks_per_measure = int(round(ticks_per_beat * numerator * beat_unit))
    if ticks_per_measure <= 0:
        ticks_per_measure = ticks_per_beat * 4
    measures = []
    start = 0
    index = 0
    while start < total_ticks:
        end = start + ticks_per_measure
        measures.append(MeasureInfo(index=index, start_tick=start, end_tick=end))
        start = end
        index += 1
    if not measures:
        measures.append(MeasureInfo(index=0, start_tick=0, end_tick=ticks_per_measure))
    return measures


def _triad_strength(hist: np.ndarray) -> Tuple[float, float, int]:
    best_major = 0.0
    best_minor = 0.0
    best_root = 0
    for root in range(12):
        major = np.zeros(12)
        minor = np.zeros(12)
        major[[root, (root + 4) % 12, (root + 7) % 12]] = 1.0
        minor[[root, (root + 3) % 12, (root + 7) % 12]] = 1.0
        major_score = cosine_similarity(hist, major)
        minor_score = cosine_similarity(hist, minor)
        if major_score > best_major:
            best_major = major_score
            best_root = root
        if minor_score > best_minor:
            best_minor = minor_score
            best_root = root
    return best_major, best_minor, best_root


def _key_clarity(hist: np.ndarray) -> Tuple[float, str]:
    best = 0.0
    label = "C:maj"
    for root in range(12):
        major_profile = np.roll(MAJOR_PROFILE, root)
        minor_profile = np.roll(MINOR_PROFILE, root)
        major_score = cosine_similarity(hist, major_profile)
        minor_score = cosine_similarity(hist, minor_profile)
        if major_score > best:
            best = major_score
            label = f"{root}:maj"
        if minor_score > best:
            best = minor_score
            label = f"{root}:min"
    return best, label


def _interval_features(onsets: List[Tuple[int, int]]) -> Dict[str, float]:
    if len(onsets) < 2:
        return {
            "interval_mean": 0.0,
            "interval_small": 0.0,
            "interval_thirds": 0.0,
            "interval_fifths": 0.0,
            "interval_octaves": 0.0,
        }
    onsets_sorted = sorted(onsets, key=lambda x: (x[0], x[1]))
    intervals = [abs(onsets_sorted[i + 1][1] - onsets_sorted[i][1]) for i in range(len(onsets_sorted) - 1)]
    if not intervals:
        return {
            "interval_mean": 0.0,
            "interval_small": 0.0,
            "interval_thirds": 0.0,
            "interval_fifths": 0.0,
            "interval_octaves": 0.0,
        }
    mean_interval = float(np.mean(intervals))
    small = sum(1 for v in intervals if v <= 2)
    thirds = sum(1 for v in intervals if v in (3, 4))
    fifths = sum(1 for v in intervals if v == 7)
    octaves = sum(1 for v in intervals if v == 12)
    total = len(intervals)
    return {
        "interval_mean": mean_interval,
        "interval_small": safe_div(small, total),
        "interval_thirds": safe_div(thirds, total),
        "interval_fifths": safe_div(fifths, total),
        "interval_octaves": safe_div(octaves, total),
    }


def _syncopation_index(onsets: List[Tuple[int, int]], ticks_per_beat: int) -> Dict[str, float]:
    if not onsets:
        return {"syncopation": 0.0, "density": 0.0}
    subdivision = max(1, ticks_per_beat // 2)
    offbeat = 0
    for tick, _ in onsets:
        if (tick // subdivision) % 2 == 1:
            offbeat += 1
    total = len(onsets)
    syncopation = safe_div(offbeat, total)
    return {"syncopation": syncopation, "density": total}


def extract_measure_features(midi_path: str) -> Tuple[List[MeasureFeatures], List[str]]:
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    numerator, denominator = _get_time_signature(midi)
    onsets = _collect_note_onsets(midi)
    total_ticks = max((tick for tick, _ in onsets), default=0)
    measures = _measure_boundaries(total_ticks + ticks_per_beat, ticks_per_beat, numerator, denominator)

    feature_names = [
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

    measure_features: List[MeasureFeatures] = []
    for measure in measures:
        measure_onsets = [(tick, pitch) for tick, pitch in onsets if measure.start_tick <= tick < measure.end_tick]
        pitches = [pitch for _, pitch in measure_onsets]
        hist = pitch_class_histogram(pitches)
        major_strength, minor_strength, _ = _triad_strength(hist)
        triad_mask = hist > 0
        extended_ratio = safe_div(float(triad_mask.sum()) - 3.0, 9.0)
        extended_ratio = clamp(extended_ratio, 0.0, 1.0)
        interval = _interval_features(measure_onsets)
        sync = _syncopation_index(measure_onsets, ticks_per_beat)
        key_clarity, _ = _key_clarity(hist)
        pitch_entropy = entropy(hist)

        vector = np.array(
            [
                major_strength,
                minor_strength,
                extended_ratio,
                interval["interval_mean"],
                interval["interval_small"],
                interval["interval_thirds"],
                interval["interval_fifths"],
                interval["interval_octaves"],
                sync["syncopation"],
                float(sync["density"]),
                key_clarity,
                pitch_entropy,
            ],
            dtype=float,
        )
        measure_features.append(MeasureFeatures(info=measure, vector=vector, names=feature_names))

    return measure_features, feature_names
