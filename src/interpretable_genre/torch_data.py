from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .midi_roll import MeasureRoll, midi_to_measure_rolls, midi_to_measure_rolls_by_track


def load_metadata_csv(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "genre" not in reader.fieldnames:
            raise ValueError("metadata must include path and genre columns")
        return [(row["path"], row["genre"]) for row in reader]


def build_label_map(metadata: List[Tuple[str, str]]) -> Dict[str, int]:
    labels = sorted({genre for _, genre in metadata if genre})
    return {label: idx for idx, label in enumerate(labels)}


def load_weak_labels_jsonl(path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    data: Dict[str, Dict[int, Dict[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            measures = item.get("measures", [])
            measure_map: Dict[int, Dict[str, float]] = {}
            for measure in measures:
                measure_map[int(measure["measure_index"])] = {
                    key: float(val) for key, val in measure.get("concepts", {}).items()
                }
            data[item["path"]] = measure_map
    return data


def load_tokenized_manifest(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


@dataclass
class SongBatch:
    rolls: torch.Tensor  # (batch, max_measures, steps, 128)
    concepts: torch.Tensor  # (batch, max_measures, num_concepts)
    mask: torch.Tensor  # (batch, max_measures)
    labels: torch.Tensor  # (batch,)


class MidiConceptDataset(Dataset):
    def __init__(
        self,
        metadata: List[Tuple[str, str]],
        weak_labels: Dict[str, Dict[int, Dict[str, float]]],
        label_map: Dict[str, int],
        concept_names: List[str],
        steps_per_beat: int = 4,
        target_steps_per_measure: int = 16,
        max_polyphony: int | None = None,
        track_aware: bool = False,
        max_tracks: int | None = None,
        tokenized_manifest: Dict[str, str] | None = None,
    ) -> None:
        self.metadata = metadata
        self.weak_labels = weak_labels
        self.label_map = label_map
        self.concept_names = concept_names
        self.steps_per_beat = steps_per_beat
        self.target_steps_per_measure = target_steps_per_measure
        self.max_polyphony = max_polyphony
        self.track_aware = track_aware
        self.max_tracks = max_tracks
        self.tokenized_manifest = tokenized_manifest

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        path, genre = self.metadata[idx]
        rolls = None
        if self.tokenized_manifest is not None and path in self.tokenized_manifest:
            token_path = self.tokenized_manifest[path]
            try:
                rolls = np.load(token_path)["rolls"]
            except Exception:
                rolls = None
        try:
            if rolls is None and self.track_aware:
                if self.max_tracks is None:
                    raise ValueError("max_tracks must be set when track_aware=True")
                tracks, _, _, _, _ = midi_to_measure_rolls_by_track(
                    path,
                    steps_per_beat=self.steps_per_beat,
                    target_steps_per_measure=self.target_steps_per_measure,
                    max_tracks=self.max_tracks,
                    max_polyphony=self.max_polyphony,
                )
                if not tracks:
                    rolls = None
                else:
                    if len(tracks[0]) == 0:
                        rolls = None
                    else:
                        while len(tracks) < self.max_tracks:
                            tracks.append(
                                [MeasureRoll(index=i, roll=np.zeros_like(tracks[0][i].roll)) for i in range(len(tracks[0]))]
                            )
                        per_track = []
                        for track in tracks[: self.max_tracks]:
                            per_track.append(np.stack([m.roll for m in track], axis=0))
                        roll_stack = np.stack(per_track, axis=1)  # (measures, tracks, steps, 128)
                        rolls = roll_stack
            elif rolls is None:
                rolls, _, _, _ = midi_to_measure_rolls(
                    path,
                    steps_per_beat=self.steps_per_beat,
                    target_steps_per_measure=self.target_steps_per_measure,
                    max_polyphony=self.max_polyphony,
                )
        except Exception:
            rolls = None
        if rolls is None or (hasattr(rolls, "size") and rolls.size == 0):
            return (
                np.zeros(
                    (0, self.max_tracks or 1, self.target_steps_per_measure, 128) if self.track_aware else (0, self.target_steps_per_measure, 128),
                    dtype=np.float32,
                ),
                np.zeros((0, len(self.concept_names)), dtype=np.float32),
                self.label_map.get(genre, -1),
            )

        if self.track_aware:
            roll_stack = rolls
        else:
            roll_stack = np.stack([m.roll for m in rolls], axis=0)
        concepts = np.zeros((roll_stack.shape[0], len(self.concept_names)), dtype=np.float32)
        concept_map = self.weak_labels.get(path, {})
        if self.track_aware:
            for i in range(roll_stack.shape[0]):
                label = concept_map.get(i, {})
                for j, name in enumerate(self.concept_names):
                    concepts[i, j] = float(label.get(name, 0.0))
        else:
            for i, measure in enumerate(rolls):
                label = concept_map.get(measure.index, {})
                for j, name in enumerate(self.concept_names):
                    concepts[i, j] = float(label.get(name, 0.0))

        return roll_stack, concepts, self.label_map.get(genre, -1)


def collate_song_batch(batch: List[Tuple[np.ndarray, np.ndarray, int]]) -> SongBatch:
    max_measures = max(item[0].shape[0] for item in batch)
    if max_measures == 0:
        steps = 1
        tracks = 1
    else:
        if batch[0][0].ndim == 4:
            tracks = max((item[0].shape[1] for item in batch), default=1)
            steps = max((item[0].shape[2] for item in batch), default=1)
        else:
            tracks = 1
            steps = max((item[0].shape[1] for item in batch), default=1)
    num_concepts = max((item[1].shape[1] for item in batch), default=1)

    if tracks == 1 and (batch[0][0].ndim == 3 or max_measures == 0):
        rolls = torch.zeros((len(batch), max_measures, steps, 128), dtype=torch.float32)
    else:
        rolls = torch.zeros((len(batch), max_measures, tracks, steps, 128), dtype=torch.float32)
    concepts = torch.zeros((len(batch), max_measures, num_concepts), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_measures), dtype=torch.float32)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    for i, (roll, concept, _) in enumerate(batch):
        count = roll.shape[0]
        if count == 0:
            continue
        if roll.ndim == 4 and roll.shape[1] != tracks:
            pad_tracks = tracks - roll.shape[1]
            if pad_tracks > 0:
                roll = np.pad(roll, ((0, 0), (0, pad_tracks), (0, 0), (0, 0)), mode="constant")
            else:
                roll = roll[:, :tracks]
        rolls[i, :count] = torch.tensor(roll, dtype=torch.float32)
        concepts[i, :count] = torch.tensor(concept, dtype=torch.float32)
        mask[i, :count] = 1.0

    return SongBatch(rolls=rolls, concepts=concepts, mask=mask, labels=labels)
