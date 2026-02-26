from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .midi_roll import (
    MeasureEvents,
    PROGRAM_PAD,
    PITCHDUR_PAD,
    midi_to_measure_events,
)


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
    rolls: torch.Tensor  # (batch, max_measures, steps, max_polyphony, 2)
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
        max_measures: int | None = None,
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
        self.max_measures = max_measures

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        path, genre = self.metadata[idx]
        tokens = None
        if self.tokenized_manifest is not None and path in self.tokenized_manifest:
            token_path = self.tokenized_manifest[path]
            try:
                tokens = np.load(token_path)["tokens"]
            except Exception:
                tokens = None
        try:
            if tokens is None:
                measures, _, _ = midi_to_measure_events(
                    path,
                    steps_per_beat=self.steps_per_beat,
                    max_polyphony=self.max_polyphony,
                    max_tracks=self.max_tracks,
                    target_steps_per_measure=self.target_steps_per_measure,
                )
                if measures:
                    tokens = np.stack([m.tokens for m in measures], axis=0)
        except Exception:
            tokens = None
        if tokens is None or (hasattr(tokens, "size") and tokens.size == 0):
            return (
                np.full(
                    (0, self.target_steps_per_measure, self.max_polyphony or 1, 2),
                    fill_value=[PROGRAM_PAD, PITCHDUR_PAD],
                    dtype=np.int32,
                ),
                np.zeros((0, len(self.concept_names)), dtype=np.float32),
                self.label_map.get(genre, -1),
            )

        roll_stack = tokens
        if self.max_measures is not None and roll_stack.shape[0] > self.max_measures:
            roll_stack = roll_stack[: self.max_measures]
        concepts = np.zeros((roll_stack.shape[0], len(self.concept_names)), dtype=np.float32)
        concept_map = self.weak_labels.get(path, {})
        for i in range(roll_stack.shape[0]):
            label = concept_map.get(i, {})
            for j, name in enumerate(self.concept_names):
                concepts[i, j] = float(label.get(name, 0.0))

        return roll_stack, concepts, self.label_map.get(genre, -1)


def collate_song_batch(batch: List[Tuple[np.ndarray, np.ndarray, int]]) -> SongBatch:
    max_measures = max(item[0].shape[0] for item in batch)
    steps = max((item[0].shape[1] for item in batch), default=1)
    max_polyphony = max((item[0].shape[2] for item in batch), default=1)
    num_concepts = max((item[1].shape[1] for item in batch), default=1)

    rolls = torch.full(
        (len(batch), max_measures, steps, max_polyphony, 2),
        fill_value=0,
        dtype=torch.long,
    )
    rolls[..., 0] = PROGRAM_PAD
    rolls[..., 1] = PITCHDUR_PAD
    concepts = torch.zeros((len(batch), max_measures, num_concepts), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_measures), dtype=torch.float32)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    for i, (roll, concept, _) in enumerate(batch):
        count = roll.shape[0]
        if count == 0:
            continue
        if roll.shape[2] != max_polyphony:
            pad_poly = max_polyphony - roll.shape[2]
            if pad_poly > 0:
                pad_block = np.full((roll.shape[0], roll.shape[1], pad_poly, 2), fill_value=[PROGRAM_PAD, PITCHDUR_PAD], dtype=roll.dtype)
                roll = np.concatenate([roll, pad_block], axis=2)
            else:
                roll = roll[:, :, :max_polyphony]
        if roll.shape[1] != steps:
            pad_steps = steps - roll.shape[1]
            if pad_steps > 0:
                pad_block = np.full((roll.shape[0], pad_steps, max_polyphony, 2), fill_value=[PROGRAM_PAD, PITCHDUR_PAD], dtype=roll.dtype)
                roll = np.concatenate([roll, pad_block], axis=1)
            else:
                roll = roll[:, :steps]
        rolls[i, :count] = torch.tensor(roll, dtype=torch.long)
        concepts[i, :count] = torch.tensor(concept, dtype=torch.float32)
        mask[i, :count] = 1.0

    return SongBatch(rolls=rolls, concepts=concepts, mask=mask, labels=labels)
