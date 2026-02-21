from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from interpretable_genre.midi_roll import midi_to_measure_rolls, midi_to_measure_rolls_by_track
from interpretable_genre.torch_data import load_metadata_csv, load_tokenized_manifest
from interpretable_genre.torch_model import VAEConceptModel
from interpretable_genre.utils import load_label_map, softmax


def _normalize_rolls(rolls: np.ndarray, config: Dict[str, int]) -> np.ndarray:
    max_tracks = config.get("max_tracks")
    steps_per_measure = config["steps_per_measure"]
    measures, tracks, steps, pitches = rolls.shape
    if max_tracks is not None:
        if tracks > max_tracks:
            rolls = rolls[:, :max_tracks]
        elif tracks < max_tracks:
            pad = np.zeros((measures, max_tracks - tracks, steps, pitches), dtype=rolls.dtype)
            rolls = np.concatenate([rolls, pad], axis=1)
    if steps < steps_per_measure:
        pad = np.zeros((measures, rolls.shape[1], steps_per_measure - steps, pitches), dtype=rolls.dtype)
        rolls = np.concatenate([rolls, pad], axis=2)
    elif steps > steps_per_measure:
        rolls = rolls[:, :, :steps_per_measure, :]
    return rolls


def _load_roll_stack(
    path: str,
    config: Dict[str, int],
    track_aware: bool,
    max_tracks: int | None,
    max_polyphony: int | None,
    tokenized_manifest: Dict[str, str] | None,
) -> np.ndarray | None:
    try:
        if tokenized_manifest and path in tokenized_manifest:
            data = np.load(tokenized_manifest[path])
            rolls = data["rolls"]
            if not track_aware:
                raise ValueError("tokenized_manifest is track-aware; pass --track_aware")
            return _normalize_rolls(rolls, config)

        if track_aware:
            tracks, _, _, _, _ = midi_to_measure_rolls_by_track(
                path,
                steps_per_beat=config["steps_per_beat"],
                target_steps_per_measure=config["steps_per_measure"],
                max_tracks=max_tracks,
                max_polyphony=max_polyphony,
            )
            if not tracks:
                return None
            per_track = [np.stack([m.roll for m in track], axis=0) for track in tracks]
            rolls = np.stack(per_track, axis=1)
            return _normalize_rolls(rolls, config)

        rolls, _, _, _ = midi_to_measure_rolls(
            path,
            steps_per_beat=config["steps_per_beat"],
            target_steps_per_measure=config["steps_per_measure"],
            max_polyphony=max_polyphony,
        )
        if not rolls:
            return None
        return np.stack([m.roll for m in rolls], axis=0)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classification accuracy on a test CSV.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--label_map", required=True, help="Path to label map json")
    parser.add_argument("--input_dir", help="Dataset directory containing test CSV and tokenized manifest")
    parser.add_argument("--test_csv", help="CSV with path,genre columns")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k accuracy to report")
    parser.add_argument("--track_aware", action="store_true", help="Use track-aware rolls")
    parser.add_argument("--max_tracks", type=int, default=None, help="Max tracks for track-aware")
    parser.add_argument("--max_polyphony", type=int, default=8, help="Max polyphony per step")
    parser.add_argument("--tokenized_manifest", help="Optional tokenized manifest JSON")
    args = parser.parse_args()

    if args.input_dir:
        if not args.test_csv:
            args.test_csv = os.path.join(args.input_dir, "test.csv")
        if not args.tokenized_manifest:
            manifest_path = os.path.join(args.input_dir, "tokenized_manifest.json")
            if os.path.exists(manifest_path):
                args.tokenized_manifest = manifest_path
    if not args.test_csv:
        raise ValueError("provide --test_csv or --input_dir")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    model = VAEConceptModel(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        num_concepts=config["num_concepts"],
        num_genres=config["num_genres"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    torch.set_grad_enabled(False)

    label_map = load_label_map(args.label_map)
    metadata = load_metadata_csv(args.test_csv)
    tokenized_manifest = load_tokenized_manifest(args.tokenized_manifest) if args.tokenized_manifest else None

    track_aware = args.track_aware or bool(config.get("track_aware", False))
    max_tracks = args.max_tracks or config.get("max_tracks")
    if track_aware and max_tracks is None:
        raise ValueError("--max_tracks is required for track-aware evaluation")

    total = 0
    correct_top1 = 0
    correct_topk = 0
    skipped = 0

    for path, genre in tqdm(metadata, desc="Evaluating"):
        if genre not in label_map:
            skipped += 1
            continue
        roll_stack = _load_roll_stack(
            path,
            config,
            track_aware,
            max_tracks,
            args.max_polyphony,
            tokenized_manifest,
        )
        if roll_stack is None:
            skipped += 1
            continue
        if track_aware:
            rolls_tensor = torch.tensor(roll_stack, dtype=torch.float32).unsqueeze(0)
        else:
            rolls_tensor = torch.tensor(roll_stack, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones((1, roll_stack.shape[0]), dtype=torch.float32)
        with torch.no_grad():
            output = model(rolls_tensor, mask)
        logits = output.class_logits.squeeze(0).cpu().numpy()
        probs = softmax(logits)
        top_idx = np.argsort(probs)[::-1]
        gold = label_map[genre]
        total += 1
        if top_idx[0] == gold:
            correct_top1 += 1
        if gold in top_idx[: args.top_k]:
            correct_topk += 1

    top1 = (correct_top1 / total) if total else 0.0
    topk = (correct_topk / total) if total else 0.0
    print(f"Total evaluated: {total}")
    print(f"Skipped: {skipped}")
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-{args.top_k} accuracy: {topk:.4f}")


if __name__ == "__main__":
    main()
