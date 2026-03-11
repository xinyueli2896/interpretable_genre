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

from interpretable_genre.midi_roll import PROGRAM_VOCAB, midi_to_measure_events
from interpretable_genre.torch_data import load_metadata_csv, load_tokenized_manifest
from interpretable_genre.torch_model import VAEConceptModel
from interpretable_genre.utils import load_label_map, softmax


def _normalize_rolls(rolls: np.ndarray, config: Dict[str, int]) -> np.ndarray:
    return rolls


def _load_roll_stack(
    path: str,
    config: Dict[str, int],
    max_tracks: int | None,
    max_polyphony: int | None,
    tokenized_manifest: Dict[str, str] | None,
) -> np.ndarray | None:
    try:
        if tokenized_manifest and path in tokenized_manifest:
            data = np.load(tokenized_manifest[path])
            rolls = data["tokens"]
            return _normalize_rolls(rolls, config)

        measures, _, _ = midi_to_measure_events(
            path,
            steps_per_beat=config["steps_per_beat"],
            max_polyphony=max_polyphony,
            max_tracks=max_tracks,
            target_steps_per_measure=config["steps_per_measure"],
        )
        if not measures:
            return None
        return np.stack([m.tokens for m in measures], axis=0)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classification accuracy on a test CSV.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--label_map", required=True, help="Path to label map json")
    parser.add_argument("--input_dir", help="Dataset directory containing test CSV and tokenized manifest")
    parser.add_argument("--test_csv", help="CSV with path,genre columns")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k accuracy to report")
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
    use_concept_bottleneck = config.get("use_concept_bottleneck", True)
    enable_decoder = config.get("enable_decoder", True)
    model = VAEConceptModel(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        num_concepts=config["num_concepts"],
        num_genres=config["num_genres"],
        hidden_dim=config["hidden_dim"],
        steps_per_measure=config["steps_per_measure"],
        max_polyphony=config.get("max_polyphony"),
        program_vocab=config.get("program_vocab", PROGRAM_VOCAB),
        pitchdur_vocab=config.get("pitchdur_vocab", 3074),
        token_embed_dim=config.get("token_embed_dim", 32),
        use_concept_bottleneck=use_concept_bottleneck,
        enable_decoder=enable_decoder,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    torch.set_grad_enabled(False)

    label_map = load_label_map(args.label_map)
    metadata = load_metadata_csv(args.test_csv)
    tokenized_manifest = load_tokenized_manifest(args.tokenized_manifest) if args.tokenized_manifest else None

    max_tracks = args.max_tracks or config.get("max_tracks")

    total = 0
    correct_top1 = 0
    correct_topk = 0
    skipped = 0
    count = 0
    for path, genre in tqdm(metadata, desc="Evaluating"):
        if genre not in label_map:
            skipped += 1
            continue
        roll_stack = _load_roll_stack(
            path,
            config,
            max_tracks,
            args.max_polyphony,
            tokenized_manifest,
        )
        if roll_stack is None:
            skipped += 1
            continue
        rolls_tensor = torch.tensor(roll_stack, dtype=torch.long).unsqueeze(0)
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
        count += 1
        if count >= 2000: break

    top1 = (correct_top1 / total) if total else 0.0
    topk = (correct_topk / total) if total else 0.0
    print(f"Total evaluated: {total}")
    print(f"Skipped: {skipped}")i
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-{args.top_k} accuracy: {topk:.4f}")


if __name__ == "__main__":
    main()
