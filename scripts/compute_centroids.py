from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from interpretable_genre.midi_roll import PROGRAM_VOCAB
from interpretable_genre.torch_data import load_metadata_csv, load_tokenized_manifest
from interpretable_genre.torch_model import VAEConceptModel
from interpretable_genre.utils import load_label_map


def _aggregate_concepts(concepts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    concepts = concepts.view(mask.shape[0], mask.shape[1], -1)
    masked = concepts * mask.unsqueeze(-1)
    denom = mask.unsqueeze(-1).sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def _load_rolls(path: str, tokenized_manifest: Dict[str, str]) -> np.ndarray:
    if path not in tokenized_manifest:
        raise ValueError(f"missing tokenized entry for {path}")
    return np.load(tokenized_manifest[path])["tokens"]



def main() -> None:
    parser = argparse.ArgumentParser(description="Compute genre centroids in concept space.")
    parser.add_argument("--model", required=True, help="Path to trained checkpoint")
    parser.add_argument("--input_dir", help="Dataset directory containing metadata and tokenized manifest")
    parser.add_argument("--metadata", help="CSV with path,genre columns")
    parser.add_argument("--tokenized_manifest", help="JSON mapping MIDI path -> tokenized .npz")
    parser.add_argument("--label_map", required=True, help="Label map JSON")
    parser.add_argument("--out_root", required=True, help="Root directory to write centroids/{checkpoint_name}")
    args = parser.parse_args()

    if args.input_dir:
        if not args.metadata:
            args.metadata = os.path.join(args.input_dir, "test.csv")
        if not args.tokenized_manifest:
            args.tokenized_manifest = os.path.join(args.input_dir, "tokenized_manifest.json")
    if not args.metadata or not args.tokenized_manifest:
        raise ValueError("provide --metadata/--tokenized_manifest or --input_dir")

    ckpt = torch.load(args.model, map_location="cpu")
    config = ckpt["config"]
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
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_map = load_label_map(args.label_map)
    inv_label = {idx: label for label, idx in label_map.items()}

    tokenized_manifest = load_tokenized_manifest(args.tokenized_manifest)
    metadata = load_metadata_csv(args.metadata)

    per_genre: Dict[str, List[np.ndarray]] = {}
    checkpoint_name = os.path.splitext(os.path.basename(args.model))[0]
    out_dir = os.path.join(args.out_root, checkpoint_name)
    os.makedirs(out_dir, exist_ok=True)
    vectors_path = os.path.join(out_dir, "vectors.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    vectors_file = open(vectors_path, "w", encoding="utf-8")
    for path, genre in tqdm(metadata, desc="Centroids"):
        if not genre:
            continue
        try:
            roll_stack = _load_rolls(path, tokenized_manifest)
        except Exception:
            continue
        rolls_tensor = torch.tensor(roll_stack, dtype=torch.long).unsqueeze(0)
        mask = torch.ones((1, roll_stack.shape[0]), dtype=torch.float32)
        with torch.no_grad():
            output = model(rolls_tensor, mask)
        agg = _aggregate_concepts(output.concepts, mask).squeeze(0).cpu().numpy()
        per_genre.setdefault(genre, []).append(agg)
        vectors_file.write(
            json.dumps(
                {
                    "path": path,
                    "genre": genre,
                    "concept_vector": agg.tolist(),
                }
            )
            + "\n"
        )
    vectors_file.close()

    centroids = {}
    for genre, vectors in per_genre.items():
        if vectors:
            centroids[genre] = np.mean(vectors, axis=0).tolist()

    out_path = os.path.join(out_dir, "centroids.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "centroids": centroids,
                "concept_names": config.get("concept_names", []),
                "label_map": inv_label,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
