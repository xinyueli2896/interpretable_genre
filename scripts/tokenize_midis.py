from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from interpretable_genre.midi_roll import midi_to_measure_rolls, midi_to_measure_rolls_by_track


def _find_midis(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".mid") or lower.endswith(".midi"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)



def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize MIDI files into numpy tensors.")
    parser.add_argument("--input_dir", help="Dataset directory containing MIDI files")
    parser.add_argument("--root", help="Root directory containing MIDI files")
    parser.add_argument("--out_dir", help="Directory to write tokenized tensors")
    parser.add_argument("--manifest_out", help="Output JSON manifest path")
    parser.add_argument("--steps_per_beat", type=int, default=16)
    parser.add_argument("--steps_per_measure", type=int, default=64)
    parser.add_argument("--max_tracks", type=int, required=True)
    parser.add_argument("--max_polyphony", type=int, default=None)
    args = parser.parse_args()

    if args.input_dir:
        if not args.root:
            args.root = args.input_dir
        if not args.out_dir:
            args.out_dir = os.path.join(args.input_dir, "tokenized_npz")
        if not args.manifest_out:
            args.manifest_out = os.path.join(args.input_dir, "tokenized_manifest.json")
    if not args.root or not args.out_dir or not args.manifest_out:
        raise ValueError("provide --input_dir or --root/--out_dir/--manifest_out")
    if args.max_tracks is None:
        raise ValueError("--max_tracks is required")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest: Dict[str, str] = {}

    midi_paths = _find_midis(args.root)
    for path in tqdm(midi_paths, desc="Tokenizing"):
        try:
            tracks, _, _, _, _ = midi_to_measure_rolls_by_track(
                path,
                steps_per_beat=args.steps_per_beat,
                target_steps_per_measure=args.steps_per_measure,
                max_tracks=args.max_tracks,
                max_polyphony=args.max_polyphony,
            )
            if not tracks:
                continue
            per_track = []
            for track in tracks:
                per_track.append(np.stack([m.roll for m in track], axis=0))
            rolls = np.stack(per_track, axis=1)  # (measures, tracks, steps, 128)
        except Exception:
            continue

        token_name = os.path.basename(path) + ".npz"
        token_path = os.path.join(args.out_dir, token_name)
        np.savez_compressed(token_path, rolls=rolls.astype(np.float32))
        manifest[path] = token_path

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
