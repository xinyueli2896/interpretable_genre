from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from interpretable_genre.midi_roll import midi_to_measure_events


def _find_midis(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".mid") or lower.endswith(".midi"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)

_CONFIG = {}


def _init_worker(config: Dict[str, object]) -> None:
    global _CONFIG
    _CONFIG = config


def _tokenize_one(path: str) -> Optional[Tuple[str, str]]:
    try:
        measures, _, _ = midi_to_measure_events(
            path,
            steps_per_beat=_CONFIG["steps_per_beat"],
            max_polyphony=_CONFIG["max_polyphony"],
            max_tracks=_CONFIG["max_tracks"],
            target_steps_per_measure=_CONFIG["steps_per_measure"],
            min_aligned_ratio=_CONFIG["min_aligned_ratio"],
            align_tolerance_steps=_CONFIG["align_tolerance_steps"],
        )
        if not measures:
            return None
        tokens = np.stack([m.tokens for m in measures], axis=0)  # (measures, steps, poly, 2)
    except Exception:
        return None

    token_name = os.path.basename(path) + ".npz"
    token_path = os.path.join(_CONFIG["out_dir"], token_name)
    np.savez_compressed(token_path, tokens=tokens.astype(np.int32))
    return path, token_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize MIDI files into numpy tensors.")
    parser.add_argument("--input_dir", help="Dataset directory containing MIDI files")
    parser.add_argument("--root", help="Root directory containing MIDI files")
    parser.add_argument("--out_dir", help="Directory to write tokenized tensors")
    parser.add_argument("--manifest_out", help="Output JSON manifest path")
    parser.add_argument("--steps_per_beat", type=int, default=4)
    parser.add_argument("--steps_per_measure", type=int, default=16)
    parser.add_argument("--max_tracks", type=int, default=None)
    parser.add_argument("--max_polyphony", type=int, default=None)
    parser.add_argument(
        "--min_aligned_ratio",
        type=float,
        default=0.7,
        help="Skip MIDIs with fewer aligned onsets than this ratio",
    )
    parser.add_argument(
        "--align_tolerance_steps",
        type=float,
        default=0.25,
        help="Alignment tolerance in steps when filtering off-grid MIDIs",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel workers for tokenization (0=cpu count)")
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

    os.makedirs(args.out_dir, exist_ok=True)
    manifest: Dict[str, str] = {}

    midi_paths = _find_midis(args.root)
    config = {
        "steps_per_beat": args.steps_per_beat,
        "steps_per_measure": args.steps_per_measure,
        "max_tracks": args.max_tracks,
        "max_polyphony": args.max_polyphony,
        "out_dir": args.out_dir,
        "min_aligned_ratio": args.min_aligned_ratio,
        "align_tolerance_steps": args.align_tolerance_steps,
    }
    num_workers = args.num_workers if args.num_workers and args.num_workers > 0 else (os.cpu_count() or 1)
    if num_workers > 1:
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(config,)) as pool:
            for result in tqdm(pool.imap_unordered(_tokenize_one, midi_paths), total=len(midi_paths), desc="Tokenizing"):
                if not result:
                    continue
                path, token_path = result
                manifest[path] = token_path
    else:
        _init_worker(config)
        for path in tqdm(midi_paths, desc="Tokenizing"):
            result = _tokenize_one(path)
            if not result:
                continue
            src_path, token_path = result
            manifest[src_path] = token_path

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
