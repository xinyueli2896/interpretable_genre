from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from interpretable_genre.midi_roll import PROGRAM_EOS, MeasureEvents, events_to_midi, midi_to_measure_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-trip MIDI -> tokens -> MIDI for sanity checks.")
    parser.add_argument("--midi_path", required=True, help="Path to MIDI file")
    parser.add_argument("--out_dir", required=True, help="Output directory for reconstructed MIDI")
    parser.add_argument("--steps_per_beat", type=int, default=4)
    parser.add_argument("--steps_per_measure", type=int, default=16)
    parser.add_argument("--max_polyphony", type=int, default=8)
    parser.add_argument("--max_tracks", type=int, default=None)
    args = parser.parse_args()

    measures, time_sig, ticks_per_beat = midi_to_measure_events(
        args.midi_path,
        steps_per_beat=args.steps_per_beat,
        max_polyphony=args.max_polyphony,
        max_tracks=args.max_tracks,
        target_steps_per_measure=args.steps_per_measure,
    )
    if not measures:
        raise ValueError("no measures found in MIDI")

    recon_midi = events_to_midi(
        [MeasureEvents(index=i, tokens=m.tokens) for i, m in enumerate(measures)],
        time_sig,
        ticks_per_beat,
        steps_per_beat=args.steps_per_beat,
    )

    tokens = np.stack([m.tokens for m in measures], axis=0)
    valid = (tokens[..., 0] < PROGRAM_EOS) & (tokens[..., 1] < 3072)
    total_notes = int(valid.sum())
    print(f"Total valid notes: {total_notes}")
    first_idx = None
    for i in range(tokens.shape[0]):
        if valid[i].any():
            first_idx = i
            break
    if first_idx is None:
        print("No valid notes found in tokens.")
    else:
        print(f"First non-empty measure index: {first_idx}")
        print("All tokens (program, pitchdur):")
        print(tokens)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.midi_path))[0]
    out_path = os.path.join(args.out_dir, f"{base}_roundtrip.mid")
    recon_midi.save(out_path)
    print(f"Saved round-trip MIDI to {out_path}")


if __name__ == "__main__":
    main()
