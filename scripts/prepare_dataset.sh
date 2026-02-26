#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <midi_root_dir> [split] [num_workers]"
  echo "Example: $0 /path/to/midis 0.2 8"
  exit 1
fi

MIDI_ROOT="$1"
SPLIT="${2:-0.2}"
if [[ $# -ge 3 ]]; then
  NUM_WORKERS="$3"
elif [[ -n "${NUM_WORKERS:-}" ]]; then
  NUM_WORKERS="$NUM_WORKERS"
else
  NUM_WORKERS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)"
fi
MAX_TRACKS="${MAX_TRACKS:-}"
MIN_ALIGNED_RATIO="${MIN_ALIGNED_RATIO:-0.7}"
ALIGN_TOLERANCE_STEPS="${ALIGN_TOLERANCE_STEPS:-0.25}"

TOKENIZE_ONLY="${TOKENIZE_ONLY:-0}"

if [[ ! -d "$MIDI_ROOT" ]]; then
  echo "MIDI root directory not found: $MIDI_ROOT"
  exit 1
fi

DATASET_NAME="$(basename "$MIDI_ROOT")"
OUTPUT_DIR="input/$DATASET_NAME"
METADATA_ARG=()
if [[ -n "${METADATA_PATH:-}" ]]; then
  METADATA_ARG=(--metadata "$METADATA_PATH")
elif [[ -f "$MIDI_ROOT/metadata_genre.csv" ]]; then
  METADATA_ARG=(--metadata "$MIDI_ROOT/metadata_genre.csv")
fi

if [[ "$TOKENIZE_ONLY" == "1" ]]; then
  echo "Skipping weak labels (TOKENIZE_ONLY=1)"
else
  echo "Step 1/2: build weak labels and train/test split"
  python scripts/build_lmd_weak_labels.py \
    --root "$MIDI_ROOT" \
    --out_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --num_workers "$NUM_WORKERS" \
    "${METADATA_ARG[@]}"
fi

echo "Step 2/2: tokenize MIDI files (track-aware)"
python scripts/tokenize_midis.py \
  --root "$MIDI_ROOT" \
  --out_dir "$OUTPUT_DIR/tokenized_npz" \
  --manifest_out "$OUTPUT_DIR/tokenized_manifest.json" \
  --steps_per_beat 4 \
  --steps_per_measure 16 \
  --min_aligned_ratio "$MIN_ALIGNED_RATIO" \
  --align_tolerance_steps "$ALIGN_TOLERANCE_STEPS" \
  ${MAX_TRACKS:+--max_tracks "$MAX_TRACKS"} \
  --max_polyphony 8 \
  --num_workers "$NUM_WORKERS"

echo "Done. Outputs in $OUTPUT_DIR:"
echo "  - train.csv"
echo "  - test.csv"
echo "  - weak_labels.jsonl"
echo "  - tokenized_manifest.json"
echo "  - tokenized_npz/"
