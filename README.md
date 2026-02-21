# Interpretable Genre Classification

Theory-constrained, measure-localized genre classification from MIDI using music-theoretic features, a concept bottleneck, and additive measure scoring. Includes measure-level evidence localization and automatic Genre Explanation Cards.

## Project layout

- `src/interpretable_genre/` core library
- `scripts/` helper scripts
- `configs/` example configs

## Install

```bash
conda create -n interpretable_genre python=3.9 -y
conda activate interpretable_genre
pip install -r requirements.txt
```

## Dataset format

Create a CSV with two columns:

- `path`: absolute or workspace-relative path to a MIDI file
- `genre`: string label

Example `input/dataset_name/metadata_genre.csv`:

```csv
path,genre
midi/jazz_001.mid,jazz
midi/rock_014.mid,rock
```

## End-to-end workflow (MIDI folders → weak labels → tokenization → train → infer)

### 1) Build weak labels from MIDI folders

```bash
python scripts/build_lmd_weak_labels.py \
  --input_dir input/dataset_name \
  --split 0.2
```

Outputs:
- `train.csv`, `test.csv` under the `--out_dir`
- `weak_labels.jsonl` under the `--out_dir`

### 2) Tokenize MIDI files (track-aware)
To speed up the training, we precompute MIDI into tokens. 

```bash
python scripts/tokenize_midis.py \
  --input_dir input/dataset_name \
  --steps_per_beat 16 \
  --steps_per_measure 64 \
  --max_tracks 8 \
  --max_polyphony 8
```

### 3) Train (VAE + soft concept bottleneck, track-aware)
The model itself. 

```bash
python -m interpretable_genre.train_torch_cbm \
  --input_dir input/dataset_name \
  --model_out artifacts/vae_cbm.pt \
  --label_map_out artifacts/labels.json \
  --track_aware \
  --max_tracks 8 \
  --max_polyphony 8 \
  --steps_per_beat 16 \
  --steps_per_measure 64
```



## Precompute centroids for interpolation

```bash
python scripts/compute_centroids.py \
  --checkpoint /path/to/checkpoint \
  --label_map artifacts/labels.json \
  --input_dir input/dataset_name \
  --out_root artifacts/centroids 
```

## Evaluate classification accuracy

```bash
python scripts/eval_accuracy.py \
  --checkpoint /path/to/checkpoint \
  --label_map artifacts/labels.json \
  --input_dir input/dataset_name \
  --track_aware \
  --max_tracks 8 \
  --max_polyphony 8
```

## Infer + reconstruction (VAE + soft concept bottleneck)

```bash
python -m interpretable_genre.infer_torch_cbm \
  --model artifacts/vae_cbm.pt \
  --label_map artifacts/labels.json \
  --midi_path midi/jazz_001.mid \
  --out_dir artifacts
```

## Infer + reconstruction (track-aware + interpolation)

```bash
python -m interpretable_genre.infer_torch_cbm \
  --model artifacts/ckpts/step_1500_val_2.2593.pt \
  --label_map artifacts/labels.json \
  --midi_path /path/to/file.mid \
  --out_dir artifacts \
  --track_aware \
  --max_tracks 8 \
  --max_polyphony 8 \
  --centroids_json /home/coder/xy/interpretable_genre/artifacts/centroids/step_1500_val_2.2593/centroids.json \
  --genre_shift_to jazz \
  --shift_scale 1.0 \
  --shift_steps 5 \
  --recon_threshold 0.05 \
  --debug_recon
```

Notes:
- The VAE uses per-measure piano-rolls with a fixed `steps_per_measure`. Measures are inferred from time signatures and then padded/truncated.
- Reconstruction is from concept space → latent → piano-roll → MIDI.
- Track-aware mode pads/clips to `--max_tracks` and caps density with `--max_polyphony`.

## Notes

- Workflow: encode each measure into a latent vector, project to a small concept space, aggregate concepts, then apply a linear classifier.
- Per-measure contributions are directly interpretable because measure-level concept scores sum to the final evidence.
- For best results, supply MIDI files with time signature metadata; otherwise a 4/4 default is used.
