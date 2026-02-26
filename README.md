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
  --root /path/to/midi_folder \
  --out_dir input/dataset_name \
  --split 0.2 \
  --num_workers 0
```

Outputs:
- `train.csv`, `test.csv` under the `--out_dir`
- `weak_labels.jsonl` under the `--out_dir`

### 2) Tokenize MIDI files (event tokens with channel-aware programs)
To speed up the training, we precompute MIDI into tokens. Program IDs are stored as
`channel * 128 + program`, so you must re-tokenize after this change.

```bash
python scripts/tokenize_midis.py \
  --root /path/to/midi_folder \
  --out_dir input/dataset_name/tokenized_npz \
  --manifest_out input/dataset_name/tokenized_manifest.json \
  --steps_per_beat 4 \
  --steps_per_measure 16 \
  --min_aligned_ratio 0.7 \
  --align_tolerance_steps 0.25 \
  --max_polyphony 8 \
  --num_workers 0
```

`--min_aligned_ratio` filters out off-grid MIDIs by checking the fraction of note onsets
that land within `--align_tolerance_steps` of the quantization grid.

### Optional: one-shot dataset prep

```bash
TOKENIZE_ONLY=1 bash scripts/prepare_dataset.sh /path/to/midi_folder 0.2
```

This writes outputs to `input/<dataset_name>` where `<dataset_name>` is the basename of the MIDI folder.

To point to an external metadata CSV:

```bash
METADATA_PATH=/path/to/metadata_genre.csv \
  bash scripts/prepare_dataset.sh /path/to/midi_folder 0.2
```

To control the alignment filter when tokenizing:

```bash
MIN_ALIGNED_RATIO=0.7 ALIGN_TOLERANCE_STEPS=0.25 \
  bash scripts/prepare_dataset.sh /path/to/midi_folder 0.2
```

### 3) Train (VAE + soft concept bottleneck)
The model itself. 

```bash
python -m interpretable_genre.train_torch_cbm \
  --input_dir input/dataset_name \
  --label_map_out artifacts/labels.json \
  --max_polyphony 8 \
  --steps_per_beat 4 \
  --steps_per_measure 16
```

To debug reconstruction, you can overfit on a single song:

```bash
python -m interpretable_genre.train_torch_cbm \
  --input_dir input/dataset_name \
  --label_map_out artifacts/labels.json \
  --overfit_one
```

If you want to isolate reconstruction loss only:

```bash
python -m interpretable_genre.train_torch_cbm \
  --input_dir input/dataset_name \
  --label_map_out artifacts/labels.json \
  --overfit_one \
  --recon_only \
  --batch_size 1
```



## Precompute centroids for interpolation

```bash
python scripts/compute_centroids.py \
  --model artifacts/ckpts/v6/step_2500_val_12.1460.pt \
  --metadata /home/coder/xy/data/lmd_matched/metadata_genre.csv \
  --label_map artifacts/labels.json \
  --input_dir input/A \
  --out_root artifacts/centroids 
```

## Evaluate classification accuracy

```bash
python scripts/eval_accuracy.py \
  --checkpoint artifacts/ckpts/v6/step_2500_val_12.1460.pt \
  --label_map artifacts/labels.json \
  --input_dir input/A \
  --max_polyphony 8
```

Note: evaluation expects tokenized data produced by the current tokenizer
(channel-aware program IDs). Re-tokenize if you trained with older tokens.



## Visualize concept space (GIF)

After computing centroids, you can render a 3D GIF of concept vectors over checkpoints:

```bash
python scripts/plot_concept_gif.py \
  --dirs artifacts/centroids/step_2500_val_12.1460 \
  --out_gif artifacts/concept_scatter.gif \
  --drop_dim 3 \
  --top_genres 3
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
  --max_polyphony 8 \
  --centroids_json /home/coder/xy/interpretable_genre/artifacts/centroids/step_1500_val_2.2593/centroids.json \
  --genre_shift_to jazz \
  --shift_scale 1.0 \
  --shift_steps 5 \
  --recon_threshold 0.05 \
  --debug_recon
```

Notes:
- Tokenization uses StreamMUSE-style event tensors: 1/16 beat steps, each step stores up to `max_polyphony` notes as `[program_id, pitch+duration]`.
- Reconstruction is from concept space → latent → event tokens → MIDI.
- `max_tracks` limits how many MIDI tracks are parsed into events; `max_polyphony` caps per-step note count.

## Notes

- Workflow: encode each measure into a latent vector, project to a small concept space, aggregate concepts, then apply a linear classifier.
- Per-measure contributions are directly interpretable because measure-level concept scores sum to the final evidence.
- For best results, supply MIDI files with time signature metadata; otherwise a 4/4 default is used.
- Best checkpoint so far: https://drive.google.com/file/d/1cv3SaMUsSlg9jBho48HUbGvBx_MO54cM/view?usp=sharing
