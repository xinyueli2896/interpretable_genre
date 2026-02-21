from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .midi_roll import MeasureRoll, midi_to_measure_rolls, midi_to_measure_rolls_by_track, rolls_to_midi, rolls_to_midi_by_track
from .torch_data import load_metadata_csv
from .torch_model import VAEConceptModel
from .utils import load_label_map, softmax


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(values)[::-1][:k]


def _aggregate_concepts(concepts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    concepts = concepts.view(mask.shape[0], mask.shape[1], -1)
    masked = concepts * mask.unsqueeze(-1)
    denom = mask.unsqueeze(-1).sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def _song_concept_vector(
    model: VAEConceptModel,
    midi_path: str,
    config: dict,
    track_aware: bool,
    max_tracks: int | None,
    max_polyphony: int | None,
) -> np.ndarray:
    if track_aware:
        tracks, _, _, _, _ = midi_to_measure_rolls_by_track(
            midi_path,
            steps_per_beat=config["steps_per_beat"],
            target_steps_per_measure=config["steps_per_measure"],
            max_tracks=max_tracks,
            max_polyphony=max_polyphony,
        )
        if not tracks:
            raise ValueError("no measures found in MIDI")
        per_track = []
        for track in tracks:
            per_track.append(np.stack([m.roll for m in track], axis=0))
        roll_stack = np.stack(per_track, axis=1)
    else:
        rolls, _, _, _ = midi_to_measure_rolls(
            midi_path,
            steps_per_beat=config["steps_per_beat"],
            target_steps_per_measure=config["steps_per_measure"],
            max_polyphony=max_polyphony,
        )
        if not rolls:
            raise ValueError("no measures found in MIDI")
        roll_stack = np.stack([m.roll for m in rolls], axis=0)

    rolls_tensor = torch.tensor(roll_stack, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones((1, roll_stack.shape[0]), dtype=torch.float32)
    with torch.no_grad():
        output = model(rolls_tensor, mask)
    return _aggregate_concepts(output.concepts, mask).squeeze(0).cpu().numpy()


def _normalize_rolls(rolls: np.ndarray, config: dict) -> np.ndarray:
    if not config.get("track_aware", False):
        return rolls
    max_tracks = int(config.get("max_tracks", 1))
    steps_per_measure = int(config.get("steps_per_measure", rolls.shape[2]))
    if rolls.ndim != 4:
        raise ValueError("expected track-aware rolls with shape (measures, tracks, steps, 128)")
    measures, tracks, steps, pitches = rolls.shape
    if pitches != 128:
        raise ValueError("expected 128 pitch bins")
    if tracks < max_tracks:
        rolls = np.pad(rolls, ((0, 0), (0, max_tracks - tracks), (0, 0), (0, 0)), mode="constant")
    elif tracks > max_tracks:
        rolls = rolls[:, :max_tracks]
    if steps < steps_per_measure:
        rolls = np.pad(rolls, ((0, 0), (0, 0), (0, steps_per_measure - steps), (0, 0)), mode="constant")
    elif steps > steps_per_measure:
        rolls = rolls[:, :, :steps_per_measure, :]
    return rolls
def main() -> None:
    parser = argparse.ArgumentParser(description="Infer genres and generate explanation from VAE-CBM model.")
    parser.add_argument("--model", required=True, help="Path to torch checkpoint")
    parser.add_argument("--label_map", required=True, help="Path to label map json")
    parser.add_argument("--midi_path", required=True, help="MIDI file to analyze")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k genres to report")
    parser.add_argument("--recon_threshold", type=float, default=0.5, help="Threshold for reconstruction")
    parser.add_argument("--max_polyphony", type=int, default=8, help="Max polyphony per step for rolls")
    parser.add_argument("--track_aware", action="store_true", help="Use track-aware modeling")
    parser.add_argument("--max_tracks", type=int, default=8, help="Max tracks for track-aware modeling")
    parser.add_argument("--concept_shift_json", help="JSON mapping concept name -> delta")
    parser.add_argument("--shift_scale", type=float, default=1.0, help="Scale for concept shift interpolation")
    parser.add_argument("--shift_steps", type=int, default=3, help="Number of interpolation steps")
    parser.add_argument("--steps_per_beat", default = 16, type=int, help="Override steps per beat for inference")
    parser.add_argument("--steps_per_measure",default = 64,  type=int, help="Override steps per measure for inference")
    parser.add_argument("--centroid_metadata", help="CSV with path,genre for centroid computation")
    parser.add_argument("--centroid_top_n", type=int, default=10, help="Top-N songs for centroid averaging")
    parser.add_argument("--genre_shift_to", help="Target genre for centroid shift")
    parser.add_argument("--centroids_json", help="Precomputed centroids.json to use for centroid shift")
    parser.add_argument("--debug_recon", action="store_true", help="Print reconstruction debug stats")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
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
    inv_label = {idx: label for label, idx in label_map.items()}

    track_aware = args.track_aware or bool(config.get("track_aware", False))
    max_tracks = args.max_tracks or config.get("max_tracks")
    if track_aware and max_tracks is None:
        raise ValueError("--max_tracks is required for track-aware inference")
    if config.get("steps_per_beat") is not None and args.steps_per_beat is not None:
        if args.steps_per_beat != config["steps_per_beat"]:
            print(
                f"Warning: steps_per_beat mismatch (ckpt={config['steps_per_beat']}, "
                f"args={args.steps_per_beat})"
            )
    if config.get("steps_per_measure") is not None and args.steps_per_measure is not None:
        if args.steps_per_measure != config["steps_per_measure"]:
            print(
                f"Warning: steps_per_measure mismatch (ckpt={config['steps_per_measure']}, "
                f"args={args.steps_per_measure})"
            )
    if config.get("max_tracks") is not None and args.max_tracks is not None:
        if args.max_tracks != config["max_tracks"]:
            print(f"Warning: max_tracks mismatch (ckpt={config['max_tracks']}, args={args.max_tracks})")
    if config.get("max_polyphony") is not None and args.max_polyphony is not None:
        if args.max_polyphony != config["max_polyphony"]:
            print(
                f"Warning: max_polyphony mismatch (ckpt={config['max_polyphony']}, "
                f"args={args.max_polyphony})"
            )

    if track_aware:
        tracks, time_sig, ticks_per_beat, programs, program_changes = midi_to_measure_rolls_by_track(
            args.midi_path,
            steps_per_beat=config["steps_per_beat"],
            target_steps_per_measure=config["steps_per_measure"],
            max_tracks=max_tracks,
            max_polyphony=args.max_polyphony,
        )
        if not tracks:
            raise ValueError("no measures found in MIDI")
        per_track = []
        for track in tracks:
            per_track.append(np.stack([m.roll for m in track], axis=0))
        roll_stack = np.stack(per_track, axis=1)  # (measures, tracks, steps, 128)
        roll_stack = _normalize_rolls(roll_stack, config)
    else:
        rolls, time_sig, ticks_per_beat, programs = midi_to_measure_rolls(
            args.midi_path,
            steps_per_beat=config["steps_per_beat"],
            target_steps_per_measure=config["steps_per_measure"],
            max_polyphony=args.max_polyphony,
        )
        if not rolls:
            raise ValueError("no measures found in MIDI")
        roll_stack = np.stack([m.roll for m in rolls], axis=0)

    rolls_tensor = torch.tensor(roll_stack, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones((1, roll_stack.shape[0]), dtype=torch.float32)

    if args.debug_recon:
        input_active = float((roll_stack > 0.5).sum())
        input_total = float(roll_stack.size)
        input_density = (input_active / input_total) if input_total else 0.0
        print(f"input_rolls active: {input_active:.0f}/{input_total:.0f} ({input_density:.6f})")

    with torch.no_grad():
        output = model(rolls_tensor, mask)

    class_logits = output.class_logits.squeeze(0).cpu().numpy()
    probs = softmax(class_logits)
    top_idx = _top_k_indices(probs, args.top_k)

    concepts = output.concepts.view(roll_stack.shape[0], -1).cpu().numpy()
    concept_names = config.get("concept_names", [f"concept_{i}" for i in range(concepts.shape[1])])
    weights = model.classifier.weight.detach().cpu().numpy()
    per_genre = []
    for idx in top_idx:
        scores = concepts @ weights[idx]
        top_measures = _top_k_indices(scores, min(5, len(scores)))
        per_genre.append(
            {
                "genre": inv_label[int(idx)],
                "confidence": float(probs[idx]),
                "top_measures": [
                    {"measure_index": int(m), "score": float(scores[m])} for m in top_measures
                ],
            }
        )

    concept_latent = output.concept_to_latent
    recon_logits = model.decoder(concept_latent)
    with torch.no_grad():
        recon_min = float(recon_logits.min().item())
        recon_max = float(recon_logits.max().item())
    print(f"recon_logits min/max: {recon_min:.4f}/{recon_max:.4f}")
    if track_aware:
        recon_logits = recon_logits.view(
            roll_stack.shape[0],
            roll_stack.shape[1],
            roll_stack.shape[2],
            roll_stack.shape[3],
        )
        recon = torch.sigmoid(recon_logits).detach().cpu().numpy()
        recon_tracks = []
        for track_idx in range(recon.shape[1]):
            track_rolls = []
            for i in range(recon.shape[0]):
                roll = (recon[i, track_idx] >= args.recon_threshold).astype(np.float32)
                track_rolls.append(MeasureRoll(index=i, roll=roll))
            recon_tracks.append(track_rolls)
        midi_out = rolls_to_midi_by_track(
            recon_tracks,
            time_sig,
            ticks_per_beat,
            steps_per_beat=config["steps_per_beat"],
            programs=programs,
            max_tracks=max_tracks,
            max_polyphony=args.max_polyphony,
            program_changes=program_changes,
        )
    else:
        recon_logits = recon_logits.view(roll_stack.shape[0], roll_stack.shape[1], 128)
        recon = torch.sigmoid(recon_logits).detach().cpu().numpy()
        recon_rolls = []
        for i in range(recon.shape[0]):
            roll = (recon[i] >= args.recon_threshold).astype(np.float32)
            recon_rolls.append(MeasureRoll(index=i, roll=roll))
        midi_out = rolls_to_midi(
            recon_rolls,
            time_sig,
            ticks_per_beat,
            steps_per_beat=config["steps_per_beat"],
            programs=programs,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "genre_explanation.json"
    midi_path = out_dir / "reconstruction.mid"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "midi_path": args.midi_path,
                "top_genres": per_genre,
            },
            f,
            indent=2,
        )
    midi_out.save(str(midi_path))

    base_name = Path(args.midi_path).stem
    original_genre = per_genre[0]["genre"] if per_genre else "unknown"
    print(f"Predicted genre: {original_genre}")

    if args.concept_shift_json:
        with open(args.concept_shift_json, "r", encoding="utf-8") as f:
            shift_map = json.load(f)
        shift_vec = np.zeros(concepts.shape[1], dtype=np.float32)
        for idx, name in enumerate(concept_names):
            if name in shift_map:
                shift_vec[idx] = float(shift_map[name])
        shift_vec = torch.tensor(shift_vec, dtype=torch.float32)

        steps = max(1, args.shift_steps)
        for i in range(steps):
            alpha = (i / max(1, steps - 1)) * args.shift_scale
            shifted = output.concepts + alpha * shift_vec.to(output.concepts.device)
            shifted_latent = model.concept_to_latent(shifted)
            shifted_logits = model.decoder(shifted_latent)
            if track_aware:
                shifted_logits = shifted_logits.view(
                    roll_stack.shape[0],
                    roll_stack.shape[1],
                    roll_stack.shape[2],
                    roll_stack.shape[3],
                )
                shifted_rolls = torch.sigmoid(shifted_logits).detach().cpu().numpy()
                recon_tracks = []
                for track_idx in range(shifted_rolls.shape[1]):
                    track_rolls = []
                    for m in range(shifted_rolls.shape[0]):
                        roll = (shifted_rolls[m, track_idx] >= args.recon_threshold).astype(np.float32)
                        track_rolls.append(MeasureRoll(index=m, roll=roll))
                    recon_tracks.append(track_rolls)
                shifted_midi = rolls_to_midi_by_track(
                    recon_tracks,
                    time_sig,
                    ticks_per_beat,
                    steps_per_beat=config["steps_per_beat"],
                    programs=programs,
                    max_tracks=max_tracks,
                    max_polyphony=args.max_polyphony,
                    program_changes=program_changes,
                )
            else:
                shifted_logits = shifted_logits.view(roll_stack.shape[0], roll_stack.shape[1], 128)
                shifted_rolls = torch.sigmoid(shifted_logits).detach().cpu().numpy()
                recon_rolls = []
                for m in range(shifted_rolls.shape[0]):
                    roll = (shifted_rolls[m] >= args.recon_threshold).astype(np.float32)
                    recon_rolls.append(MeasureRoll(index=m, roll=roll))
                shifted_midi = rolls_to_midi(
                    recon_rolls,
                    time_sig,
                    ticks_per_beat,
                    steps_per_beat=config["steps_per_beat"],
                    programs=programs,
                )
            shifted_path = out_dir / f"interpolation_{i}.mid"
            shifted_midi.save(str(shifted_path))

    if (args.centroid_metadata or args.centroids_json) and args.genre_shift_to:
        if args.centroids_json:
            with open(args.centroids_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            centroid_map = data.get("centroids", {})
            if args.genre_shift_to not in centroid_map:
                raise ValueError("genre_shift_to not found in centroids_json")
            centroid_to = np.array(centroid_map[args.genre_shift_to], dtype=float)
        else:
            metadata = load_metadata_csv(args.centroid_metadata)
            if not metadata:
                raise ValueError("centroid_metadata is empty")
            to_paths = [path for path, genre in metadata if genre == args.genre_shift_to]
            if not to_paths:
                raise ValueError("genre_shift_to not found in centroid_metadata")
            to_vectors = [
                _song_concept_vector(model, p, config, track_aware, max_tracks, args.max_polyphony)
                for p in to_paths
            ]
            centroid_to = np.mean(to_vectors, axis=0)

        current_vec = _aggregate_concepts(output.concepts, mask).squeeze(0).cpu().numpy()
        direction = torch.tensor(centroid_to - current_vec, dtype=torch.float32)

        steps = max(1, args.shift_steps)
        for i in range(steps):
            alpha = (i / max(1, steps - 1)) * args.shift_scale
            shifted = output.concepts + alpha * direction.to(output.concepts.device)
            shifted_latent = model.concept_to_latent(shifted)
            shifted_logits = model.decoder(shifted_latent)
            if track_aware:
                shifted_logits = shifted_logits.view(
                    roll_stack.shape[0],
                    roll_stack.shape[1],
                    roll_stack.shape[2],
                    roll_stack.shape[3],
                )
                shifted_rolls = torch.sigmoid(shifted_logits).detach().cpu().numpy()
                recon_tracks = []
                for track_idx in range(shifted_rolls.shape[1]):
                    track_rolls = []
                    for m in range(shifted_rolls.shape[0]):
                        roll = (shifted_rolls[m, track_idx] >= args.recon_threshold).astype(np.float32)
                        track_rolls.append(MeasureRoll(index=m, roll=roll))
                    recon_tracks.append(track_rolls)
                shifted_midi = rolls_to_midi_by_track(
                    recon_tracks,
                    time_sig,
                    ticks_per_beat,
                    steps_per_beat=config["steps_per_beat"],
                    programs=programs,
                    max_tracks=max_tracks,
                    max_polyphony=args.max_polyphony,
                    program_changes=program_changes,
                )
            else:
                shifted_logits = shifted_logits.view(roll_stack.shape[0], roll_stack.shape[1], 128)
                shifted_rolls = torch.sigmoid(shifted_logits).detach().cpu().numpy()
                recon_rolls = []
                for m in range(shifted_rolls.shape[0]):
                    roll = (shifted_rolls[m] >= args.recon_threshold).astype(np.float32)
                    recon_rolls.append(MeasureRoll(index=m, roll=roll))
                shifted_midi = rolls_to_midi(
                    recon_rolls,
                    time_sig,
                    ticks_per_beat,
                    steps_per_beat=config["steps_per_beat"],
                    programs=programs,
                )
            shifted_path = out_dir / (
                f"{base_name}_{original_genre}_to_{args.genre_shift_to}"
                f"_scale_{args.shift_scale}_steps_{args.shift_steps}_{i}.mid"
            )
            shifted_midi.save(str(shifted_path))


if __name__ == "__main__":
    main()
