from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .midi_roll import (
    MeasureEvents,
    PITCHDUR_EOS,
    PITCHDUR_PAD,
    PROGRAM_EOS,
    PROGRAM_PAD,
    PROGRAM_VOCAB,
    events_to_midi,
    midi_to_measure_events,
)
from .torch_data import load_metadata_csv
from .torch_model import VAEConceptModel
from .utils import load_label_map, softmax


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(values)[::-1][:k]


def _aggregate_concepts(concepts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = concepts * mask.unsqueeze(-1)
    denom = mask.unsqueeze(-1).sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def _decode_from_logits(
    program_logits: torch.Tensor,
    pitchdur_logits: torch.Tensor,
    B: int,
    M: int,
    steps: int,
    polyphony: int,
    program_vocab: int,
    pitchdur_vocab: int,
):
    """
    program_logits: (B*M, T, program_vocab)
    """
    T = steps * polyphony

    program_ids = program_logits.argmax(dim=-1).view(B, M, T)
    pitchdur_ids = pitchdur_logits.argmax(dim=-1).view(B, M, T)

    program_ids = program_ids.view(B, M, steps, polyphony).cpu().numpy()
    pitchdur_ids = pitchdur_ids.view(B, M, steps, polyphony).cpu().numpy()

    recon_measures = []
    for i in range(M):
        tokens = np.stack([program_ids[0, i], pitchdur_ids[0, i]], axis=-1)
        recon_measures.append(MeasureEvents(index=i, tokens=tokens))
    return recon_measures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--midi_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--concept_shift_json")
    parser.add_argument("--shift_scale", type=float, default=1.0)
    parser.add_argument("--shift_steps", type=int, default=3)
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    config = ckpt["config"]

    model = VAEConceptModel(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"],
        num_concepts=config["num_concepts"],
        num_genres=config["num_genres"],
        hidden_dim=config["hidden_dim"],
        steps_per_measure=config["steps_per_measure"],
        max_polyphony=config["max_polyphony"],
        program_vocab=config["program_vocab"],
        pitchdur_vocab=config["pitchdur_vocab"],
        token_embed_dim=config["token_embed_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_map = load_label_map(args.label_map)
    inv_label = {idx: label for label, idx in label_map.items()}

    measures, time_sig, ticks_per_beat = midi_to_measure_events(
        args.midi_path,
        steps_per_beat=config["steps_per_beat"],
        max_polyphony=config["max_polyphony"],
        target_steps_per_measure=config["steps_per_measure"],
    )

    roll_stack = np.stack([m.tokens for m in measures], axis=0)

    rolls_tensor = torch.tensor(roll_stack, dtype=torch.long).unsqueeze(0)
    mask = torch.ones((1, roll_stack.shape[0]), dtype=torch.float32)

    with torch.no_grad():
        output = model(rolls_tensor, mask)

    # --- Genre prediction ---
    class_logits = output.class_logits.squeeze(0).cpu().numpy()
    probs = softmax(class_logits)
    top_idx = _top_k_indices(probs, args.top_k)

    print("\nTop genres:")
    for idx in top_idx:
        print(inv_label[int(idx)], float(probs[idx]))

    # --- Reconstruction from z ---
    B, M = rolls_tensor.shape[0], rolls_tensor.shape[1]
    steps = config["steps_per_measure"]
    polyphony = config["max_polyphony"]

    program_logits = output.recon_program_logits
    pitchdur_logits = output.recon_pitchdur_logits

    recon_measures = _decode_from_logits(
        program_logits,
        pitchdur_logits,
        B,
        M,
        steps,
        polyphony,
        config["program_vocab"],
        config["pitchdur_vocab"],
    )

    midi_out = events_to_midi(
        recon_measures,
        time_sig,
        ticks_per_beat,
        steps_per_beat=config["steps_per_beat"],
    )

    ckpt_name = Path(args.model).stem
    out_dir = Path(args.out_dir) / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_out.save(str(out_dir / "reconstruction.mid"))

    # --- Concept editing ---
    if args.concept_shift_json:
        with open(args.concept_shift_json) as f:
            shift_map = json.load(f)

        concept_names = config["concept_names"]
        shift_vec = torch.zeros(len(concept_names))

        for i, name in enumerate(concept_names):
            if name in shift_map:
                shift_vec[i] = shift_map[name]

        steps_interp = max(1, args.shift_steps)

        for i in range(steps_interp):
            alpha = (i / max(1, steps_interp - 1)) * args.shift_scale

            shifted_concepts = output.concepts + alpha * shift_vec

            # map to latent concept subspace
            z_concept_new = model.concept_to_latent(torch.sigmoid(shifted_concepts))

            # concatenate residual
            z_full = torch.cat([z_concept_new, output.z_residual], dim=-1)

            z_flat = z_full.view(B * M, -1)
            hidden = model.decoder(z_flat)

            program_logits = model.program_head(hidden).view(
                B * M,
                steps * polyphony,
                config["program_vocab"],
            )
            pitchdur_logits = model.pitchdur_head(hidden).view(
                B * M,
                steps * polyphony,
                config["pitchdur_vocab"],
            )

            recon_measures = _decode_from_logits(
                program_logits,
                pitchdur_logits,
                B,
                M,
                steps,
                polyphony,
                config["program_vocab"],
                config["pitchdur_vocab"],
            )

            midi_shifted = events_to_midi(
                recon_measures,
                time_sig,
                ticks_per_beat,
                steps_per_beat=config["steps_per_beat"],
            )

            midi_shifted.save(str(out_dir / f"interpolation_{i}.mid"))


if __name__ == "__main__":
    main()
