from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .midi_roll import midi_to_measure_rolls, midi_to_measure_rolls_by_track, rolls_to_midi, rolls_to_midi_by_track
from .torch_data import MidiConceptDataset, build_label_map, collate_song_batch, load_metadata_csv, load_weak_labels_jsonl
from .torch_model import VAEConceptModel, kl_divergence
from .utils import save_label_map


def _infer_concept_names(weak_labels_path: str) -> List[str]:
    with open(weak_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            measures = item.get("measures", [])
            if measures:
                return sorted(measures[0].get("concepts", {}).keys())
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE + concept bottleneck genre classifier.")
    parser.add_argument("--input_dir", help="Dataset directory containing train/test CSVs and weak labels")
    parser.add_argument("--metadata", help="CSV with path,genre columns (train)")
    parser.add_argument("--train_csv", help="Train CSV with path,genre columns")
    parser.add_argument("--test_csv", help="Test CSV with path,genre columns")
    parser.add_argument("--weak_labels", help="JSONL with per-measure concepts")
    parser.add_argument("--tokenized_manifest", help="JSON manifest mapping MIDI path -> tokenized .npz")
    parser.add_argument("--model_out", required=True, help="Output path for model checkpoint")
    parser.add_argument("--label_map_out", required=True, help="Output path for label map json")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps_per_beat", type=int, default=4)
    parser.add_argument("--steps_per_measure", type=int, default=16)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--concept_weight", type=float, default=1.0)
    parser.add_argument("--class_weight", type=float, default=1.0)
    parser.add_argument("--concept_recon_weight", type=float, default=0.5)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="interpretable-genre", help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None, help="Optional W&B run name")
    parser.add_argument("--tokenized_samples_dir", help="Save tokenized MIDI samples here (optional)")
    parser.add_argument("--tokenized_samples_max", type=int, default=10)
    parser.add_argument("--max_tracks", type=int, default=None, help="Max tracks for tokenized sample export")
    parser.add_argument("--max_polyphony", type=int, default=None, help="Max polyphony per step for rolls")
    parser.add_argument("--track_aware", action="store_true", help="Use track-aware modeling (fixed max_tracks)")
    parser.add_argument("--checkpoint_dir", help="Directory to save periodic checkpoints")
    parser.add_argument("--max_checkpoints", type=int, default=10)
    parser.add_argument("--ddp", action="store_true", help="Enable PyTorch DistributedDataParallel")
    parser.add_argument("--ddp_backend", default="nccl", help="DDP backend (nccl or gloo)")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP")
    parser.add_argument("--ddp_find_unused", action="store_true", help="Enable DDP find_unused_parameters")
    parser.add_argument("--wandb_train_log_every", type=int, default=100, help="W&B train loss log interval (steps)")
    parser.add_argument("--val_log_every", type=int, default=500, help="Val loss + checkpoint interval (steps)")
    args = parser.parse_args()

    if args.ddp and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.input_dir:
        if not args.train_csv:
            args.train_csv = os.path.join(args.input_dir, "train.csv")
        if not args.test_csv:
            args.test_csv = os.path.join(args.input_dir, "test.csv")
        if not args.weak_labels:
            args.weak_labels = os.path.join(args.input_dir, "weak_labels.jsonl")
        if not args.tokenized_manifest:
            manifest_path = os.path.join(args.input_dir, "tokenized_manifest.json")
            if os.path.exists(manifest_path):
                args.tokenized_manifest = manifest_path
    if not args.metadata and not args.train_csv:
        raise ValueError("provide --metadata or --train_csv or --input_dir")
    if not args.weak_labels:
        raise ValueError("provide --weak_labels or --input_dir")
    if args.track_aware and args.max_tracks is None:
        raise ValueError("--max_tracks is required when --track_aware is set")
    if args.tokenized_manifest and not args.track_aware:
        raise ValueError("--track_aware is required when using --tokenized_manifest")
    train_path = args.train_csv or args.metadata
    train_metadata = load_metadata_csv(train_path)
    label_map = build_label_map(train_metadata)
    if not label_map:
        raise ValueError("no genres found in metadata")

    concept_names = _infer_concept_names(args.weak_labels)
    if not concept_names:
        raise ValueError("no concepts found in weak labels file")

    weak_labels = load_weak_labels_jsonl(args.weak_labels)
    tokenized_manifest = None
    if args.tokenized_manifest:
        from .torch_data import load_tokenized_manifest

        tokenized_manifest = load_tokenized_manifest(args.tokenized_manifest)
    if args.ddp:
        dist.init_process_group(backend=args.ddp_backend)
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
    else:
        rank = 0
    dataset = MidiConceptDataset(
        train_metadata,
        weak_labels,
        label_map,
        concept_names,
        steps_per_beat=args.steps_per_beat,
        target_steps_per_measure=args.steps_per_measure,
        max_polyphony=args.max_polyphony,
        track_aware=args.track_aware,
        max_tracks=args.max_tracks,
        tokenized_manifest=tokenized_manifest,
    )
    if args.ddp:
        train_sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_song_batch,
        )
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_song_batch)
    test_loader = None
    if args.test_csv:
        test_metadata = load_metadata_csv(args.test_csv)
        test_dataset = MidiConceptDataset(
            test_metadata,
            weak_labels,
            label_map,
            concept_names,
            steps_per_beat=args.steps_per_beat,
            target_steps_per_measure=args.steps_per_measure,
            max_polyphony=args.max_polyphony,
            track_aware=args.track_aware,
            max_tracks=args.max_tracks,
            tokenized_manifest=tokenized_manifest,
        )
        if args.ddp:
            if rank == 0:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_song_batch,
                )
            else:
                test_loader = None
        else:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_song_batch)

    if args.track_aware:
        input_dim = args.steps_per_measure * 128 * args.max_tracks
    else:
        input_dim = args.steps_per_measure * 128
    model = VAEConceptModel(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        num_concepts=len(concept_names),
        num_genres=len(label_map),
        hidden_dim=args.hidden_dim,
    )

    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
    model.to(device)
    if args.ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.ddp_find_unused,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    recon_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    if args.tokenized_samples_dir and rank == 0:
        os.makedirs(args.tokenized_samples_dir, exist_ok=True)
        saved = 0
        for path, _ in train_metadata:
            if saved >= args.tokenized_samples_max:
                break
            try:
                tracks, time_sig, ticks_per_beat, programs, program_changes = midi_to_measure_rolls_by_track(
                    path,
                    steps_per_beat=args.steps_per_beat,
                    target_steps_per_measure=args.steps_per_measure,
                    max_tracks=args.max_tracks,
                    max_polyphony=args.max_polyphony,
                )
            except Exception:
                continue
            if not tracks:
                continue
            midi_out = rolls_to_midi_by_track(
                tracks,
                time_sig,
                ticks_per_beat,
                steps_per_beat=args.steps_per_beat,
                programs=programs,
                max_tracks=args.max_tracks,
                max_polyphony=args.max_polyphony,
                program_changes=program_changes,
            )
            out_path = os.path.join(args.tokenized_samples_dir, f"tokenized_{saved:02d}.mid")
            midi_out.save(out_path)
            saved += 1

    wandb_run = None
    if args.use_wandb and rank == 0:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed; pip install wandb or disable --use_wandb") from exc
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    checkpoint_records = []
    global_step = 0
    last_val_loss = None
    if args.checkpoint_dir and rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_steps = len(loader)
        if args.ddp and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for step_idx, batch in enumerate(loader, start=1):
            global_step += 1
            rolls = batch.rolls.to(device)
            concepts_target = batch.concepts.to(device)
            mask = batch.mask.to(device)
            labels = batch.labels.to(device)
            if rolls.shape[1] == 0:
                continue

            output = model(rolls, mask)

            target_flat = rolls.view(rolls.shape[0] * rolls.shape[1], -1)
            recon = recon_loss_fn(output.recon_logits, target_flat)
            recon = recon.mean(dim=1).mean()

            kl = kl_divergence(output.mu, output.logvar)

            pred_concepts = output.concepts.view(rolls.shape[0], rolls.shape[1], -1)
            diff = (pred_concepts - concepts_target) ** 2
            concept_loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0)

            class_logits = output.class_logits
            valid = labels >= 0
            if valid.any():
                class_loss = nn.functional.cross_entropy(class_logits[valid], labels[valid])
            else:
                # Keep classifier params in the graph so DDP doesn't see them as unused.
                class_loss = (class_logits * 0.0).sum()

            concept_recon_logits = model.module.decoder(output.concept_to_latent) if args.ddp else model.decoder(output.concept_to_latent)
            concept_recon = recon_loss_fn(concept_recon_logits, target_flat).mean(dim=1).mean()

            loss = (
                args.recon_weight * recon
                + args.kl_weight * kl
                + args.concept_weight * concept_loss
                + args.class_weight * class_loss
                + args.concept_recon_weight * concept_recon
            )

            optimizer.zero_grad()
            loss.backward()
            if args.ddp_find_unused and rank == 0:
                unused = [name for name, param in model.named_parameters() if param.grad is None]
                if unused:
                    print(f"DDP unused parameters: {', '.join(unused)}")
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            pct = 100.0 * step_idx / max(1, total_steps)
            if rank == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {step_idx}/{total_steps} ({pct:.1f}%)",
                    end="\r",
                )
            if wandb_run is not None and rank == 0 and global_step % args.wandb_train_log_every == 0:
                wandb_run.log(
                    {
                        "train_loss": float(loss.detach().cpu()),
                        "train_recon_loss": float(recon.detach().cpu()),
                        "train_kl_loss": float(kl.detach().cpu()),
                        "train_concept_loss": float(concept_loss.detach().cpu()),
                        "train_class_loss": float(class_loss.detach().cpu()),
                        "train_concept_recon_loss": float(concept_recon.detach().cpu()),
                        "step": global_step,
                    }
                )

            if (
                test_loader is not None
                and rank == 0
                and args.val_log_every > 0
                and global_step % args.val_log_every == 0
            ):
                model.eval()
                test_loss = 0.0
                recon_sum = 0.0
                kl_sum = 0.0
                concept_sum = 0.0
                class_sum = 0.0
                concept_recon_sum = 0.0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_loader):
                        if batch_idx >= 5:
                            break
                        rolls = batch.rolls.to(device)
                        concepts_target = batch.concepts.to(device)
                        mask = batch.mask.to(device)
                        labels = batch.labels.to(device)
                        if rolls.shape[1] == 0:
                            continue

                        output = model(rolls, mask)

                        target_flat = rolls.view(rolls.shape[0] * rolls.shape[1], -1)
                        recon = recon_loss_fn(output.recon_logits, target_flat)
                        recon = recon.mean(dim=1).mean()

                        kl = kl_divergence(output.mu, output.logvar)

                        pred_concepts = output.concepts.view(rolls.shape[0], rolls.shape[1], -1)
                        diff = (pred_concepts - concepts_target) ** 2
                        concept_loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0)

                        class_logits = output.class_logits
                        valid = labels >= 0
                        if valid.any():
                            class_loss = nn.functional.cross_entropy(class_logits[valid], labels[valid])
                        else:
                            class_loss = (class_logits * 0.0).sum()

                        concept_recon_logits = model.module.decoder(output.concept_to_latent) if args.ddp else model.decoder(output.concept_to_latent)
                        concept_recon = recon_loss_fn(concept_recon_logits, target_flat).mean(dim=1).mean()

                        loss = (
                            args.recon_weight * recon
                            + args.kl_weight * kl
                            + args.concept_weight * concept_loss
                            + args.class_weight * class_loss
                            + args.concept_recon_weight * concept_recon
                        )
                        test_loss += float(loss.detach().cpu())
                        recon_sum += float(recon.detach().cpu())
                        kl_sum += float(kl.detach().cpu())
                        concept_sum += float(concept_loss.detach().cpu())
                        class_sum += float(class_loss.detach().cpu())
                        concept_recon_sum += float(concept_recon.detach().cpu())
                test_avg = test_loss / max(1, min(5, len(test_loader)))
                last_val_loss = test_avg
                if wandb_run is not None:
                    denom = max(1, min(5, len(test_loader)))
                    wandb_run.log(
                        {
                            "val_loss": test_avg,
                            "val_recon_loss": recon_sum / denom,
                            "val_kl_loss": kl_sum / denom,
                            "val_concept_loss": concept_sum / denom,
                            "val_class_loss": class_sum / denom,
                            "val_concept_recon_loss": concept_recon_sum / denom,
                            "step": global_step,
                        }
                    )

                if args.checkpoint_dir:
                    val_tag = "na" if last_val_loss is None else f"{last_val_loss:.4f}"
                    ckpt_path = os.path.join(
                        args.checkpoint_dir,
                        f"step_{global_step}_val_{val_tag}.pt",
                    )
                    torch.save(
                        {
                            "model_state": model.module.state_dict() if args.ddp else model.state_dict(),
                            "config": {
                                "input_dim": input_dim,
                                "latent_dim": args.latent_dim,
                                "hidden_dim": args.hidden_dim,
                                "num_concepts": len(concept_names),
                                "num_genres": len(label_map),
                                "concept_names": concept_names,
                                "steps_per_beat": args.steps_per_beat,
                                "steps_per_measure": args.steps_per_measure,
                                "track_aware": args.track_aware,
                                "max_tracks": args.max_tracks,
                                "max_polyphony": args.max_polyphony,
                            },
                            "step": global_step,
                            "val_loss": last_val_loss,
                        },
                        ckpt_path,
                    )
                    if last_val_loss is not None:
                        checkpoint_records.append((ckpt_path, float(last_val_loss)))
                        checkpoint_records.sort(key=lambda x: x[1])
                        while len(checkpoint_records) > args.max_checkpoints:
                            remove_path, _ = checkpoint_records.pop(-1)
                            if os.path.exists(remove_path):
                                os.remove(remove_path)
                    else:
                        checkpoint_records.append((ckpt_path, float("inf")))
                        if len(checkpoint_records) > args.max_checkpoints:
                            remove_path, _ = checkpoint_records.pop(0)
                            if os.path.exists(remove_path):
                                os.remove(remove_path)
                model.train()

        if rank == 0:
            avg = total_loss / max(1, len(loader))
            print(f"epoch {epoch + 1}/{args.epochs} avg_loss={avg:.4f}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        torch.save(
            {
                "model_state": model.module.state_dict() if args.ddp else model.state_dict(),
                "config": {
                    "input_dim": input_dim,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_concepts": len(concept_names),
                    "num_genres": len(label_map),
                    "concept_names": concept_names,
                    "steps_per_beat": args.steps_per_beat,
                    "steps_per_measure": args.steps_per_measure,
                    "track_aware": args.track_aware,
                    "max_tracks": args.max_tracks,
                    "max_polyphony": args.max_polyphony,
                },
            },
            out_path,
        )
        save_label_map(args.label_map_out, label_map)
    if wandb_run is not None:
        wandb_run.finish()
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
