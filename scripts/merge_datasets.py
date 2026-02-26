from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def _read_csv(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "genre" not in reader.fieldnames:
            raise ValueError(f"{path} missing path/genre columns")
        return [(row["path"], row["genre"]) for row in reader]


def _write_csv(path: str, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "genre"])
        writer.writeheader()
        for midi_path, genre in rows:
            writer.writerow({"path": midi_path, "genre": genre})


def _read_jsonl(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def _read_manifest(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def _merge_metadata(
    all_rows: List[Tuple[str, str]], dedupe: bool
) -> List[Tuple[str, str]]:
    if not dedupe:
        return all_rows
    merged: Dict[str, str] = {}
    for midi_path, genre in all_rows:
        if midi_path in merged and merged[midi_path] != genre:
            print(f"Warning: conflicting genre for {midi_path}: {merged[midi_path]} vs {genre}")
        merged.setdefault(midi_path, genre)
    return list(merged.items())


def _merge_manifests(manifests: List[Dict[str, str]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for manifest in manifests:
        for key, value in manifest.items():
            if key in merged and merged[key] != value:
                print(f"Warning: conflicting manifest entry for {key}: {merged[key]} vs {value}")
            merged.setdefault(key, value)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple input datasets into one.")
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Input dataset dirs (each has train/test/weak_labels/manifest)")
    parser.add_argument("--out_dir", required=True, help="Output dataset dir (e.g., input/combined)")
    parser.add_argument("--dedupe", action="store_true", help="Dedupe by path (default: true)")
    parser.add_argument("--no_dedupe", action="store_true", help="Disable dedupe by path")
    args = parser.parse_args()

    dedupe = True
    if args.no_dedupe:
        dedupe = False
    if args.dedupe:
        dedupe = True

    os.makedirs(args.out_dir, exist_ok=True)

    train_rows: List[Tuple[str, str]] = []
    test_rows: List[Tuple[str, str]] = []
    weak_lines: List[str] = []
    manifests: List[Dict[str, str]] = []

    for input_dir in args.input_dirs:
        train_path = os.path.join(input_dir, "train.csv")
        test_path = os.path.join(input_dir, "test.csv")
        weak_path = os.path.join(input_dir, "weak_labels.jsonl")
        manifest_path = os.path.join(input_dir, "tokenized_manifest.json")

        if os.path.exists(train_path):
            train_rows.extend(_read_csv(train_path))
        if os.path.exists(test_path):
            test_rows.extend(_read_csv(test_path))
        weak_lines.extend(_read_jsonl(weak_path))
        manifests.append(_read_manifest(manifest_path))

    train_rows = _merge_metadata(train_rows, dedupe)
    test_rows = _merge_metadata(test_rows, dedupe)
    merged_manifest = _merge_manifests(manifests)

    _write_csv(os.path.join(args.out_dir, "train.csv"), train_rows)
    _write_csv(os.path.join(args.out_dir, "test.csv"), test_rows)

    with open(os.path.join(args.out_dir, "weak_labels.jsonl"), "w", encoding="utf-8") as f:
        for line in weak_lines:
            f.write(line.rstrip("\n") + "\n")

    with open(os.path.join(args.out_dir, "tokenized_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(merged_manifest, f, indent=2)

    print(f"Wrote merged dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
