from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from multiprocessing import Pool
import sys
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from interpretable_genre.weak_labels import build_stats_for_paths, generate_weak_label_for_path


def _find_midis(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".mid") or lower.endswith(".midi"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)



def _make_id(relpath: str) -> str:
    return hashlib.md5(relpath.encode("utf-8")).hexdigest()


def _load_genre_map(metadata_path: str | None, root: str) -> Dict[str, str]:
    if not metadata_path:
        return {}
    genre_map: Dict[str, str] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "genre" not in reader.fieldnames:
            raise ValueError("metadata must include path and genre columns")
        for row in reader:
            path = row["path"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(root, path))
            genre_map[path] = row["genre"]
    return genre_map


def _infer_genre_from_parent(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def _write_index(index_path: str, rows: List[Tuple[str, str, str, str]]) -> None:
    with open(index_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "genre", "split"])
        writer.writerows(rows)


def _write_split(split_path: str, rows: List[Tuple[str, str, str, str]], split_name: str) -> None:
    with open(split_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "genre"])
        for row in rows:
            if row[3] == split_name:
                writer.writerow(row[:3])


def _write_jsonl(path: str, items: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item))
            f.write("\n")


_STATS = None
_GENRE_MAP: Dict[str, str] = {}
_ROOT = ""


def _init_worker(stats: Dict[str, float], genre_map: Dict[str, str], root: str) -> None:
    global _STATS, _GENRE_MAP, _ROOT
    _STATS = stats
    _GENRE_MAP = genre_map
    _ROOT = root


def _weak_label_one(path: str) -> Optional[Dict[str, object]]:
    try:
        item = generate_weak_label_for_path(path, _STATS, _GENRE_MAP.get(path, ""))
    except Exception:
        return None
    rel = os.path.relpath(item["path"], _ROOT)
    item["id"] = _make_id(rel)
    item["relpath"] = rel
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak labels and split index for LMD.")
    parser.add_argument("--input_dir", help="Dataset directory containing MIDI files and metadata")
    parser.add_argument("--root", help="Root directory for LMD matched MIDI files")
    parser.add_argument("--out_dir", help="Output directory (default: root)")
    parser.add_argument("--metadata", help="Optional CSV with path,genre for labels")
    parser.add_argument("--split", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel workers for weak labels (0=cpu count)")
    parser.add_argument(
        "--infer_genre_from_parent",
        action="store_true",
        help="Use parent folder name as genre when metadata is not provided",
    )
    args = parser.parse_args()

    if args.input_dir:
        if not args.root:
            args.root = args.input_dir
        if not args.out_dir:
            args.out_dir = args.input_dir
        if not args.metadata:
            candidate = os.path.join(args.input_dir, "metadata_genre.csv")
            if os.path.exists(candidate):
                args.metadata = candidate
    if not args.root:
        raise ValueError("provide --root or --input_dir")

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir or args.root)
    os.makedirs(out_dir, exist_ok=True)

    paths = _find_midis(root)
    if not paths:
        raise ValueError("no MIDI files found under root")

    genre_map = _load_genre_map(args.metadata, root)
    if not genre_map and args.infer_genre_from_parent:
        genre_map = {path: _infer_genre_from_parent(path) for path in paths}

    relpaths = [os.path.relpath(path, root) for path in paths]
    ids = [_make_id(relpath) for relpath in relpaths]

    rng = random.Random(args.seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    split_point = int(round(len(indices) * (1.0 - args.split)))
    train_idx = set(indices[:split_point])

    rows = []
    for i, path in enumerate(tqdm(paths, desc="Indexing")):
        split = "train" if i in train_idx else "test"
        genre = genre_map.get(path, "")
        rows.append((ids[i], path, genre, split))

    index_path = os.path.join(out_dir, "index.csv")
    _write_index(index_path, rows)

    _write_split(os.path.join(out_dir, "train.csv"), rows, "train")
    _write_split(os.path.join(out_dir, "test.csv"), rows, "test")

    num_workers = args.num_workers if args.num_workers and args.num_workers > 0 else (os.cpu_count() or 1)
    stats = build_stats_for_paths(paths, progress=True, desc="Stats", num_workers=num_workers)
    weak_labels_path = os.path.join(out_dir, "weak_labels.jsonl")
    with open(weak_labels_path, "w", encoding="utf-8") as f:
        skipped = 0
        num_workers = args.num_workers if args.num_workers and args.num_workers > 0 else (os.cpu_count() or 1)
        if num_workers > 1:
            with Pool(processes=num_workers, initializer=_init_worker, initargs=(stats, genre_map, root)) as pool:
                for item in tqdm(pool.imap_unordered(_weak_label_one, paths), total=len(paths), desc="Weak labels"):
                    if not item:
                        skipped += 1
                        continue
                    if not item["measures"]:
                        skipped += 1
                    f.write(json.dumps(item))
                    f.write("\n")
        else:
            _init_worker(stats, genre_map, root)
            for path in tqdm(paths, desc="Weak labels"):
                item = _weak_label_one(path)
                if not item:
                    skipped += 1
                    continue
                if not item["measures"]:
                    skipped += 1
                f.write(json.dumps(item))
                f.write("\n")
        if skipped:
            sys.stderr.write(f"Skipped {skipped} files with unreadable MIDI data.\n")


if __name__ == "__main__":
    main()
