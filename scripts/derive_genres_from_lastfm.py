from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

from tqdm import tqdm


def _find_midis(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if lower.endswith(".mid") or lower.endswith(".midi"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def _extract_track_id(path: str) -> str | None:
    parts = path.split(os.sep)
    for part in parts:
        if part.startswith("TR") and len(part) >= 18:
            return part[:18]
    return None


def _lastfm_json_path(root: str, track_id: str) -> str:
    return os.path.join(root, track_id[2], track_id[3], track_id[4], f"{track_id}.json")


def _load_tags(path: str) -> List[Tuple[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = []
    for item in data.get("tags", []):
        if len(item) < 2:
            continue
        tag = str(item[0]).strip()
        try:
            weight = float(item[1])
        except ValueError:
            weight = 0.0
        tags.append((tag, weight))
    return tags



def _normalize(text: str) -> str:
    return text.lower().replace("-", " ").replace("_", " ")


def _genre_keywords() -> Dict[str, List[str]]:
    return {
        "rock": ["rock", "alternative", "indie", "grunge", "classic rock"],
        "pop": ["pop", "dance pop", "synthpop", "electropop"],
        "hiphop": ["hip hop", "hiphop", "rap"],
        "electronic": ["electronic", "edm", "house", "techno", "trance", "dubstep", "drum and bass", "dnb"],
        "jazz": ["jazz", "swing", "bebop", "fusion"],
        "classical": ["classical", "baroque", "romantic", "opera"],
        "metal": ["metal", "heavy metal", "thrash", "death metal", "black metal", "power metal"],
        "country": ["country", "bluegrass"],
        "folk": ["folk", "singer songwriter"],
        "blues": ["blues"],
        "reggae": ["reggae", "ska", "dub"],
        "rnb": ["r&b", "rnb", "soul", "neo soul"],
        "latin": ["latin", "salsa", "bossa", "reggaeton", "bachata"],
    }


def _score_genres(tags: List[Tuple[str, float]]) -> Dict[str, float]:
    keywords = _genre_keywords()
    scores = {genre: 0.0 for genre in keywords}
    for tag, weight in tags:
        norm = _normalize(tag)
        for genre, keys in keywords.items():
            if any(key in norm for key in keys):
                scores[genre] += weight
    return scores


def _pick_genre(scores: Dict[str, float], min_score: float, min_ratio: float) -> Tuple[str, float]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] < min_score:
        return "", 0.0
    if len(ranked) > 1 and ranked[1][1] > 0 and ranked[0][1] < ranked[1][1] * min_ratio:
        return "", 0.0
    return ranked[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive coarse genres from Last.fm tags.")
    parser.add_argument("--lmd_root", required=True, help="Root of lmd_matched")
    parser.add_argument("--lastfm_train", required=True, help="Root of lastfm_train")
    parser.add_argument("--lastfm_test", required=True, help="Root of lastfm_test")
    parser.add_argument("--out_csv", required=True, help="Output CSV with path,genre,track_id")
    parser.add_argument("--min_score", type=float, default=30.0, help="Minimum tag score to assign genre")
    parser.add_argument("--min_ratio", type=float, default=1.1, help="Top score must exceed runner-up by ratio")
    args = parser.parse_args()

    midi_paths = _find_midis(args.lmd_root)
    if not midi_paths:
        raise ValueError("no MIDI files found under lmd_root")

    rows = []
    for path in tqdm(midi_paths, desc="Deriving genres"):
        track_id = _extract_track_id(path)
        if not track_id:
            continue
        train_path = _lastfm_json_path(args.lastfm_train, track_id)
        test_path = _lastfm_json_path(args.lastfm_test, track_id)
        if os.path.exists(train_path):
            tags = _load_tags(train_path)
        elif os.path.exists(test_path):
            tags = _load_tags(test_path)
        else:
            tags = []
        if tags:
            scores = _score_genres(tags)
            genre, score = _pick_genre(scores, args.min_score, args.min_ratio)
        else:
            genre, score = "", 0.0
        rows.append((path, genre, track_id, f"{score:.2f}"))

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "genre", "track_id", "genre_score"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
