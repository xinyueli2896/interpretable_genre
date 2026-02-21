from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm


def _load_vectors(path: str) -> Tuple[np.ndarray, List[str]]:
    vectors = []
    genres = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            vectors.append(item["concept_vector"])
            genres.append(item.get("genre", ""))
    return np.array(vectors, dtype=float), genres


def _load_centroids(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    centroids = {}
    for genre, vec in data.get("centroids", {}).items():
        centroids[genre] = np.array(vec, dtype=float)
    return centroids


def _sample_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(0)
    return rng.choice(n, size=max_points, replace=False)


def _collect_genres(dirs: List[str], top_k: int) -> List[str]:
    counts: Dict[str, int] = {}
    for d in dirs:
        vec_path = os.path.join(d, "vectors.jsonl")
        if not os.path.exists(vec_path):
            continue
        _, g = _load_vectors(vec_path)
        for item in g:
            if not item:
                continue
            counts[item] = counts.get(item, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [genre for genre, _ in ranked[:top_k]]


def _drop_dimension(x: np.ndarray, drop_dim: int) -> np.ndarray:
    if x.shape[1] <= drop_dim or drop_dim < 0:
        raise ValueError("drop_dim is out of range for concept vectors")
    return np.delete(x, drop_dim, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot animated GIF of concept vectors and centroids.")
    parser.add_argument("--dirs", nargs="+", required=True, help="Centroid directories (each with vectors.jsonl)")
    parser.add_argument("--out_gif", required=True, help="Output GIF path")
    parser.add_argument("--max_points", type=int, default=5000, help="Max points to plot per frame")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--figsize", type=float, nargs=2, default=(7.0, 6.0))
    parser.add_argument("--drop_dim", type=int, default=3, help="Which concept dimension to discard")
    parser.add_argument("--top_genres", type=int, default=3, help="Show only top-N genres by count")
    args = parser.parse_args()

    dirs = args.dirs
    genres = _collect_genres(dirs, args.top_genres)
    cmap = plt.get_cmap("tab20")
    color_map = {genre: cmap(i % 20) for i, genre in enumerate(genres)}

    frames = []
    for d in tqdm(dirs, desc="Frames"):
        vec_path = os.path.join(d, "vectors.jsonl")
        cent_path = os.path.join(d, "centroids.json")
        if not os.path.exists(vec_path) or not os.path.exists(cent_path):
            continue
        vectors, labels = _load_vectors(vec_path)
        centroids = _load_centroids(cent_path)
        if vectors.size == 0:
            continue
        idx = _sample_indices(vectors.shape[0], args.max_points)
        vectors = vectors[idx]
        labels = [labels[i] for i in idx]

        proj = _drop_dimension(vectors, args.drop_dim)
        fig = plt.figure(figsize=tuple(args.figsize))
        ax = fig.add_subplot(111, projection="3d")
        for genre in genres:
            mask = [g == genre for g in labels]
            if not any(mask):
                continue
            pts = proj[mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, alpha=0.55, color=color_map[genre], label=genre)

        for genre, vec in centroids.items():
            if genre not in color_map:
                continue
            centroid_proj = _drop_dimension(vec.reshape(1, -1), args.drop_dim)
            ax.scatter(
                centroid_proj[:, 0],
                centroid_proj[:, 1],
                centroid_proj[:, 2],
                s=90,
                marker="X",
                color=color_map[genre],
                edgecolor="black",
                linewidths=0.6,
            )

        ax.set_title(os.path.basename(d))
        dims = [0, 1, 2, 3]
        kept = [d for d in dims if d != args.drop_dim]
        ax.set_xlabel(f"Concept {kept[0]}")
        ax.set_ylabel(f"Concept {kept[1]}")
        ax.set_zlabel(f"Concept {kept[2]}")
        ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    if not frames:
        raise ValueError("no frames generated")
    imageio.mimsave(args.out_gif, frames, fps=1)


if __name__ == "__main__":
    main()
