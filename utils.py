from __future__ import annotations

import csv
from pathlib import Path


def generate_xmidi_genre_metadata(
    dataset_dir: str | Path | None = None,
    output_name: str = "metadata_genre.csv",
) -> Path:
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parent / "XMIDI_Dataset"
    else:
        dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_path = dataset_dir / output_name
    midi_files = sorted(dataset_dir.glob("XMIDI_*.midi"))
    rows: list[tuple[str, str]] = []

    for midi_path in midi_files:
        stem_parts = midi_path.stem.split("_")
        if len(stem_parts) != 4 or stem_parts[0] != "XMIDI":
            raise ValueError(f"Unexpected filename format: {midi_path.name}")
        genre = stem_parts[2]
        rows.append((midi_path.name, genre))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "genre"])
        writer.writerows(rows)

    return output_path


if __name__ == "__main__":
    output = generate_xmidi_genre_metadata()
    print(f"Wrote {output}")
