from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .explain import build_explanation
from .utils import load_label_map


def _render_card(card: dict) -> str:
    lines = [
        "Genre Explanation Card",
        "=======================",
        f"MIDI: {card['midi_path']}",
        f"Predicted genre: {card['predicted_genre']}",
        f"Confidence: {card['confidence']:.3f}",
        "",
        f"Summary: {card['summary']}",
        "",
        "Top contributing concepts:",
    ]
    for item in card["top_features"]:
        lines.append(f"- {item['feature']}: {item['contribution']:.4f}")
    lines.append("")
    lines.append("Top measures:")
    for item in card["top_measures"]:
        lines.append(
            f"- measure {item['measure_index']} (ticks {item['start_tick']}:{item['end_tick']}): {item['score']:.4f}"
        )
    lines.append("")
    lines.append("Occlusion impact:")
    for item in card["occlusion"]:
        lines.append(f"- measure {item['measure_index']}: {item['delta']:.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict and generate explanation card.")
    parser.add_argument("--model", required=True, help="Path to joblib model")
    parser.add_argument("--label_map", required=True, help="Path to label map json")
    parser.add_argument("--midi_path", required=True, help="MIDI file to analyze")
    parser.add_argument("--out_dir", required=True, help="Output directory for card")
    args = parser.parse_args()

    payload = joblib.load(args.model)
    model = payload["model"]
    projector = payload["projector"]
    concept_names = payload["concept_names"]

    label_map = load_label_map(args.label_map)
    card = build_explanation(model, projector, concept_names, label_map, args.midi_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "genre_explanation.json"
    txt_path = out_dir / "genre_explanation.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    txt_path.write_text(_render_card(card), encoding="utf-8")


if __name__ == "__main__":
    main()
