from __future__ import annotations

import argparse
from collections import defaultdict

import mido


def _channel_of(msg: mido.Message) -> int:
    return int(getattr(msg, "channel", 0) or 0)


def _print_track_programs(midi: mido.MidiFile) -> None:
    print("Per-track program changes:")
    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        changes = []
        for msg in track:
            abs_tick += msg.time
            if msg.type == "program_change":
                changes.append((abs_tick, _channel_of(msg), int(msg.program)))
        if not changes:
            print(f"  track {track_idx}: (no program_change)")
            continue
        printable = ", ".join(f"t{t}@ch{ch}=p{p}" for t, ch, p in changes)
        print(f"  track {track_idx}: {printable}")


def _print_channel_note_counts(midi: mido.MidiFile) -> None:
    print("\nPer-channel program note counts (from merged tracks):")
    abs_tick = 0
    program_by_channel: dict[int, int] = defaultdict(int)
    note_counts: dict[tuple[int, int], int] = defaultdict(int)
    for msg in mido.merge_tracks(midi.tracks):
        abs_tick += msg.time
        if msg.type == "program_change":
            program_by_channel[_channel_of(msg)] = int(msg.program)
        if msg.type == "note_on" and msg.velocity > 0:
            channel = _channel_of(msg)
            program = 127 if channel == 9 else program_by_channel.get(channel, 0)
            note_counts[(channel, program)] += 1

    if not note_counts:
        print("  (no note_on events)")
        return

    for (channel, program), count in sorted(note_counts.items()):
        print(f"  channel {channel} program {program}: {count} notes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect program changes and note programs in a MIDI file.")
    parser.add_argument("--midi_path", required=True, help="Path to MIDI file")
    args = parser.parse_args()

    midi = mido.MidiFile(args.midi_path)
    _print_track_programs(midi)
    _print_channel_note_counts(midi)


if __name__ == "__main__":
    main()
