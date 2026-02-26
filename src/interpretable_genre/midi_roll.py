from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import mido
import numpy as np


@dataclass
class MeasureRoll:
    index: int
    roll: np.ndarray  # shape: (steps, 128)


@dataclass
class MeasureEvents:
    index: int
    tokens: np.ndarray  # shape: (steps, max_polyphony, 2) => (program_id, pitch_dur_id)


DURATION_TEMPLATES = np.array(
    [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
    dtype=np.int32,
)
PROGRAM_CHANNELS = 16
PROGRAM_BASE = 128
PROGRAM_EOS = PROGRAM_CHANNELS * PROGRAM_BASE
PROGRAM_PAD = PROGRAM_EOS + 1
PROGRAM_VOCAB = PROGRAM_PAD + 1
PITCHDUR_EOS = len(DURATION_TEMPLATES) * 128
PITCHDUR_PAD = PITCHDUR_EOS + 1
PITCHDUR_VOCAB = PITCHDUR_PAD + 1


def _get_time_signature(midi: mido.MidiFile) -> Tuple[int, int]:
    for track in midi.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def _collect_notes(midi: mido.MidiFile) -> List[Tuple[int, int, int]]:
    notes: List[Tuple[int, int, int]] = []
    active: dict[int, int] = {}
    abs_tick = 0
    for msg in mido.merge_tracks(midi.tracks):
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = abs_tick
        elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
            start = active.pop(msg.note, None)
            if start is not None and abs_tick > start:
                notes.append((start, abs_tick, msg.note))
    return notes


def _collect_notes_by_track(midi: mido.MidiFile) -> List[List[Tuple[int, int, int]]]:
    per_track: List[List[Tuple[int, int, int]]] = [[] for _ in midi.tracks]
    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        active: Dict[int, int] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_tick
            elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                start = active.pop(msg.note, None)
                if start is not None and abs_tick > start:
                    per_track[track_idx].append((start, abs_tick, msg.note))
    return per_track


def _collect_notes_with_programs(
    midi: mido.MidiFile, max_tracks: int | None = None
) -> List[Tuple[int, int, int, int]]:
    notes: List[Tuple[int, int, int, int]] = []

    def channel_for(msg: mido.Message) -> int:
        return int(getattr(msg, "channel", 0) or 0)

    for track_idx, track in enumerate(midi.tracks):
        if max_tracks is not None and track_idx >= max_tracks:
            break
        abs_tick = 0
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        program_by_channel: Dict[int, int] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "program_change":
                program_by_channel[channel_for(msg)] = int(msg.program)
            if msg.type == "note_on" and msg.velocity > 0:
                channel = channel_for(msg)
                program = 127 if channel == 9 else program_by_channel.get(channel, 0)
                program_token = channel * PROGRAM_BASE + program
                active[(channel, msg.note)] = (abs_tick, program_token)
            elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                channel = channel_for(msg)
                start_info = active.pop((channel, msg.note), None)
                if start_info is not None:
                    start, note_program = start_info
                    if abs_tick > start:
                        notes.append((start, abs_tick, msg.note, note_program))
    return notes


def _extract_programs(midi: mido.MidiFile) -> List[int]:
    programs: List[int] = []
    for track in midi.tracks:
        program = 0
        for msg in track:
            if msg.type == "program_change":
                program = int(msg.program)
                break
        programs.append(program)
    return programs


def _collect_program_changes_by_track(midi: mido.MidiFile) -> List[List[Tuple[int, int]]]:
    per_track: List[List[Tuple[int, int]]] = [[] for _ in midi.tracks]
    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "program_change":
                per_track[track_idx].append((abs_tick, int(msg.program)))
        if not per_track[track_idx]:
            per_track[track_idx].append((0, 0))
    return per_track


def midi_to_measure_rolls(
    midi_path: str,
    steps_per_beat: int = 4,
    target_steps_per_measure: int | None = None,
    max_polyphony: int | None = None,
) -> Tuple[List[MeasureRoll], Tuple[int, int], int, List[int]]:
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    numerator, denominator = _get_time_signature(midi)
    programs = _extract_programs(midi)
    notes = _collect_notes(midi)
    if not notes:
        return [], (numerator, denominator), ticks_per_beat, programs

    ticks_per_step = max(1, int(round(ticks_per_beat / steps_per_beat)))
    beats_per_measure = numerator * (4 / denominator)
    steps_per_measure = max(1, int(round(steps_per_beat * beats_per_measure)))

    max_tick = max(end for _, end, _ in notes)
    total_steps = int(np.ceil(max_tick / ticks_per_step))
    total_measures = max(1, int(np.ceil(total_steps / steps_per_measure)))

    measures = [np.zeros((steps_per_measure, 128), dtype=np.float32) for _ in range(total_measures)]
    for start_tick, end_tick, pitch in notes:
        start_step = int(round(start_tick / ticks_per_step))
        end_step = max(start_step + 1, int(round(end_tick / ticks_per_step)))
        for step in range(start_step, end_step):
            measure_idx = step // steps_per_measure
            step_idx = step % steps_per_measure
            if measure_idx < total_measures:
                if max_polyphony is None:
                    measures[measure_idx][step_idx, pitch] = 1.0
                else:
                    current = np.where(measures[measure_idx][step_idx] > 0.0)[0]
                    if current.size < max_polyphony:
                        measures[measure_idx][step_idx, pitch] = 1.0

    rolls = []
    for i in range(total_measures):
        roll = measures[i]
        if target_steps_per_measure is not None and target_steps_per_measure != steps_per_measure:
            if roll.shape[0] > target_steps_per_measure:
                roll = roll[:target_steps_per_measure]
            else:
                pad = target_steps_per_measure - roll.shape[0]
                roll = np.pad(roll, ((0, pad), (0, 0)), mode="constant")
        rolls.append(MeasureRoll(index=i, roll=roll))
    return rolls, (numerator, denominator), ticks_per_beat, programs


def _quantize_duration_steps(duration_steps: int) -> int:
    boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    return int(np.searchsorted(boundaries, duration_steps))


def midi_to_measure_events(
    midi_path: str,
    steps_per_beat: int = 4,
    max_polyphony: int = 8,
    max_tracks: int | None = None,
    target_steps_per_measure: int | None = None,
    min_aligned_ratio: float | None = None,
    align_tolerance_steps: float = 0.25,
) -> Tuple[List[MeasureEvents], Tuple[int, int], int]:
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    numerator, denominator = _get_time_signature(midi)
    notes = _collect_notes_with_programs(midi, max_tracks=max_tracks)
    if not notes:
        return [], (numerator, denominator), ticks_per_beat

    ticks_per_step = max(1, int(round(ticks_per_beat / steps_per_beat)))
    beats_per_measure = numerator * (4 / denominator)
    steps_per_measure = max(1, int(round(steps_per_beat * beats_per_measure)))
    duration_scale = max(1, int(round(steps_per_beat / 4)))

    if min_aligned_ratio is not None:
        aligned = 0
        for start_tick, _, _, _ in notes:
            step_float = start_tick / ticks_per_step
            if abs(step_float - round(step_float)) <= align_tolerance_steps:
                aligned += 1
        aligned_ratio = aligned / max(1, len(notes))
        if aligned_ratio < min_aligned_ratio:
            return [], (numerator, denominator), ticks_per_beat

    max_tick = max(end for _, end, _, _ in notes)
    total_steps = int(np.ceil(max_tick / ticks_per_step))
    total_measures = max(1, int(np.ceil(total_steps / steps_per_measure)))

    measures = [
        np.full((steps_per_measure, max_polyphony, 2), fill_value=[PROGRAM_PAD, PITCHDUR_PAD], dtype=np.int32)
        for _ in range(total_measures)
    ]
    poly_counts = [np.zeros(steps_per_measure, dtype=np.int32) for _ in range(total_measures)]

    for start_tick, end_tick, pitch, program in notes:
        # print(program)
        start_step = int(round(start_tick / ticks_per_step))
        duration_steps = max(1, int(round((end_tick - start_tick) / ticks_per_step)))
        duration_units = max(1, int(round(duration_steps / duration_scale)))
        dur_idx = _quantize_duration_steps(duration_units)
        pitchdur = dur_idx * 128 + int(pitch)
        if pitchdur >= PITCHDUR_EOS:
            pitchdur = PITCHDUR_EOS - 1
        measure_idx = start_step // steps_per_measure
        step_idx = start_step % steps_per_measure
        if measure_idx >= total_measures:
            continue
        slot = poly_counts[measure_idx][step_idx]
        if slot >= max_polyphony:
            continue
        measures[measure_idx][step_idx, slot, 0] = int(program)
        measures[measure_idx][step_idx, slot, 1] = int(pitchdur)
        poly_counts[measure_idx][step_idx] += 1

    for m in range(total_measures):
        for s in range(steps_per_measure):
            count = poly_counts[m][s]
            if count < max_polyphony:
                measures[m][s, count, 0] = PROGRAM_EOS
                measures[m][s, count, 1] = PITCHDUR_EOS

    tokens_list = []
    for i in range(total_measures):
        tokens = measures[i]
        if target_steps_per_measure is not None and target_steps_per_measure != steps_per_measure:
            if tokens.shape[0] > target_steps_per_measure:
                tokens = tokens[:target_steps_per_measure]
            else:
                pad = target_steps_per_measure - tokens.shape[0]
                pad_block = np.full((pad, max_polyphony, 2), fill_value=[PROGRAM_PAD, PITCHDUR_PAD], dtype=np.int32)
                tokens = np.concatenate([tokens, pad_block], axis=0)
        tokens_list.append(MeasureEvents(index=i, tokens=tokens))
    return tokens_list, (numerator, denominator), ticks_per_beat


def events_to_midi(
    measures: List[MeasureEvents],
    time_signature: Tuple[int, int],
    ticks_per_beat: int,
    steps_per_beat: int = 4,
) -> mido.MidiFile:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    numerator, denominator = time_signature
    beats_per_measure = numerator * (4 / denominator)
    steps_per_measure = measures[0].tokens.shape[0] if measures else 1
    ticks_per_step = max(1, int(round(ticks_per_beat / steps_per_beat)))
    duration_scale = max(1, int(round(steps_per_beat / 4)))

    tracks: Dict[int, mido.MidiTrack] = {}
    events_by_program: Dict[int, List[Tuple[int, mido.Message]]] = {}
    for measure in measures:
        for step_idx in range(measure.tokens.shape[0]):
            abs_step = measure.index * steps_per_measure + step_idx
            abs_tick = abs_step * ticks_per_step
            for slot in range(measure.tokens.shape[1]):
                program = int(measure.tokens[step_idx, slot, 0])
                pitchdur = int(measure.tokens[step_idx, slot, 1])
                if program == PROGRAM_PAD:
                    continue
                if program == PROGRAM_EOS:
                    break
                if pitchdur >= PITCHDUR_EOS:
                    continue
                pitch = pitchdur % 128
                dur_idx = pitchdur // 128
                if dur_idx < 0 or dur_idx >= len(DURATION_TEMPLATES):
                    continue
                duration_steps = int(DURATION_TEMPLATES[dur_idx]) * duration_scale
                end_tick = abs_tick + duration_steps * ticks_per_step
                channel = int(program // PROGRAM_BASE)
                events_by_program.setdefault(program, [])
                events_by_program[program].append(
                    (abs_tick, mido.Message("note_on", note=int(pitch), velocity=64, channel=channel, time=0))
                )
                events_by_program[program].append(
                    (end_tick, mido.Message("note_off", note=int(pitch), velocity=64, channel=channel, time=0))
                )

    for program, events in events_by_program.items():
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(
            mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0)
        )
        channel = int(program // PROGRAM_BASE)
        raw_program = int(program % PROGRAM_BASE)
        track.append(mido.Message("program_change", program=raw_program, channel=channel, time=0))
        events.sort(key=lambda x: x[0])
        last_tick = 0
        for tick, msg in events:
            msg.time = tick - last_tick
            track.append(msg)
            last_tick = tick
    return midi


def midi_to_measure_rolls_by_track(
    midi_path: str,
    steps_per_beat: int = 4,
    target_steps_per_measure: int | None = None,
    max_tracks: int | None = None,
    max_polyphony: int | None = None,
) -> Tuple[List[List[MeasureRoll]], Tuple[int, int], int, List[int], List[List[Tuple[int, int]]]]:
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    numerator, denominator = _get_time_signature(midi)
    programs = _extract_programs(midi)
    program_changes = _collect_program_changes_by_track(midi)
    notes_by_track = _collect_notes_by_track(midi)
    if max_tracks is not None:
        notes_by_track = notes_by_track[:max_tracks]
        programs = programs[:max_tracks]
        program_changes = program_changes[:max_tracks]

    ticks_per_step = max(1, int(round(ticks_per_beat / steps_per_beat)))
    beats_per_measure = numerator * (4 / denominator)
    steps_per_measure = max(1, int(round(steps_per_beat * beats_per_measure)))

    max_tick = 0
    for notes in notes_by_track:
        if notes:
            max_tick = max(max_tick, max(end for _, end, _ in notes))
    if max_tick == 0:
        return [], (numerator, denominator), ticks_per_beat, programs, program_changes

    total_steps = int(np.ceil(max_tick / ticks_per_step))
    total_measures = max(1, int(np.ceil(total_steps / steps_per_measure)))

    tracks: List[List[MeasureRoll]] = []
    for notes in notes_by_track:
        measures = [np.zeros((steps_per_measure, 128), dtype=np.float32) for _ in range(total_measures)]
        for start_tick, end_tick, pitch in notes:
            start_step = int(round(start_tick / ticks_per_step))
            end_step = max(start_step + 1, int(round(end_tick / ticks_per_step)))
            for step in range(start_step, end_step):
                measure_idx = step // steps_per_measure
                step_idx = step % steps_per_measure
                if measure_idx < total_measures:
                    if max_polyphony is None:
                        measures[measure_idx][step_idx, pitch] = 1.0
                    else:
                        current = np.where(measures[measure_idx][step_idx] > 0.0)[0]
                        if current.size < max_polyphony:
                            measures[measure_idx][step_idx, pitch] = 1.0
        roll_list = []
        for i in range(total_measures):
            roll = measures[i]
            if target_steps_per_measure is not None and target_steps_per_measure != steps_per_measure:
                if roll.shape[0] > target_steps_per_measure:
                    roll = roll[:target_steps_per_measure]
                else:
                    pad = target_steps_per_measure - roll.shape[0]
                    roll = np.pad(roll, ((0, pad), (0, 0)), mode="constant")
            roll_list.append(MeasureRoll(index=i, roll=roll))
        tracks.append(roll_list)

    # program changes are returned separately so they can be reinserted on export
    return tracks, (numerator, denominator), ticks_per_beat, programs, program_changes


def rolls_to_midi(
    rolls: List[MeasureRoll],
    time_signature: Tuple[int, int],
    ticks_per_beat: int,
    steps_per_beat: int = 4,
    programs: List[int] | None = None,
) -> mido.MidiFile:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    numerator, denominator = time_signature
    track_count = max(1, len(programs) if programs is not None else 1)
    tracks: List[mido.MidiTrack] = []
    for idx in range(track_count):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0))
        if programs is not None:
            channel = idx % 16
            track.append(mido.Message("program_change", program=int(programs[idx]), channel=channel, time=0))
        tracks.append(track)

    beats_per_measure = numerator * (4 / denominator)
    ticks_per_measure = int(round(ticks_per_beat * beats_per_measure))
    token_steps_per_measure = rolls[0].roll.shape[0] if rolls else max(1, int(round(steps_per_beat * beats_per_measure)))
    ticks_per_step = max(1, int(round(ticks_per_measure / max(1, token_steps_per_measure))))

    events: List[Tuple[int, mido.Message]] = []
    for measure in rolls:
        for step_idx in range(measure.roll.shape[0]):
            abs_step = measure.index * token_steps_per_measure + step_idx
            abs_tick = abs_step * ticks_per_step
            active_pitches = np.where(measure.roll[step_idx] > 0.5)[0]
            for pitch in active_pitches:
                events.append((abs_tick, mido.Message("note_on", note=int(pitch), velocity=64, time=0)))
                events.append((abs_tick + ticks_per_step, mido.Message("note_off", note=int(pitch), velocity=64, time=0)))

    events.sort(key=lambda x: x[0])
    last_tick = 0
    for tick, msg in events:
        msg.time = tick - last_tick
        tracks[0].append(msg)
        last_tick = tick

    return midi


def rolls_to_midi_by_track(
    tracks: List[List[MeasureRoll]],
    time_signature: Tuple[int, int],
    ticks_per_beat: int,
    steps_per_beat: int = 4,
    programs: List[int] | None = None,
    max_tracks: int | None = None,
    max_polyphony: int | None = None,
    program_changes: List[List[Tuple[int, int]]] | None = None,
) -> mido.MidiFile:
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    numerator, denominator = time_signature
    track_count = max(1, len(tracks))
    if max_tracks is not None:
        track_count = min(track_count, max_tracks)
        tracks = tracks[:track_count]
    while len(tracks) < track_count:
        tracks.append([])
    if programs is None:
        programs = [0 for _ in range(track_count)]
    else:
        programs = programs[:track_count]
        if len(programs) < track_count:
            programs.extend([0 for _ in range(track_count - len(programs))])

    beats_per_measure = numerator * (4 / denominator)
    ticks_per_measure = int(round(ticks_per_beat * beats_per_measure))
    token_steps_per_measure = tracks[0][0].roll.shape[0] if tracks and tracks[0] else max(1, int(round(steps_per_beat * beats_per_measure)))
    ticks_per_step = max(1, int(round(ticks_per_measure / max(1, token_steps_per_measure))))

    for track_idx in range(track_count):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0))
        channel = track_idx % 16
        events: List[Tuple[int, mido.Message]] = []
        if program_changes is None or track_idx >= len(program_changes):
            events.append((0, mido.Message("program_change", program=int(programs[track_idx]), channel=channel, time=0)))
        else:
            for tick, program in program_changes[track_idx]:
                events.append((int(tick), mido.Message("program_change", program=int(program), channel=channel, time=0)))
        for measure in tracks[track_idx]:
            step_count = measure.roll.shape[0]
            for step_idx in range(step_count):
                abs_step = measure.index * token_steps_per_measure + step_idx
                abs_tick = abs_step * ticks_per_step
                active_pitches = np.where(measure.roll[step_idx] > 0.5)[0]
                if max_polyphony is not None and active_pitches.size > max_polyphony:
                    active_pitches = active_pitches[:max_polyphony]
                for pitch in active_pitches:
                    events.append((abs_tick, mido.Message("note_on", note=int(pitch), velocity=64, channel=channel, time=0)))
                    events.append(
                        (abs_tick + ticks_per_step, mido.Message("note_off", note=int(pitch), velocity=64, channel=channel, time=0))
                    )
        events.sort(key=lambda x: x[0])
        last_tick = 0
        for tick, msg in events:
            msg.time = tick - last_tick
            track.append(msg)
            last_tick = tick

    return midi
