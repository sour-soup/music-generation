import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note, instrument_name_to_program


def midi_to_notes(midi_file: str, quantization=16) -> np.ndarray:
    pm = PrettyMIDI(midi_file)
    tempo = pm.get_tempo_changes()[1][0]
    beat_length = 60.0 / tempo
    sixteenth_length = beat_length / quantization

    quantized_notes = []
    instrument = pm.instruments[0]
    for note in instrument.notes:
        start_quantized = round(note.start / sixteenth_length)
        end_quantized = round(note.end / sixteenth_length)
        quantized_notes.append((start_quantized, end_quantized, note.pitch))
    quantized_notes.sort()

    notes = []
    for note in quantized_notes:
        notes.append([note[2], note[1] - note[0]])
    return np.array(notes)


def transpose_notes(notes: np.ndarray, semitones: int) -> np.ndarray:
    new_notes = [[note[0] + semitones, note[1]] for note in notes]
    return np.array(new_notes)


def create_midi(notes: np.ndarray, output: str, bpm=90, quantization=16):
    pm = PrettyMIDI()

    instrument = Instrument(instrument_name_to_program("Acoustic Grand Piano"))
    h = 60 / bpm / quantization
    start = 0
    for note in notes:
        new_note = Note(start=start, end=start + h * note[1], pitch=note[0], velocity=100)
        start += h * note[1]
        instrument.notes.append(new_note)
    pm.instruments.append(instrument)
    pm.write(output)
