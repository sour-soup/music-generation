import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import midi_to_notes, transpose_notes


class MidiNotesDataset(Dataset):
    def __init__(self, midi_files, sequence_length, transpositions=range(-6, 6), quantization=16):
        self.midi_files = midi_files
        self.sequence_length = sequence_length
        self.transpositions = transpositions
        self.quantization = quantization
        self.data = []
        self._load_data()

        notes = sorted(list(set(x[0] for sequence in self.data for x in sequence)))
        durations = sorted(list(set(x[1] for sequence in self.data for x in sequence)))
        self.note_to_idx = {note: i for i, note in enumerate(notes)}
        self.idx_to_note = {i: note for i, note in enumerate(notes)}
        self.duration_to_idx = {duration: i for i, duration in enumerate(durations)}
        self.idx_to_duration = {i: duration for i, duration in enumerate(durations)}

        self.notes_data = [[self.note_to_idx[note[0]] for note in sequence] for sequence in self.data]
        self.durations_data = [[self.duration_to_idx[note[1]] for note in sequence] for sequence in self.data]

    def _load_data(self):
        print("Чтение файлов...")
        for midi_file in tqdm(self.midi_files):
            notes = midi_to_notes(midi_file, self.quantization)
            for t in self.transpositions:
                transposed_notes = transpose_notes(notes, t)
                self._create_sequences(transposed_notes)

    def _create_sequences(self, notes):
        for i in range(len(notes) - self.sequence_length + 1):
            sequence = notes[i:i + self.sequence_length]
            self.data.append(sequence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        notes = torch.tensor(self.notes_data[idx], dtype=torch.float)
        durations = torch.tensor(self.durations_data[idx], dtype=torch.float)
        note_target = notes[-1].long()
        duration_target = durations[-1].long()
        notes = (notes[:-1] / len(self.note_to_idx)).reshape(-1, 1)
        durations = (durations[:-1] / len(self.duration_to_idx)).reshape(-1, 1)
        inputs = torch.cat((notes, durations), dim=-1)
        return inputs, note_target, duration_target
