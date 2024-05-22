import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import midi_to_notes, transpose_notes


class MidiNotesDataset(Dataset):
    def __init__(self, midi_files, sequence_length, max_file_num=None, transpositions=range(-6, 6), quantization=16):
        self.midi_files = midi_files
        self.sequence_length = sequence_length
        self.transpositions = transpositions
        self.quantization = quantization
        self.max_file_num = len(midi_files) if max_file_num is None else min(len(midi_files), max_file_num)
        self.data = []
        self._load_data()

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
        return torch.tensor(self.data[idx], dtype=torch.float)
