import torch
import numpy as np
from models import MusicGenerationModel
from utils import create_midi, midi_to_notes


def generate_sequence(model, initial_sequence, prediction_len, device):
    model.eval()
    seq_len = len(initial_sequence)
    generated = initial_sequence.tolist()
    with torch.no_grad():
        for _ in range(prediction_len):
            notes = np.reshape(generated[-seq_len:], (1, seq_len, 2)) / float(n_notes_vocab)
            notes = torch.tensor(notes, dtype=torch.float32)
            note_out, duration_out = model(notes)
            note_out = torch.argsort(note_out, dim=-1, descending=True).reshape(n_notes_vocab)[:1]
            duration_out = torch.argsort(duration_out, dim=-1, descending=True).reshape(n_durations_vocab)[:1]
            note = np.random.choice(note_out)
            duration = np.random.choice(duration_out)
            generated.append([idx_to_note[note], idx_to_duration[duration]])
    return np.array(generated)


data_dir = "./data"
models_dir = "./models/test"
model_path = f"{models_dir}/best-model.pth"
vocabs_path = f"{models_dir}/vocabs.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

vocabs = torch.load(vocabs_path, map_location=device)
idx_to_note = vocabs[0]
idx_to_duration = vocabs[1]

n_notes_vocab = len(idx_to_note)
n_durations_vocab = len(idx_to_duration)
note_to_idx = {idx_to_note[x]: x for x in idx_to_note}
duration_to_idx = {idx_to_duration[x]: x for x in idx_to_duration}

model = MusicGenerationModel(2, 512, n_notes_vocab, n_durations_vocab).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

seq_len = 50 - 1
gen_seq_len = 200
sample = midi_to_notes(f"{data_dir}/irish/1.mid")
start = np.random.randint(0, len(sample) - seq_len)
initial_sequence = sample[start : start + seq_len]
generated_sequence = generate_sequence(model, initial_sequence, gen_seq_len, device)

output_midi_path = f"{data_dir}/generated_sequence.mid"
create_midi(generated_sequence, output_midi_path)
print(f"Generated MIDI saved to {output_midi_path}")
