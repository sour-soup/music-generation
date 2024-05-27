import torch
import numpy as np
from models import MusicGenerationModel
from utils import create_midi, midi_to_notes
from torch.distributions.categorical import Categorical


def generate_sequence(model, initial_sequence, prediction_len, device, temperature=0.7):
    model.eval()
    seq_len = len(initial_sequence)
    generated = initial_sequence.tolist()
    with torch.no_grad():
        for _ in range(prediction_len):
            notes = np.reshape(generated[-seq_len:], (1, seq_len, 2)) / float(n_notes_vocab)
            notes = torch.tensor(notes, dtype=torch.float32)
            note_logits, duration_logits = model(notes)
            note_logits *= temperature
            duration_logits *= temperature
            note = Categorical(logits=note_logits).sample().item()
            duration = Categorical(logits=duration_logits).sample().item()
            generated.append([idx_to_note[note], idx_to_duration[duration]])
    return np.array(generated[-prediction_len:])


data_dir = "./data"
models_dir = "./models/train"
model_path = f"{models_dir}/model300.pth"
vocabs_path = f"{models_dir}/vocabs.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

vocabs = torch.load(vocabs_path, map_location=device)
idx_to_note = vocabs[0]
idx_to_duration = vocabs[1]

n_notes_vocab = len(idx_to_note) - 1
n_durations_vocab = len(idx_to_duration) - 1
note_to_idx = {idx_to_note[x]: x for x in idx_to_note}
duration_to_idx = {idx_to_duration[x]: x for x in idx_to_duration}

model = MusicGenerationModel(2, 512, n_notes_vocab, n_durations_vocab).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

seq_len = 50 - 1
gen_seq_len = 200
sample = midi_to_notes(f"{data_dir}/irish/21.mid")
start = np.random.randint(0, len(sample) - seq_len)
initial_sequence = sample[start: start + seq_len]
generated_sequence = generate_sequence(model, initial_sequence, gen_seq_len, device)

output_midi_path = f"{data_dir}/generated_sequence.mid"
create_midi(generated_sequence, output_midi_path)
print(f"Generated MIDI saved to {output_midi_path}")
