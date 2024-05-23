import torch
import numpy as np
from models import MusicGenerationModel
from utils import create_midi, midi_to_notes


def generate_sequence(model, initial_sequence, seq_len, device):
    generated = initial_sequence.tolist()
    current_sequence = torch.tensor(initial_sequence, dtype=torch.float).unsqueeze(0).to(device)
    for _ in range(seq_len):
        model.eval()
        with torch.no_grad():
            note_out, duration_out = model(current_sequence)
            note = torch.argmax(note_out, dim=1).item()
            duration = torch.argmax(duration_out, dim=1).item()
            generated.append([note, duration])
            current_sequence = torch.tensor(generated[-len(initial_sequence):], dtype=torch.float).unsqueeze(0).to(device)
    return np.array(generated)


models_dir = "./models/"
model_path = f"{models_dir}model200.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MusicGenerationModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


gen_seq_len = 150
initial_sequence = midi_to_notes("./data/irish/1.mid")
generated_sequence = generate_sequence(model, initial_sequence, gen_seq_len, device)

output_midi_path = "generated_sequence.mid"
create_midi(generated_sequence, output_midi_path)
print(f"Generated MIDI saved to {output_midi_path}")