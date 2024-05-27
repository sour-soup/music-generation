import numpy as np
import torch
from torch import nn, optim
from glob import glob
from tqdm import tqdm
from dataset import MidiNotesDataset
from models import MusicGenerationModel

batch_size = 64
seq_len = 50
max_file_num = 50
hidden_dim = 512
epochs = 2000
learning_rate = 0.001

data_dir = "./data/irish/*.mid"
files = glob(data_dir)[:max_file_num]

models_dir = "./models/train"
save_model_file = "model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Вычисляем на {device}...")

training_data = MidiNotesDataset(files, seq_len, transpositions=range(-3,3))
print(f"Загружено {len(training_data)} последовательностей...")
loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

n_notes_vocab = len(training_data.note_to_idx)
n_durations_vocab = len(training_data.duration_to_idx)
idx_to_note = training_data.idx_to_note
idx_to_duration = training_data.idx_to_duration
torch.save([idx_to_note, idx_to_duration], f"{models_dir}/vocabs.pth")

model = MusicGenerationModel(2, hidden_dim, n_notes_vocab, n_durations_vocab).to(device)
criterion_note = nn.CrossEntropyLoss()
criterion_duration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_model = None
best_loss = np.inf
for epoch in tqdm(range(epochs)):
    model.train()
    for batch in loader:
        inputs, note_targets, duration_targets = batch
        inputs.to(device)
        note_targets.to(device)
        duration_targets.to(device)

        optimizer.zero_grad()
        note_out, duration_out = model(inputs)

        loss_note = criterion_note(note_out, note_targets)
        loss_duration = criterion_duration(duration_out, duration_targets)
        loss = loss_note + loss_duration
        loss.backward()
        optimizer.step()
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs, note_targets, duration_targets = batch
            inputs.to(device)
            note_targets.to(device)
            duration_targets.to(device)

            optimizer.zero_grad()
            note_out, duration_out = model(inputs)

            loss_note = criterion_note(note_out, note_targets)
            loss_duration = criterion_duration(duration_out, duration_targets)
            loss += loss_note + loss_duration
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss / len(loader)}')
    if (epoch + 1) % 10 == 0:
        torch.save(best_model, f"{models_dir}/model{epoch + 1}.pth")
        print(f"Модель сохранена на эпохе {epoch + 1}")

torch.save(best_model, f"{models_dir}/best-model.pth")
print(f"Обучение завершено! Loss: {best_loss}")
