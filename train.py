import torch
from torch import nn, optim
from glob import glob
from tqdm import tqdm
from dataset import MidiNotesDataset
from models import MusicGenerationModel

batch_size = 64
seq_len = 25
max_file_num = 20
epochs = 200
learning_rate = 0.005

data_dir = "./data/irish/*.mid"
files = glob(data_dir)[:max_file_num]

models_dir = "./models/"
save_model_file = "model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Вычисляем на {device}...")

training_data = MidiNotesDataset(files, seq_len)
print(f"Загружено {len(training_data)} последовательностей...")
loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

model = MusicGenerationModel().to(device)
criterion_note = nn.CrossEntropyLoss()
criterion_duration = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for batch in loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        note_targets = targets[:, 0].long()
        duration_targets = targets[:, 1].long()

        if torch.any(note_targets >= 128) or torch.any(duration_targets >= 128):
            print("Ошибка: целевые значения выходят за пределы допустимого диапазона.")
            continue

        optimizer.zero_grad()

        note_out, duration_out = model(inputs)

        loss_note = criterion_note(note_out, note_targets)
        loss_duration = criterion_duration(duration_out, duration_targets)
        loss = loss_note + loss_duration
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader)}')
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{models_dir}model{epoch+1}.pth")
        print(f"Модель сохранена на эпохе {epoch + 1}")

print("Обучение завершено!")