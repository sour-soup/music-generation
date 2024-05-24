from torch import nn
import torch.nn.functional as F


class MusicGenerationModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, output_dim_note=128, output_dim_duration=128, dropout=0.1):
        super(MusicGenerationModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_note = nn.Linear(hidden_dim, output_dim_note)
        self.fc_duration = nn.Linear(hidden_dim, output_dim_duration)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        dropout_out = self.dropout(lstm_out)
        note_out = self.fc_note(dropout_out)
        duration_out = self.fc_duration(dropout_out)
        note_out = F.log_softmax(note_out, dim=1)
        duration_out = F.log_softmax(duration_out, dim=1)
        return note_out, duration_out
