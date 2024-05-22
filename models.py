from torch import nn


class MusicGenerationModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim_note=128, output_dim_duration=128):
        super(MusicGenerationModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_note = nn.Linear(hidden_dim, output_dim_note)
        self.fc_duration = nn.Linear(hidden_dim, output_dim_duration)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        attn_output, _ = self.attention(lstm_out1, lstm_out1, lstm_out1)
        lstm_out2, _ = self.lstm2(attn_output)
        last_hidden_state = lstm_out2[:, -1, :]
        note_out = self.fc_note(last_hidden_state)
        duration_out = self.fc_duration(last_hidden_state)
        return note_out, duration_out
