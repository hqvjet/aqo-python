import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRU, self).__init__()
        self.name = 'GRU'
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(1)

        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
