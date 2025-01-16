import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=3, dropout=0.3):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.hidden_size = hidden_size
        
        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Non-linear activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)  # Get output from LSTM
        lstm_out = hn[-1]  # Take the last hidden state of the last layer
        
        # Fully connected layers
        out = self.fc1(lstm_out)  # First FC layer
        out = self.relu(out)      # Non-linear activation
        out = self.fc2(out)       # Second FC layer
        return out
