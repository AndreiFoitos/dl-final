import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    RNN model with one fully connected layer
    """
    def __init__(self, input_size=1, hidden_size=256, num_layers=20, nonlinearity='relu', dropout=0, num_classes=10):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity = nonlinearity,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(nn.Linear(hidden_size, num_classes))

    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions