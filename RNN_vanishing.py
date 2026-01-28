import torch
import torch.nn as nn
import torch.nn.init as init


def init_weights_xavier(module):
    """Initialize weights using Xavier/Glorot initialization"""
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.RNN) or isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal init for recurrent weights
            elif 'bias' in name:
                init.zeros_(param.data)


def init_weights_he(module):
    """Initialize weights using He initialization (good for ReLU)"""
    if isinstance(module, nn.Linear):
        init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.RNN) or isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)


class VanillaRNN(nn.Module):
    """
    Vanilla RNN model - prone to vanishing gradients with many layers
    """
    def __init__(self, input_size=1, hidden_size=256, num_layers=20, nonlinearity='tanh', dropout=0, num_classes=10, init_method='xavier'):
        super(VanillaRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Apply initialization
        if init_method == 'xavier':
            self.apply(init_weights_xavier)
        elif init_method == 'he':
            self.apply(init_weights_he)
        # Default PyTorch initialization otherwise
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions


class ResidualRNN(nn.Module):
    """
    RNN with residual connections - helps with vanishing gradients
    """
    def __init__(self, input_size=1, hidden_size=256, num_layers=20, nonlinearity='relu', dropout=0, num_classes=10, init_method='xavier'):
        super(ResidualRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Use single-layer RNNs with residual connections
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity=nonlinearity))
        
        for _ in range(num_layers - 1):
            self.rnn_layers.append(nn.RNN(hidden_size, hidden_size, batch_first=True, nonlinearity=nonlinearity))
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Apply initialization
        if init_method == 'xavier':
            self.apply(init_weights_xavier)
        elif init_method == 'he':
            self.apply(init_weights_he)
    
    def forward(self, x):
        # Apply residual connections between layers
        output = x
        for i, rnn_layer in enumerate(self.rnn_layers):
            rnn_out, _ = rnn_layer(output)
            if i > 0 and output.shape[-1] == rnn_out.shape[-1]:
                # Residual connection (only if dimensions match)
                output = output + rnn_out
            else:
                output = rnn_out
        
        last_hidden = output[:, -1, :]
        predictions = self.fc(last_hidden)
        return predictions
