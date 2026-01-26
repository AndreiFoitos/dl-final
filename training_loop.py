import numpy as np
import torch
import torch.nn as nn

from data.data_pipeline import create_dataloaders
from RNN_model import RNN


def train_model(model, train_loader, num_epochs=100, learning_rate=1, gradient_clip=None, weight_scaler=None):
    """
    Training loop for RNN model.
    
    :param model: RNN model.
    :param train_loader: Training data.
    :param num_epochs: Number of epochs model is trained.
    :param learning_rate: Learning rate used for training.
    :param gradient_clip: Value when gradients are clipped. If None gradients don't get clipped.
    :param weight_scaler: Scaling value for the weights, if None weigths are not getting scaled.
    """

    if weight_scaler != None:
        for name, param in model.named_parameters():
            if 'weight_hh' in name:
                param.data *= weight_scaler

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_losses = []
    grad_norms = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_grad_norms = []
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()

            if gradient_clip != None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            epoch_grad_norms.append(grad_norm.item())
            optimizer.step()
            
            train_loss += loss.item()

            if torch.isnan(loss):
                print(f"NaN at epoch {epoch}, batch {batch_idx}!")
                break
  
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        grad_norms.append(np.mean(epoch_grad_norms))
        print(f"Epoch {epoch}: Loss={loss:.4f}, GradNorm={np.mean(epoch_grad_norms):.2e}")

        if torch.isnan(loss):
            break

    return train_losses, grad_norms

if __name__ == '__main__':
    #For exploding gradiensts set batch_size to 64 and don't apply clipping
    #For mitigation method for exploding gradiensts set gradient_clipping to 0.5 and set batch_size to 32

    DATA_CONFIG = {
    "use_normalized": False,     
    "add_noise": False,      
    "noise_std": 0.05,
    "use_pca": False,
    "pca_components": 512,
    "val_split": 0.2,
    "batch_size":64,
    "shuffle": True,
    }

    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessed_dir="preprocessed",
        config=DATA_CONFIG
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    model = RNN(input_size=1, hidden_size=256, num_layers=3, nonlinearity='relu', num_classes=10).to(device)

    train_losses, grad_norms = train_model(model, train_loader, learning_rate=1)

