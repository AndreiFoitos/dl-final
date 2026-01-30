import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

from data.data_pipeline import create_dataloaders
from cnn_model_exploding import CNN


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1, mitigation_on=False):
    """
    Training loop for CNN model.
    
    :param model: CNN model.
    :param train_loader: Training data.
    :param num_epochs: Number of epochs model is trained.
    :param learning_rate: Learning rate used for training.
    :param mitigation_on: if False training loop will results in exploding gradients
    else will result in mitigtation, will clip gradients, add weight_decay and learning rate decay with scheduler.
    This will result in gradient and loss going down over time.
    """

    if mitigation_on:
        gradient_clip = 0.5
        weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()

    if not mitigation_on:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
    if mitigation_on:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
        
    train_losses = []
    grad_norms = []
    validation_losses = []

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

            if mitigation_on:
                mitigation = 'mitigation'
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            else:
                mitigation = 'no_mitigation'
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            if torch.isnan(loss):
                print(f"NaN at epoch {epoch}, batch {batch_idx}!")
                break

            epoch_grad_norms.append(grad_norm.item())
            optimizer.step()
            
            train_loss += loss.item()

        last_10_grads = epoch_grad_norms[-10:]
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
                
        grad_norms.append(np.mean(epoch_grad_norms))

        if mitigation_on:
            scheduler.step()

            last_lr = scheduler.get_last_lr()
            print(last_lr)

        if torch.isnan(loss):
            break

        # #for if you want to use val
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for sequences, targets in val_loader:
        #         sequences = sequences.to(device)
        #         targets = targets.to(device)
        #         predictions = model(sequences)
        #         loss = criterion(predictions, targets)
        #         val_loss += loss.item()
        
        # val_loss /= len(val_loader)
        # validation_losses.append(val_loss)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, GradNorm={np.mean(epoch_grad_norms):.2e}")

    return train_losses, grad_norms, validation_losses, last_10_grads, mitigation


if __name__ == '__main__':
    #For exploding set mitigation_on to False and for mitigation set mitigation on to True

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

    model = CNN(num_conv_layers=5, base_channels=32, num_classes=10).to(device)

    train_losses, grad_norms, validation_losses, last_10_grads, mitigation = train_model(model, train_loader, val_loader, 
                                                                                         learning_rate=0.35, mitigation_on=True)

    epochs = np.arange(0, len(train_losses))

    # Plot training loss
    plt.figure()
    plt.plot(epochs, train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title('Training loss over epochs')
    plt.savefig(f'plots/training_loss_{mitigation}.png', dpi=150)

    # Plot gradient norms
    plt.figure()
    plt.plot(epochs, grad_norms, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Gradient norm over epochs')
    plt.savefig(f'plots/grad_norms_{mitigation}.png', dpi=150)

    batches = np.arange(1, len(last_10_grads) + 1)

    #Plot last 10 grad_norms
    plt.figure()
    plt.plot(batches, last_10_grads, marker='o')
    plt.xlabel('Batches')
    plt.ylabel('Gradient norm')
    plt.title('Gradient norm over last 10 bathes')
    plt.savefig(f'plots/grad_norms_last_10_{mitigation}.png', dpi=150)