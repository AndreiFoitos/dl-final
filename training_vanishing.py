import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from data.data_pipeline import create_dataloaders
from RNN_vanishing import VanillaRNN, ResidualRNN


def train_model(model, train_loader, device, num_epochs=100, learning_rate=0.01, gradient_clip=None, 
                gradient_clip_min=None, weight_decay=0, model_type='vanilla'):
    """
    Training loop for exploring vanishing gradients.
    
    :param model: RNN model (VanillaRNN or ResidualRNN)
    :param train_loader: Training data
    :param device: Device to run training on (cuda/cpu)
    :param num_epochs: Number of epochs
    :param learning_rate: Learning rate
    :param gradient_clip: Maximum gradient norm (for clipping)
    :param gradient_clip_min: Minimum gradient norm threshold (to detect vanishing)
    :param weight_decay: Weight decay for regularization
    :param model_type: Type of model ('vanilla', 'residual')
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    grad_norms = []
    grad_norms_per_layer = []  # Track gradients per layer
    validation_losses = []
    
    # Track minimum gradient norms to detect vanishing
    min_grad_norms = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_grad_norms = []
        epoch_min_grad_norms = []
        epoch_grads_per_layer = {name: [] for name, _ in model.named_parameters()}
        
        print(f"  Starting epoch {epoch}, processing {len(train_loader)} batches...")
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"    Processing batch {batch_idx}/{len(train_loader)}")
            
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradients per layer
            layer_grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    layer_grad_norms[name] = grad_norm
                    epoch_grads_per_layer[name].append(grad_norm)
                else:
                    layer_grad_norms[name] = 0.0
                    epoch_grads_per_layer[name].append(0.0)
            
            # Compute overall gradient norm
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            
            # Apply gradient clipping if specified
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            # Check for vanishing gradients (very small gradients)
            if gradient_clip_min is not None:
                if total_grad_norm < gradient_clip_min:
                    # Scale up very small gradients (one mitigation technique)
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = param.grad.data * (gradient_clip_min / (total_grad_norm + 1e-8))
            
            # Track minimum gradient norm across layers
            min_layer_grad = min([v for v in layer_grad_norms.values() if v > 0] or [0])
            epoch_min_grad_norms.append(min_layer_grad)
            
            epoch_grad_norms.append(total_grad_norm.item())
            optimizer.step()
            
            train_loss += loss.item()
            
            if torch.isnan(loss):
                print(f"NaN at epoch {epoch}, batch {batch_idx}!")
                break
        
        last_10_grads = epoch_grad_norms[-10:]
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        grad_norms.append(np.mean(epoch_grad_norms))
        min_grad_norms.append(np.mean(epoch_min_grad_norms))
        
        # Average gradients per layer
        avg_grads_per_layer = {name: np.mean(vals) for name, vals in epoch_grads_per_layer.items()}
        grad_norms_per_layer.append(avg_grads_per_layer)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, GradNorm={np.mean(epoch_grad_norms):.2e}, "
              f"MinLayerGrad={np.mean(epoch_min_grad_norms):.2e}")
        
        if torch.isnan(loss):
            break
    
    return train_losses, grad_norms, min_grad_norms, grad_norms_per_layer, validation_losses, last_10_grads


def plot_vanishing_gradients(train_losses, grad_norms, min_grad_norms, grad_norms_per_layer, 
                            last_10_grads, model_type, mitigation_type, save_dir='plots/vanishing_gradient'):
    """Create comprehensive plots for vanishing gradient analysis"""
    
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(0, len(train_losses))
    
    # Plot 1: Training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss - {model_type.upper()} ({mitigation_type})')
    plt.yscale('log')  # Log scale to see slow learning
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/training_loss_{model_type}_{mitigation_type}.png', dpi=150)
    plt.close()
    
    # Plot 2: Overall gradient norms
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norms, marker='o', markersize=3, label='Total Gradient Norm')
    plt.plot(epochs, min_grad_norms, marker='s', markersize=3, label='Min Layer Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title(f'Gradient Norms - {model_type.upper()} ({mitigation_type})')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/grad_norms_{model_type}_{mitigation_type}.png', dpi=150)
    plt.close()
    
    # Plot 3: Gradients per layer (last epoch)
    if grad_norms_per_layer:
        plt.figure(figsize=(12, 6))
        last_layer_grads = grad_norms_per_layer[-1]
        layer_names = list(last_layer_grads.keys())
        grad_values = list(last_layer_grads.values())
        
        plt.barh(range(len(layer_names)), grad_values)
        plt.yticks(range(len(layer_names)), layer_names)
        plt.xlabel('Gradient Norm')
        plt.title(f'Gradient Norms per Layer (Last Epoch) - {model_type.upper()} ({mitigation_type})')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/grad_per_layer_{model_type}_{mitigation_type}.png', dpi=150)
        plt.close()
    
    # Plot 4: Last 10 batch gradients
    batches = np.arange(1, len(last_10_grads) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(batches, last_10_grads, marker='o', markersize=5)
    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.title(f'Gradient Norms - Last 10 Batches - {model_type.upper()} ({mitigation_type})')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/grad_norms_last_10_{model_type}_{mitigation_type}.png', dpi=150)
    plt.close()
    
    # Plot 5: Gradient evolution over time (heatmap of layer gradients)
    if len(grad_norms_per_layer) > 0 and len(grad_norms_per_layer[0]) > 0:
        layer_names = list(grad_norms_per_layer[0].keys())
        grad_matrix = np.array([[epoch[layer] for layer in layer_names] 
                               for epoch in grad_norms_per_layer])
        
        plt.figure(figsize=(12, 8))
        im = plt.imshow(grad_matrix.T, aspect='auto', cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
        plt.colorbar(im, label='Gradient Norm')
        plt.xlabel('Epoch')
        plt.ylabel('Layer')
        plt.yticks(range(len(layer_names)), layer_names)
        plt.title(f'Gradient Norm Evolution Across Layers - {model_type.upper()} ({mitigation_type})')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/grad_heatmap_{model_type}_{mitigation_type}.png', dpi=150)
        plt.close()


if __name__ == '__main__':
    # Configuration for vanishing gradient experiments
    # Vanishing gradients occur with: deep networks, tanh activation, small learning rates
    
    DATA_CONFIG = {
        "use_normalized": True,  # Normalized data helps with training stability
        "add_noise": False,
        "noise_std": 0.05,
        "use_pca": False,
        "pca_components": 512,
        "val_split": 0.2,
        "batch_size": 64,
        "shuffle": True,
    }
    
    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessed_dir="preprocessed",
        config=DATA_CONFIG
    )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Experiment configurations
    experiments = [
        {
            'name': 'vanilla_tanh_no_mitigation',
            'model_class': VanillaRNN,
            'model_kwargs': {'input_size': 1, 'hidden_size': 256, 'num_layers': 20, 
                           'nonlinearity': 'tanh', 'num_classes': 10, 'init_method': 'default'},
            'train_kwargs': {'learning_rate': 0.01, 'gradient_clip': None, 'gradient_clip_min': None},
            'model_type': 'vanilla',
            'mitigation': 'no_mitigation'
        },
        {
            'name': 'vanilla_tanh_xavier',
            'model_class': VanillaRNN,
            'model_kwargs': {'input_size': 1, 'hidden_size': 256, 'num_layers': 20,
                           'nonlinearity': 'tanh', 'num_classes': 10, 'init_method': 'xavier'},
            'train_kwargs': {'learning_rate': 0.01, 'gradient_clip': None, 'gradient_clip_min': None},
            'model_type': 'vanilla',
            'mitigation': 'xavier_init'
        },
        {
            'name': 'residual_relu',
            'model_class': ResidualRNN,
            'model_kwargs': {'input_size': 1, 'hidden_size': 256, 'num_layers': 20,
                           'nonlinearity': 'relu', 'num_classes': 10, 'init_method': 'xavier'},
            'train_kwargs': {'learning_rate': 0.01, 'gradient_clip': None, 'gradient_clip_min': None},
            'model_type': 'residual',
            'mitigation': 'residual_connections'
        },
    ]
    
    # Run experiments
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*60}")
        
        print(f"  Creating model: {exp['name']}")
        model = exp['model_class'](**exp['model_kwargs']).to(device)
        print(f"  Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Starting training...")
        
        # Test forward pass first
        print(f"  Testing forward pass...")
        test_batch = next(iter(train_loader))
        test_seq, test_target = test_batch
        test_seq = test_seq.to(device)
        with torch.no_grad():
            test_out = model(test_seq)
        print(f"  Forward pass successful. Output shape: {test_out.shape}")
        
        train_losses, grad_norms, min_grad_norms, grad_norms_per_layer, validation_losses, last_10_grads = \
            train_model(model, train_loader, device, num_epochs=50, model_type=exp['model_type'], **exp['train_kwargs'])
        
        plot_vanishing_gradients(
            train_losses, grad_norms, min_grad_norms, grad_norms_per_layer,
            last_10_grads, exp['model_type'], exp['mitigation']
        )
        
        print(f"Completed: {exp['name']}")
