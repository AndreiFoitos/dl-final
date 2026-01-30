import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from data.data_pipeline import create_dataloaders
from cnn_model_exploding import CNN


def track_weight_norms(model):
    """Track weight norms for each layer."""
    weight_norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_norms[name] = param.data.norm(2).item()
    return weight_norms


class ActivationMonitor:
    """Helper class to register hooks and track activation statistics."""
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def hook_fn(self, name):
        def fn(module, input, output):
            out_flat = output.detach().view(-1)
            if torch.isnan(out_flat).any():
                mean_val = float('nan')
                std_val = float('nan')
            else:
                mean_val = out_flat.mean().item()
                std_val = out_flat.std().item()
                
            self.activations[name] = {
                'mean': mean_val,
                'std': std_val
            }
        return fn

    def register_hooks(self, model):
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self.hooks.append(layer.register_forward_hook(self.hook_fn(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_activations(self):
        self.activations = {}


def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=1, mitigation_on=False):
    """
    Training loop for CNN model with explosion monitoring.
    """
    
    if mitigation_on:
        gradient_clip = 0.5
        weight_decay = 1e-4
        mitigation = 'mitigation'
    else:
        mitigation = 'no_mitigation'

    criterion = nn.CrossEntropyLoss()

    if not mitigation_on:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    monitor = ActivationMonitor()
    monitor.register_hooks(model)
        
    train_losses = []
    grad_norms = []
    grad_norms_per_layer = []
    weight_norms_per_epoch = []
    activation_stats_per_epoch = []
    validation_losses = []

    print(f"Starting training with mitigation={mitigation_on}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_grad_norms = []
        epoch_grads_per_layer = {name: [] for name, _ in model.named_parameters()}
        
        epoch_activation_captured = False
        current_epoch_activations = {}

        print(f"Epoch {epoch}: {len(train_loader)} batches")
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)

            monitor.clear_activations()

            predictions = model(sequences)
            loss = criterion(predictions, targets)

            if torch.isnan(loss):
                print(f"!!! NaN Loss detected at epoch {epoch}, batch {batch_idx} !!!")
                current_epoch_activations = monitor.activations.copy()
                activation_stats_per_epoch.append(current_epoch_activations)
                train_losses.append(np.nan)
                weight_norms_per_epoch.append(track_weight_norms(model))
                grad_norms_per_layer.append({})
                grad_norms.append(np.nan)
                return train_losses, grad_norms, grad_norms_per_layer, weight_norms_per_epoch, [], validation_losses, mitigation, activation_stats_per_epoch

            if not epoch_activation_captured:
                current_epoch_activations = monitor.activations.copy()
                epoch_activation_captured = True

            optimizer.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad.data.norm(2).item()
                    epoch_grads_per_layer[name].append(g)
                else:
                    epoch_grads_per_layer[name].append(0.0)

            if mitigation_on:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            epoch_grad_norms.append(grad_norm.item())
            optimizer.step()
            
            train_loss += loss.item()

            # if batch_idx % 10 == 0:
            #     print(
            #         f"  batch {batch_idx + 1}/{len(train_loader)} "
            #         f"| loss={loss.item():.4f} | grad_norm={grad_norm.item():.2e}"
            #     )

        last_10_grads = epoch_grad_norms[-10:]
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
                
        mean_grad = np.mean(epoch_grad_norms)
        grad_norms.append(mean_grad)

        avg_grads_per_layer = {
            name: np.mean(vals) for name, vals in epoch_grads_per_layer.items()
        }
        grad_norms_per_layer.append(avg_grads_per_layer)

        weight_norms_per_epoch.append(track_weight_norms(model))
        activation_stats_per_epoch.append(current_epoch_activations)

        if mitigation_on:
            scheduler.step()
            last_lr = scheduler.get_last_lr()
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, GradNorm={mean_grad:.2e}, LR={last_lr[0]:.6f}")
        else:
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, GradNorm={mean_grad:.2e}")

    monitor.remove_hooks()

    return (train_losses, grad_norms, grad_norms_per_layer, 
            weight_norms_per_epoch, last_10_grads, validation_losses, 
            mitigation, activation_stats_per_epoch)


def plot_exploding_cnn(train_losses, grad_norms, grad_norms_per_layer, weight_norms_per_epoch, 
                       last_10_grads, mitigation, activation_stats, learning_rate=0.0, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join("plots", "exploding", mitigation)
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = np.arange(0, len(train_losses))

    # Training loss plot
    plt.figure()
    plt.plot(epochs, train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title(f'Training loss - {mitigation}')
    plt.savefig(f'{save_dir}/1_training_loss_{mitigation}.png', dpi=150)
    plt.close()

    # Gradient norms over epochs (log scale)
    plt.figure()
    valid_grads = [g if not np.isnan(g) else 0 for g in grad_norms]
    plt.plot(epochs, valid_grads, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title(f'Total Gradient Norm - {mitigation}')
    plt.yscale('log')
    plt.savefig(f'{save_dir}/2_grad_norms_{mitigation}.png', dpi=150)
    plt.close()

    # Gradient Depth Slope (first epoch vs last valid epoch)
    if grad_norms_per_layer and len(grad_norms_per_layer) > 0:
        plt.figure(figsize=(12, 6))
        
        first_epoch_grads = grad_norms_per_layer[0]
        layer_names = [k for k in first_epoch_grads.keys() if 'weight' in k]
        
        first_epoch_values = [first_epoch_grads.get(k, 0) for k in layer_names]
        plt.plot(range(len(layer_names)), first_epoch_values, marker='o', label='Epoch 0', linewidth=2)
        
        if len(grad_norms_per_layer) > 1:
            last_idx = -1
            if not grad_norms_per_layer[last_idx]: 
                last_idx = -2
            
            if abs(last_idx) <= len(grad_norms_per_layer):
                last_epoch_grads = grad_norms_per_layer[last_idx]
                last_epoch_values = [last_epoch_grads.get(k, 0) for k in layer_names]
                plt.plot(range(len(layer_names)), last_epoch_values, marker='x', linestyle='--', label='Last Valid Epoch')

        plt.xticks(range(len(layer_names)), [n.replace('.weight', '') for n in layer_names], rotation=90)
        plt.yscale('log')
        plt.xlabel('Layer Depth')
        plt.ylabel('Gradient Norm (Log Scale)')
        plt.title(f'Gradient Magnitude vs Depth - {mitigation}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/3_grad_depth_slope_{mitigation}.png', dpi=150)
        plt.close()

    if activation_stats and len(activation_stats) > 0:
        plt.figure(figsize=(12, 6))
        
        last_stats = activation_stats[-1]
        layer_names = list(last_stats.keys())
        stds = [last_stats[k]['std'] for k in layer_names]
        
        clean_stds = [0 if np.isnan(x) else x for x in stds]
        
        plt.plot(range(len(layer_names)), clean_stds, marker='o', color='red')
        plt.xticks(range(len(layer_names)), layer_names, rotation=90)
        plt.xlabel("Layer")
        plt.ylabel("Activation Std Dev")
        plt.title(f'Activation Explosion (Last State) - {mitigation}')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/4_activation_explosion_{mitigation}.png", dpi=150)
        plt.close()

    if weight_norms_per_epoch and grad_norms_per_layer:
        plt.figure(figsize=(10, 6))
        
        if len(weight_norms_per_epoch[0]) > 0:
            target_layer = list(weight_norms_per_epoch[0].keys())[0]
            
            ratios = []
            for i in range(len(weight_norms_per_epoch)):
                if not weight_norms_per_epoch[i] or not grad_norms_per_layer[i]:
                    continue
                    
                w_norm = weight_norms_per_epoch[i].get(target_layer, 1.0)
                g_norm = grad_norms_per_layer[i].get(target_layer, 0.0)
                
                if w_norm == 0: w_norm = 1e-9
                ratio = (learning_rate * g_norm) / w_norm
                ratios.append(ratio)
                
            plt.plot(range(len(ratios)), ratios, marker='o')
            plt.yscale('log')
            plt.xlabel("Epoch")
            plt.ylabel("Update Ratio")
            plt.title(f'Weight Update Ratio ({target_layer}) - {mitigation}')
            plt.savefig(f"{save_dir}/5_update_ratio_{mitigation}.png", dpi=150)
            plt.close()

    print(f"All plots saved to {save_dir}/")


if __name__ == '__main__':
    # For exploding set mitigation_on to False
    # For mitigation set mitigation_on to True

    DATA_CONFIG = {
        "use_normalized": False,     
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

    experiments = [
        {
            "name": "exploding_no_mitigation",
            "model_kwargs": {
                "num_conv_layers": 5,
                "base_channels": 32,
                "num_classes": 10
            },
            "train_kwargs": {
                "learning_rate": 0.35,
                "mitigation_on": False
            }
        },
        # Uncomment to run mitigation experiment
        # {
        #     "name": "exploding_with_mitigation",
        #     "model_kwargs": {
        #         "num_conv_layers": 5,
        #         "base_channels": 32,
        #         "num_classes": 10
        #     },
        #     "train_kwargs": {
        #         "learning_rate": 0.35,
        #         "mitigation_on": True
        #     }
        # }
    ]

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        print("Model config:", exp["model_kwargs"])
        print("Training config:", exp["train_kwargs"])

        model = CNN(**exp["model_kwargs"]).to(device)
        print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

        (train_losses, grad_norms, grad_norms_per_layer, weight_norms_per_epoch, 
         last_10_grads, validation_losses, mitigation, activation_stats) = train_model(
            model, 
            train_loader, 
            val_loader, 
            device,
            num_epochs=100,
            **exp["train_kwargs"]
        )

        save_dir = os.path.join("plots", "exploding", exp["name"])
        plot_exploding_cnn(
            train_losses,
            grad_norms,
            grad_norms_per_layer,
            weight_norms_per_epoch,
            last_10_grads,
            mitigation,
            activation_stats,
            learning_rate=exp["train_kwargs"]["learning_rate"],
            save_dir=save_dir,
        )

        final_loss = train_losses[-1] if train_losses else 0
        final_grad = grad_norms[-1] if grad_norms else 0
        
        print(f"\nCompleted: {exp['name']}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Final grad norm: {final_grad:.2e}")