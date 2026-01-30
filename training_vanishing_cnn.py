import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from data.data_pipeline import create_dataloaders
from cnn_model_vanishing import VanishingCNN, ResidualCNN


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
            self.activations[name] = {
                'mean': out_flat.mean().item(),
                'std': out_flat.std().item()
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


def train_model(model, train_loader, device, num_epochs=50, learning_rate=0.01, gradient_clip=None, weight_decay=0.0, model_type="cnn"):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    monitor = ActivationMonitor()
    monitor.register_hooks(model)

    train_losses = []
    grad_norms = []
    grad_norms_per_layer = []
    weight_norms_per_epoch = []
    activation_stats_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        epoch_grad_norms = []
        epoch_grads_per_layer = {name: [] for name, _ in model.named_parameters()}
        
        # Capture activation stats from the first batch of the epoch only
        epoch_activation_captured = False
        current_epoch_activations = {}

        print(f"Epoch {epoch}: {len(train_loader)} batches")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Clear previous batch activations in monitor
            monitor.clear_activations()

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Capture activations for this epoch
            if not epoch_activation_captured:
                current_epoch_activations = monitor.activations.copy()
                epoch_activation_captured = True

            optimizer.zero_grad()
            loss.backward()

            # Track per-layer gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad.data.norm(2).item()
                    epoch_grads_per_layer[name].append(g)
                else:
                    epoch_grads_per_layer[name].append(0.0)

            # Track total gradient norm
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            epoch_grad_norms.append(total_grad_norm.item())
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"  batch {batch_idx}/{len(train_loader)} "
                    f"| grad_norm={total_grad_norm.item():.2e}"
                )

        
        # Store Loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Store Average Grads
        mean_grad = np.mean(epoch_grad_norms)
        grad_norms.append(mean_grad)

        # Store Layer-wise Grads
        avg_grads_per_layer = {
            name: np.mean(vals) for name, vals in epoch_grads_per_layer.items()
        }
        grad_norms_per_layer.append(avg_grads_per_layer)

        # Store Weight Norms
        weight_norms_per_epoch.append(track_weight_norms(model))

        # Store Activation Stats
        activation_stats_per_epoch.append(current_epoch_activations)
        
        # Store Last 10 grads
        last_10_grads = epoch_grad_norms[-10:]

        print(
            f"Epoch {epoch}: Loss={train_loss:.4f}, "
            f"GradNorm={mean_grad:.2e}"
        )

    monitor.remove_hooks()

    return (train_losses, grad_norms, grad_norms_per_layer, 
            last_10_grads, weight_norms_per_epoch, activation_stats_per_epoch)


def plot_vanishing_cnn(train_losses, grad_norms, grad_norms_per_layer, last_10_grads, 
                       weight_norms, activation_stats,
                       model_type, mitigation_type, learning_rate, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join("plots", "vanishing", mitigation_type)
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(0, len(train_losses))
    
    # Training loss
    plt.figure()
    plt.plot(epochs, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(f"Training Loss - {mitigation_type}")
    plt.savefig(f"{save_dir}/1_training_loss_{mitigation_type}.png", dpi=150)
    plt.close()

    # Total Gradient Norm
    plt.figure()
    plt.plot(epochs, grad_norms, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm")
    plt.title(f"Total Gradient Norm - {mitigation_type}")
    plt.yscale("log")
    plt.savefig(f"{save_dir}/2_total_grad_norms_{mitigation_type}.png", dpi=150)
    plt.close()

    if grad_norms_per_layer:
        plt.figure(figsize=(12, 6))
        
        # Get layers that have 'weight' in them
        first_epoch_grads = grad_norms_per_layer[0]
        layer_names = [k for k in first_epoch_grads.keys() if 'weight' in k]
        
        # Plot First Epoch
        first_epoch_values = [first_epoch_grads[k] for k in layer_names]
        plt.plot(range(len(layer_names)), first_epoch_values, marker='o', label='Epoch 0 (Start)', linewidth=2)
        
        # Plot Last Epoch
        last_epoch_grads = grad_norms_per_layer[-1]
        last_epoch_values = [last_epoch_grads[k] for k in layer_names]
        plt.plot(range(len(layer_names)), last_epoch_values, marker='x', linestyle='--', label=f'Epoch {len(epochs)-1} (End)')

        plt.xticks(range(len(layer_names)), [n.replace('.weight', '') for n in layer_names], rotation=90)
        plt.yscale('log')
        plt.xlabel('Layer Depth (Input -> Output)')
        plt.ylabel('Gradient Norm (Log Scale)')
        plt.title(f'Gradient Magnitude vs Depth (The Slope) - {mitigation_type}')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/3_gradient_depth_slope_{mitigation_type}.png", dpi=150)
        plt.close()

    if grad_norms_per_layer:
        plt.figure(figsize=(10, 6))
        
        # Identify first conv and last fc weights
        all_keys = list(grad_norms_per_layer[0].keys())
        first_layer_name = all_keys[0] # Usually conv1.weight
        last_layer_name = all_keys[-2] # Usually fc.weight (last one might be bias)

        first_layer_trace = [epoch[first_layer_name] for epoch in grad_norms_per_layer]
        last_layer_trace = [epoch[last_layer_name] for epoch in grad_norms_per_layer]
        
        plt.plot(epochs, first_layer_trace, label=f'First Layer ({first_layer_name})', linewidth=2)
        plt.plot(epochs, last_layer_trace, label=f'Last Layer ({last_layer_name})', linewidth=2, linestyle='--')
        
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title(f'First vs Last Layer Gradients - {mitigation_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/4_first_vs_last_layer_{mitigation_type}.png", dpi=150)
        plt.close()

    if activation_stats and len(activation_stats) > 0:
        plt.figure(figsize=(12, 6))
        
        # Get stats from the last epoch
        last_epoch_stats = activation_stats[-1]
        layer_names = list(last_epoch_stats.keys())
        stds = [last_epoch_stats[k]['std'] for k in layer_names]
        
        plt.plot(range(len(layer_names)), stds, marker='o')
        plt.xticks(range(len(layer_names)), layer_names, rotation=90)
        plt.xlabel("Layer")
        plt.ylabel("Activation Standard Deviation")
        plt.title(f'Activation Std Dev per Layer (Last Epoch) - {mitigation_type}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/5_activation_std_{mitigation_type}.png", dpi=150)
        plt.close()

    if weight_norms and grad_norms_per_layer:
        plt.figure(figsize=(10, 6))
        
        # Calculate for the first layer (most prone to vanishing)
        target_layer = list(weight_norms[0].keys())[0] # first layer
        
        ratios = []
        for i in range(len(weight_norms)):
            w_norm = weight_norms[i].get(target_layer, 1.0)
            g_norm = grad_norms_per_layer[i].get(target_layer, 0.0)
            
            # Ratio = (lr * grad) / weight
            if w_norm == 0: w_norm = 1e-9
            ratio = (learning_rate * g_norm) / w_norm
            ratios.append(ratio)
            
        plt.plot(epochs, ratios, marker='o')
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Update Ratio")
        plt.title(f'Weight Update Ratio ({target_layer}) - {mitigation_type}\n(Learning Rate * Grad / Weight)')
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(f"{save_dir}/6_update_ratio_{mitigation_type}.png", dpi=150)
        plt.close()
        
    print(f"All 6 diagnostic plots saved to {save_dir}/")


if __name__ == "__main__":
    DATA_CONFIG = {
        "use_normalized": True,
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
        config=DATA_CONFIG,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    experiments = [
        {
            "name": "cnn_sigmoid_no_mitigation",
            "model_class": VanishingCNN,
            "model_kwargs": {
                "num_conv_layers": 20,
                "base_channels": 16,
                "num_classes": 10,
                "init_method": "small",
            },
            "train_kwargs": {
                "learning_rate": 0.01,
                "gradient_clip": None,
                "weight_decay": 0.0,
            },
            "model_type": "cnn",
            "mitigation": "no_mitigation",
        },
        # Uncomment below to run the mitigation experiments
        # {
        #     "name": "cnn_residual_he",
        #     "model_class": ResidualCNN,
        #     "model_kwargs": {
        #         "num_blocks": 6,
        #         "base_channels": 32,
        #         "num_classes": 10,
        #         "init_method": "he",
        #     },
        #     "train_kwargs": {
        #         "learning_rate": 0.01,
        #         "gradient_clip": None,
        #         "weight_decay": 1e-4,
        #     },
        #     "model_type": "cnn_residual",
        #     "mitigation": "residual_he",
        # },
    ]

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        print("  Model config:", exp["model_kwargs"])
        print("  Training config:", exp["train_kwargs"])

        print("  Creating model...")
        model = exp["model_class"](**exp["model_kwargs"]).to(device)
        print(f"  Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

        (train_losses, grad_norms, grad_norms_per_layer, 
         last_10_grads, weight_norms, activation_stats) = train_model(
            model,
            train_loader,
            device,
            num_epochs=5,
            model_type=exp["model_type"],
            **exp["train_kwargs"],
        )

        save_dir = os.path.join("plots", "vanishing", exp["name"])
        plot_vanishing_cnn(
            train_losses,
            grad_norms,
            grad_norms_per_layer,
            last_10_grads,
            weight_norms,
            activation_stats,
            exp["model_type"],
            exp["mitigation"],
            learning_rate=exp["train_kwargs"]["learning_rate"],
            save_dir=save_dir,
        )

        print(f"Completed: {exp['name']}")