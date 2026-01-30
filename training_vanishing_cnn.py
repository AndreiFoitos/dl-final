import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from data.data_pipeline import create_dataloaders
from cnn_model_vanishing import VanishingCNN, ResidualCNN


def train_model(model, train_loader, device, num_epochs=50, learning_rate=0.01, gradient_clip=None, weight_decay=0.0, model_type="cnn"):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    grad_norms = []
    grad_norms_per_layer = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        epoch_grad_norms = []
        epoch_grads_per_layer = {name: [] for name, _ in model.named_parameters()}

        print(f"Epoch {epoch}: {len(train_loader)} batches")
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            layer_grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad.data.norm(2).item()
                    layer_grad_norms[name] = g
                    epoch_grads_per_layer[name].append(g)
                else:
                    layer_grad_norms[name] = 0.0
                    epoch_grads_per_layer[name].append(0.0)

            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            epoch_grad_norms.append(total_grad_norm.item())
            optimizer.step()

            train_loss += loss.item()

            print(
                f"  batch {batch_idx + 1}/{len(train_loader)} "
                f"| grad_norm={total_grad_norm.item():.2e}"
            )

        last_10_grads = epoch_grad_norms[-10:]
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        mean_grad = np.mean(epoch_grad_norms)
        grad_norms.append(mean_grad)

        avg_grads_per_layer = {
            name: np.mean(vals) for name, vals in epoch_grads_per_layer.items()
        }
        grad_norms_per_layer.append(avg_grads_per_layer)

        print(
            f"Epoch {epoch}: Loss={train_loss:.4f}, "
            f"GradNorm={mean_grad:.2e}"
        )

    return train_losses, grad_norms, grad_norms_per_layer, last_10_grads


def plot_vanishing_cnn(train_losses, grad_norms, grad_norms_per_layer, last_10_grads, model_type, mitigation_type, save_dir="plots/vanishing_cnn"):
    
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(0, len(train_losses))

    # Training loss plot
    plt.figure()
    plt.plot(epochs, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss over epochs")
    plt.savefig(
        f"{save_dir}/training_loss_{mitigation_type}.png",
        dpi=150,
    )
    plt.close()

    # Overall gradient norms plot
    plt.figure()
    plt.plot(epochs, grad_norms, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm")
    plt.title("Gradient norm over epochs")
    plt.yscale("log")
    plt.savefig(
        f"{save_dir}/grad_norms_{mitigation_type}.png",
        dpi=150,
    )
    plt.close()

    # Last 10 batch gradients plot
    batches = np.arange(1, len(last_10_grads) + 1)
    plt.figure()
    plt.plot(batches, last_10_grads, marker="o")
    plt.xlabel("Batches")
    plt.ylabel("Gradient norm")
    plt.title("Gradient norm over last 10 batches")
    plt.yscale("log")
    plt.savefig(
        f"{save_dir}/grad_norms_last_10_{mitigation_type}.png",
        dpi=150,
    )
    plt.close()

    # Gradients per layer plot
    if grad_norms_per_layer:
        plt.figure(figsize=(12, 6))
        last_layer_grads = grad_norms_per_layer[-1]
        layer_names = list(last_layer_grads.keys())
        grad_values = list(last_layer_grads.values())

        plt.barh(range(len(layer_names)), grad_values)
        plt.yticks(range(len(layer_names)), layer_names)
        plt.xlabel("Gradient norm")
        plt.title(
            f"Gradient norms per layer (last epoch) - {model_type.upper()} "
            f"({mitigation_type})"
        )
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/grad_per_layer_{model_type}_{mitigation_type}.png",
            dpi=150,
        )
        plt.close()

    # Gradient evolution heatmap plot
    if len(grad_norms_per_layer) > 0 and len(grad_norms_per_layer[0]) > 0:
        layer_names = list(grad_norms_per_layer[0].keys())
        grad_matrix = np.array(
            [[epoch[layer] for layer in layer_names] for epoch in grad_norms_per_layer]
        )

        plt.figure(figsize=(12, 8))
        im = plt.imshow(
            grad_matrix.T,
            aspect="auto",
            cmap="viridis",
            norm=plt.matplotlib.colors.LogNorm(),
        )
        plt.colorbar(im, label="Gradient norm")
        plt.xlabel("Epoch")
        plt.ylabel("Layer")
        plt.yticks(range(len(layer_names)), layer_names)
        plt.title(
            f"Gradient norm evolution across layers - {model_type.upper()} "
            f"({mitigation_type})"
        )
        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/grad_heatmap_{model_type}_{mitigation_type}.png",
            dpi=150,
        )
        plt.close()


if __name__ == "__main__":
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
                "learning_rate": 0.00001,
                "gradient_clip": None,
                "weight_decay": 0.0,
            },
            "model_type": "cnn",
            "mitigation": "no_mitigation",
        },
        # {
        #     "name": "cnn_sigmoid_xavier",
        #     "model_class": VanishingCNN,
        #     "model_kwargs": {
        #         "num_conv_layers": 10,
        #         "base_channels": 16,
        #         "num_classes": 10,
        #         "init_method": "xavier",
        #     },
        #     "train_kwargs": {
        #         "learning_rate": 0.01,
        #         "gradient_clip": None,
        #         "weight_decay": 0.0,
        #     },
        #     "model_type": "cnn",
        #     "mitigation": "xavier_init",
        # },
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

        (train_losses, grad_norms, grad_norms_per_layer, last_10_grads) = train_model(
            model,
            train_loader,
            device,
            num_epochs=5,
            model_type=exp["model_type"],
            **exp["train_kwargs"],
        )

        plot_vanishing_cnn(
            train_losses,
            grad_norms,
            grad_norms_per_layer,
            last_10_grads,
            exp["model_type"],
            exp["mitigation"],
        )

        print(f"Completed: {exp['name']}")
