import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

# -------------------------
# Configuration Defaults
# -------------------------
DEFAULT_CONFIG = {
    "use_normalized": True,      # raw vs normalized data
    "add_noise": False,          # Gaussian noise
    "noise_std": 0.05,
    "use_pca": False,            # Optional PCA
    "pca_components": 512,
    "val_split": 0.1,
    "batch_size": 128,
    "shuffle": True,
}


# -------------------------
# Feature engineering
# -------------------------
def add_gaussian_noise(X, std):
    noise = np.random.normal(0, std, size=X.shape).astype(np.float32)
    return X + noise

def apply_pca(X_train, X_val, X_test, n_components):
    pca = PCA(n_components=n_components, whiten=True)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    return X_train, X_val, X_test

# -------------------------
# Load data
# -------------------------
def load_data(preprocessed_dir, config=DEFAULT_CONFIG):

    # Choose raw vs normalized
    if config["use_normalized"]:
        X_train = np.load(f"data/{preprocessed_dir}/X_train_norm_cnn.npy")
        X_test = np.load(f"data/{preprocessed_dir}/X_test_norm_cnn.npy")
    else:
        X_train = np.load(f"data/{preprocessed_dir}/X_train_raw_cnn.npy")
        X_test = np.load(f"data/{preprocessed_dir}/X_test_raw_cnn.npy")

    y_train = np.load(f"data/{preprocessed_dir}/y_train.npy")
    y_test = np.load(f"data/{preprocessed_dir}/y_test.npy")

    return X_train, y_train, X_test, y_test

# -------------------------
# Train / validation split
# -------------------------
def train_val_split(X, y, val_split):
    N = X.shape[0]
    idx = np.random.permutation(N)

    val_size = int(N * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
    )

# -------------------------
# Main data pipeline
# -------------------------
def create_dataloaders(preprocessed_dir, config=DEFAULT_CONFIG):
    X_train, y_train, X_test, y_test = load_data(preprocessed_dir, config)

    # Train / validation split
    X_train, y_train, X_val, y_val = train_val_split(
        X_train, y_train, config["val_split"]
    )

    # Feature engineering
    if config["add_noise"]:
        X_train = add_gaussian_noise(X_train, config["noise_std"])

    if config["use_pca"]:
        X_train, X_val, X_test = apply_pca(
            X_train, X_val, X_test, config["pca_components"]
        )

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader, test_loader





""" this is how you load the data in your training script:
from data.data_pipeline import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    preprocessed_dir="preprocessed",
    config=DATA_CONFIG
)
"""


"""use data config like this in your training script:
DATA_CONFIG = {
    "use_normalized": False, #True for mitigation      
    "add_noise": False,   # true for mitigation      
    "noise_std": 0.05,
    "use_pca": False,            #true for mitigation 
    "pca_components": 512,
    "val_split": 0.2,
    "batch_size": 128,
    "shuffle": True,
}"""