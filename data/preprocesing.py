import pickle
import numpy as np
import os
import tarfile



#Before doing anything, run: pip install -r requirements.txt
#then download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
#and place the cifar-10-python.tar.gz file in the data/ directory.
# This script will extract and preprocess the data, saving both raw and normalized versions.



# -------------------------
# Configuration
# -------------------------
TAR_FILE = "cifar-10-python.tar.gz"
CIFAR_DIR = "cifar-10-batches-py"
SAVE_DIR = "preprocessed"

# CIFAR-10 mean/std (RGB)
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# -------------------------
# Extract dataset
# -------------------------
if not os.path.exists(CIFAR_DIR):
    print("Extracting CIFAR-10...")
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        tar.extractall()
    print("Extraction done.")

# -------------------------
# Utility functions
# -------------------------
def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")

def load_batch(file):
    batch = unpickle(file)
    X = batch[b"data"]           # (N, 3072)
    y = np.array(batch[b"labels"])
    return X, y

def normalize(X):
    """ Normalize flattened CIFAR-10 images """
    X = X.reshape(-1, 3, 32, 32)
    X = X / 255.0
    mean = CIFAR10_MEAN.reshape(1, 3, 1, 1)
    std = CIFAR10_STD.reshape(1, 3, 1, 1)
    X = (X - mean) / std
    return X.reshape(X.shape[0], -1)

# -------------------------
# Load training data
# -------------------------
X_train, y_train = [], []

for i in range(1, 6):
    X, y = load_batch(os.path.join(CIFAR_DIR, f"data_batch_{i}"))
    X_train.append(X)
    y_train.append(y)

X_train = np.concatenate(X_train, axis=0).astype(np.float32)
y_train = np.concatenate(y_train, axis=0)

# -------------------------
# Load test data
# -------------------------
X_test, y_test = load_batch(os.path.join(CIFAR_DIR, "test_batch"))
X_test = X_test.astype(np.float32)

# -------------------------
# Create raw and normalized versions
# -------------------------
X_train_raw = X_train / 255.0
X_test_raw  = X_test / 255.0

X_train_norm = normalize(X_train)
X_test_norm  = normalize(X_test)

X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], 3072, 1)
X_test_raw  = X_test_raw.reshape(X_test_raw.shape[0], 3072, 1)
X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], 3072, 1)
X_test_norm  = X_test_norm.reshape(X_test_norm.shape[0], 3072, 1)

# -------------------------
# Save
# -------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

np.save(os.path.join(SAVE_DIR, "X_train_raw.npy"), X_train_raw)
np.save(os.path.join(SAVE_DIR, "X_test_raw.npy"), X_test_raw)

np.save(os.path.join(SAVE_DIR, "X_train_norm.npy"), X_train_norm)
np.save(os.path.join(SAVE_DIR, "X_test_norm.npy"), X_test_norm)

np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

print("Preprocessing complete.")
print("Raw train shape:", X_train_raw.shape)
print("Normalized train shape:", X_train_norm.shape)
print("Raw test shape:", X_test_raw.shape)
print("Normalized test shape:", X_test_norm.shape)

# Phase 1 — Gradient Instability use X_train_raw.npy
# Phase 2 — Advanced Optimization use X_train_norm.npy

#I guess normlization of the data can be one of the mitigation techniques for gradient instability?