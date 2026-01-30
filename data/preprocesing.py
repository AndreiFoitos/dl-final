import pickle
import numpy as np
import os
import tarfile
import shutil

TAR_FILE = "data/cifar-10-python.tar.gz"
CIFAR_DIR = "data/cifar-10-batches-py"
SAVE_DIR = "data/preprocessed"

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

if os.path.exists("cifar-10-batches-py") and not os.path.exists(CIFAR_DIR):
    print("Moving CIFAR-10 dataset to data/ directory...")
    shutil.move("cifar-10-batches-py", CIFAR_DIR)
    print("Move complete.")

if not os.path.exists(CIFAR_DIR):
    print("Extracting CIFAR-10...")
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        tar.extractall("data/")
    print("Extraction done.")


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def load_batch(file):
    batch = unpickle(file)
    X = batch[b"data"]
    y = np.array(batch[b"labels"])
    return X, y


X_train, y_train = [], []

for i in range(1, 6):
    X, y = load_batch(os.path.join(CIFAR_DIR, f"data_batch_{i}"))
    X_train.append(X)
    y_train.append(y)

X_train = np.concatenate(X_train, axis=0).astype(np.float32)
y_train = np.concatenate(y_train, axis=0)

X_test, y_test = load_batch(os.path.join(CIFAR_DIR, "test_batch"))
X_test = X_test.astype(np.float32)

X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

X_train_raw = X_train / 255.0
X_test_raw = X_test / 255.0

mean = CIFAR10_MEAN.reshape(1, 3, 1, 1)
std = CIFAR10_STD.reshape(1, 3, 1, 1)

X_train_norm = (X_train_raw - mean) / std
X_test_norm = (X_test_raw - mean) / std

os.makedirs(SAVE_DIR, exist_ok=True)

np.save(os.path.join(SAVE_DIR, "X_train_raw_cnn.npy"), X_train_raw)
np.save(os.path.join(SAVE_DIR, "X_test_raw_cnn.npy"), X_test_raw)
np.save(os.path.join(SAVE_DIR, "X_train_norm_cnn.npy"), X_train_norm)
np.save(os.path.join(SAVE_DIR, "X_test_norm_cnn.npy"), X_test_norm)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

print("Preprocessing complete.")
print("Raw train shape:", X_train_raw.shape)
print("Normalized train shape:", X_train_norm.shape)
print("Raw test shape:", X_test_raw.shape)
print("Normalized test shape:", X_test_norm.shape)