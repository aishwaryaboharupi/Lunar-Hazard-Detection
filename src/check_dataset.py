import numpy as np
import os
import glob

PROCESSED_DIR = "dataset/processed/"

# Get batch file paths
X_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
y_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "y_batch_*.npy")))

# Determine dataset size (assuming all batches are of equal size)
sample_X = np.load(X_files[0])  # Load one batch to check shape
num_samples = sum(np.load(f).shape[0] for f in X_files)  # Total samples
X_shape = (num_samples,) + sample_X.shape[1:]  # (Total, H, W)
y_shape = (num_samples,)

# Use memory-mapped arrays to store dataset efficiently
X_train_path = os.path.join(PROCESSED_DIR, "X_train.npy")
y_train_path = os.path.join(PROCESSED_DIR, "y_train.npy")

X_train = np.memmap(X_train_path, dtype=np.float32, mode="w+", shape=X_shape)
y_train = np.memmap(y_train_path, dtype=np.int32, mode="w+", shape=y_shape)

# Load batch-wise to prevent memory overflow
start = 0
for x_file, y_file in zip(X_files, y_files):
    X_batch = np.load(x_file)
    y_batch = np.load(y_file)
    batch_size = X_batch.shape[0]

    # Store batch in memory-mapped file
    X_train[start : start + batch_size] = X_batch
    y_train[start : start + batch_size] = y_batch
    start += batch_size

# Ensure data is written
X_train.flush()
y_train.flush()

print(f"✅ Successfully saved large dataset using memory-mapping: {X_train_path}, {y_train_path}")
