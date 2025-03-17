import numpy as np
import os
import glob

PROCESSED_DIR = "dataset/processed/"

# Get all batch files
X_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
y_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "y_batch_*.npy")))

# Load and verify batches
for x_file, y_file in zip(X_files, y_files):
    X_batch = np.load(x_file)
    y_batch = np.load(y_file)

    print(f"✅ Loaded {x_file} with shape {X_batch.shape}")
    print(f"✅ Loaded {y_file} with shape {y_batch.shape}")

print("🔍 Verification complete!")

# Concatenate labels (smaller in size)
y_train = np.concatenate([np.load(f) for f in y_files])

# Use memory-efficient loading for images
X_shape = (len(y_train),) + X_batch.shape[1:]  # Assume all batches have same image size
X_train = np.memmap(os.path.join(PROCESSED_DIR, "X_train.npy"), dtype=np.float32, mode="w+", shape=X_shape)

offset = 0
for x_file in X_files:
    X_batch = np.load(x_file)
    batch_size = X_batch.shape[0]
    X_train[offset:offset + batch_size] = X_batch
    offset += batch_size

print(f"✅ Loaded {len(X_train)} images and {len(y_train)} labels.")

# Save y_train
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)

print("💾 Saved combined dataset as X_train.npy and y_train.npy.")
