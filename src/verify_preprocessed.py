import numpy as np
import os
import glob

PROCESSED_DIR = "dataset/processed/"

# Get all batch files
X_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
y_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "y_batch_*.npy")))

# Load and concatenate batches
X_train = np.concatenate([np.load(f) for f in X_files])
y_train = np.concatenate([np.load(f) for f in y_files])

print(f"✅ Loaded {len(X_train)} images and {len(y_train)} labels.")

# If needed, save as a single file for future use
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)

print("💾 Saved combined dataset as X_train.npy and y_train.npy.")
