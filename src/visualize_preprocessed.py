import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Path to processed dataset
PROCESSED_DIR = "dataset/processed/"

# Get a few sample files
X_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "X_batch_*.npy")))
y_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "y_batch_*.npy")))

if not X_files or not y_files:
    print("❌ No processed data found! Make sure preprocessing is complete.")
    exit()

# Load a sample batch
X_sample = np.load(X_files[0])
y_sample = np.load(y_files[0])

# Check if labels are single values or images
if len(y_sample.shape) == 1:  # If it's a 1D array, it's likely categorical labels
    label_type = "categorical"
else:
    label_type = "image"

# Select a few random images to visualize
num_samples = min(5, len(X_sample))  # Show up to 5 samples
fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

for i in range(num_samples):
    axes[0, i].imshow(X_sample[i], cmap='gray')
    axes[0, i].set_title(f"Image {i+1}")
    axes[0, i].axis("off")

    if label_type == "image":
        axes[1, i].imshow(y_sample[i], cmap='gray')
    else:
        axes[1, i].text(0.5, 0.5, str(y_sample[i]), fontsize=12, ha="center", va="center")
        axes[1, i].set_facecolor("lightgray")

    axes[1, i].set_title(f"Label {i+1}")
    axes[1, i].axis("off")

plt.suptitle("Preprocessed Data Visualization")
plt.tight_layout()
plt.show()
