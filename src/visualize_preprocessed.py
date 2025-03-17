import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

# Set paths for preprocessed dataset
PROCESSED_DIR = "dataset/processed/"
X_train_path = os.path.join(PROCESSED_DIR, "X_train.npy")
y_train_path = os.path.join(PROCESSED_DIR, "y_train.npy")

# Load dataset
X_train = np.load(X_train_path, allow_pickle=True)
y_train = np.load(y_train_path)

# Ensure data has the correct shape
if len(X_train.shape) == 4:  # Expected shape (num_samples, height, width, channels)
    X_train = np.squeeze(X_train, axis=-1)  # Remove single channel dimension if present

# Select random samples
num_samples = 5
indices = np.random.choice(X_train.shape[0], num_samples, replace=False)
X_sample = X_train[indices]
y_sample = y_train[indices]

# Create a figure for visualization
fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))

for i in range(num_samples):
    # Original input image
    axes[0, i].imshow(X_sample[i], cmap='gray')
    axes[0, i].set_title(f"Input {i+1}")
    axes[0, i].axis("off")

    # Ground truth label
    axes[1, i].imshow(y_sample[i], cmap='gray')
    axes[1, i].set_title(f"Label {i+1}")
    axes[1, i].axis("off")

    # Overlay input + label
    overlay = 0.5 * X_sample[i] + 0.5 * y_sample[i]  # Blending input and label
    axes[2, i].imshow(overlay, cmap='gray')
    axes[2, i].set_title(f"Overlay {i+1}")
    axes[2, i].axis("off")

plt.suptitle("Preprocessed Data Visualization", fontsize=16)
plt.tight_layout()
plt.show()

# ---- 1️⃣ Pixel Intensity Histogram ----
plt.figure(figsize=(8, 5))
plt.hist(X_sample.flatten(), bins=50, alpha=0.6, label="Input")
plt.hist(y_sample.flatten(), bins=50, alpha=0.6, label="Label")
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ---- 2️⃣ 3D Surface Plot (Elevation) ----
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(X_sample.shape[1])
y = np.arange(X_sample.shape[2])
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, X_sample[0], cmap='viridis')

ax.set_title("3D Surface Plot of Input Data")
plt.show()

# ---- 3️⃣ GIF Animation of Preprocessed Data ----
gif_path = os.path.join(PROCESSED_DIR, "data_animation.gif")

frames = []
for i in range(10):  # Show 10 different samples
    idx = np.random.randint(0, X_train.shape[0])
    frame = np.concatenate([X_train[idx], y_train[idx]], axis=1)  # Side-by-side
    frames.append((frame * 255).astype(np.uint8))

imageio.mimsave(gif_path, frames, duration=0.5)
print(f"✅ GIF saved at: {gif_path}")
