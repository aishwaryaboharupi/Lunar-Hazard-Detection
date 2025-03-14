import os
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split

# Define paths
DATASET_DIR = "dataset/"
PROCESSED_DIR = "dataset/processed/"
FILE_PATH = os.path.join(DATASET_DIR, "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif")

# Create processed directory if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Tile size (Adjust if needed)
TILE_SIZE = 256  # Increase from 128 to 256 to reduce number of tiles
BATCH_SIZE = 5000  # Process 5000 tiles at a time

# Check if file exists
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"❌ ERROR: File not found - {FILE_PATH}")
else:
    print(f"✅ File found: {FILE_PATH}")

def load_raster_in_batches(file_path, tile_size, batch_size):
    """Load raster data in batches of tiles."""
    images = []
    
    with rasterio.open(file_path) as dataset:
        height, width = dataset.height, dataset.width
        print(f"📏 Raster dimensions: {width}x{height}")

        count = 0
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                # Read tile
                window = Window(col, row, tile_size, tile_size)
                tile = dataset.read(1, window=window)  # Read first band
                
                if tile.shape == (tile_size, tile_size):  # Ensure valid shape
                    images.append(tile)
                    count += 1

                # Save batch every BATCH_SIZE tiles
                if count % batch_size == 0:
                    save_batch(images, count)
                    images = []  # Reset images list
    
    return images

def save_batch(images, batch_num):
    """Save batch of images to disk."""
    images = np.array(images)
    images = images.astype(np.float32)  # Convert to float to avoid overflow
    images -= np.min(images)  # Ensure min value is 0
    images /= np.max(images)  # Normalize between 0 and 1

    labels = np.zeros(len(images))  # Placeholder labels
    
    print(f"💾 Saving batch {batch_num} with {len(images)} images...")
    np.save(os.path.join(PROCESSED_DIR, f"X_batch_{batch_num}.npy"), images)
    np.save(os.path.join(PROCESSED_DIR, f"y_batch_{batch_num}.npy"), labels)

def preprocess_and_save():
    """Load, preprocess, and save dataset in batches."""
    print("🔄 Loading raster tiles in batches...")
    load_raster_in_batches(FILE_PATH, TILE_SIZE, BATCH_SIZE)
    print(f"✅ Preprocessing complete! Data saved in {PROCESSED_DIR}")

if __name__ == "__main__":
    preprocess_and_save()
