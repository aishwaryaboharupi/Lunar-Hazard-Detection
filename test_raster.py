import rasterio

file_path = "dataset/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

try:
    with rasterio.open(file_path) as dataset:
        print("✅ Successfully opened the file!")
        print(f"Width: {dataset.width}, Height: {dataset.height}")
        print(f"Number of bands: {dataset.count}")
except Exception as e:
    print(f"❌ ERROR: {e}")
