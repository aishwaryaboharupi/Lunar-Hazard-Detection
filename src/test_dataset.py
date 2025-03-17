import numpy as np

X_train_path = "dataset/processed/X_train.npy"

try:
    X_train = np.load(X_train_path, allow_pickle=True)
    print("Successfully loaded! Shape:", X_train.shape)
except Exception as e:
    print("Error loading dataset:", e)
