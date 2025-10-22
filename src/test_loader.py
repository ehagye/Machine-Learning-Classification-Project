import numpy as np

from src.data_loader import load_dataset

data = load_dataset("data/TrainData1.txt", "data/TrainLabel1.txt", "data/TestData1.txt")

print(data["X_train"].shape)
print(data["y_train"][:5])
print(np.isnan(data["X_train"]).sum())
