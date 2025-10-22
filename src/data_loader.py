from pathlib import Path

import numpy as np

missing_entry = 1e99


def load_dataset(train_path, label_path=None, test_path=None):
    """
    Loads training/test datasets and changes the missing values (1e99) to be Nan instead.

    Parameters
    -----------
    train_path: a string
        this is a path to the training data file.
    label_path: string or none
        this is a path to the training labels file.
    test_path: string or none
        this is a path to the test data file.

    Returns
    --------
    dict type
        A dictionary with the keys X_train, Y_train, and X_test. 
        (only when they are given as parameters)
    """

    # This takes the given file paths and turns them into Path objects
    train_path, label_path, test_path = map(
        lambda p: Path(p) if p is not None else None,
        (train_path, label_path, test_path),
    )

    # loading the data from the given files and replacing the missing entries with NaN
    def load(path):
        arr = np.loadtxt(path)
        arr = np.where(arr == missing_entry, np.nan, arr)
        return arr

    data = {"X_train": load(train_path)}
    if label_path:
        data["y_train"] = load(label_path).astype(int).ravel()
    if test_path:
        data["X_test"] = load(test_path)
    return data
