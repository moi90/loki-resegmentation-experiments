import numpy as np
import gzip


def save_gz(fn: str, array: np.ndarray):
    with gzip.GzipFile(fn, "w") as f:
        np.save(f, array)


def load_gz(fn: str, **kwargs) -> np.ndarray:
    with gzip.GzipFile(fn, "r") as f:
        return np.load(f, **kwargs)
