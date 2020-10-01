import numpy as np
import h5py
from pathlib import Path


def load_predicate_embeddings(file_path, dim=300):
    dim = "d" + str(dim)
    assert dim in ["d300", "d200", "d100", "d50"]
    if not Path(file_path).exists():
        print("file not found: ", file_path)
        exit(-1)

    with h5py.File(file_path, "r") as data:
        embeddings = data[dim][()]

    return embeddings

