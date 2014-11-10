import numpy as np
def create_digit_dataset():
    return (np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0]], dtype=np.float32),
            np.arange(0, 11).astype(np.int32))

def create_xor_dataset():
      return (np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]], dtype=np.float32),
                               np.array([
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1]], dtype=np.float32))

__all__ = ["create_digit_dataset", "create_xor_dataset"]