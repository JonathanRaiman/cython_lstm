import pyximport, numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .cython_utils import vector_outer_product

__all__ = ["vector_outer_product"]