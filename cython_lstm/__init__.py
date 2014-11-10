"""
Simple Package for creating artificial
neural networks including recurrent,
recursive, and LSTM neural networks.

"""
import pyximport, numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .cython_utils import vector_outer_product, tensor_delta_down, tensor_delta_down_with_output

__all__ = ["vector_outer_product", "tensor_delta_down", "tensor_delta_down_with_output"]