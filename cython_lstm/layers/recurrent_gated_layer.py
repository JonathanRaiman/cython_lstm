from .layer import Layer
from .temporal_layer import TemporalLayer
import numpy as np
REAL = np.float32

class RecurrentGatedLayer(TemporalLayer):
    """
    RecurrentGatedLayer
    -------------------

    Simple layer that outputs a scalar for each
    input stream that "gates" the input. This
    is a memory-less process so it has no hidden
    states.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize recurrent gated layer knowing that it makes
        no modifications to the input except gating it, so
        it's effective output size is its input size, and its
        weight matrix is input dimension by 1 .
        """
        kwargs["output_size"] = 1
        TemporalLayer.__init__(self, *args, **kwargs)