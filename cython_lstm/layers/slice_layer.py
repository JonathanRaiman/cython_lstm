from .base_layer import BaseLayer

class SliceLayer(BaseLayer):
    """

    Takes a subset of its input, and passes
    it forward.
    In temporal sequences, this takes the last
    element of a sequence and passes it onwards.

    """

    def __init__(self, index):
        BaseLayer.__init__(self)
        self.index         = index
        self._activation   = None
        self.dimensionless = True

    def forward_propagate(self, x):
        return [layer.forward_propagate(x[self.index]) for layer in self._temporal_forward_layers]

    def clear(self):
        self.step        = 0

    def allocate_activation(self, *args):
        pass

    @property
    def input_size(self):
        """
        The input size is the activation of the previous node.

        This value is currently incorrect for 3d tensors.
        """
        return self._backward_layer.input_size if self._backward_layer is not None else None

    @property
    def output_size(self):
        """
        The output size is the result of a slice on the activation of the previous node.

        This value is currently incorrect for 3d tensors.
        """
        return self._backward_layer.input_size if self._backward_layer is not None else None