from .base_layer import BaseLayer
import numpy as np

class SliceLayer(BaseLayer):
    """
    Takes a subset of its input, and passes
    it forward.
    """

    def __init__(self, index):
        BaseLayer.__init__(self)
        if not hasattr(index, '__len__'):
            index = (index,)
        self.index = index
        self.gradients = []

    def select_slice(self, input):
        if type(input) is list:
            if len(self.index) > 1:
                return input[self.index[0]][self.index[1:]]
            else:
                return input[self.index[0]]
        else:
            return input[self.index]

    def activate(self, x, out=None):
        """
        Activation for a slice layer is dead simple:

        > y = x[index]

        """
        input = self.select_slice(x[0])

        if out is None:
            # no copy is performed here
            # thankfully
            self._activation = input
            return self._activation
        else:
            out[:] = input
            return out

    def update_grad_input(self, input, output, grad_output):
        """
        Gradient for a slice is simply zeros for all non
        sliced dimensions, and pass the grad_output inside
        the sliced piece.

        > grad = np.zeros_like(input)
        > grad[index] = ParentGradient

        """
        if self.gradinput is None:
            if type(input) is list:
                self.gradinput = [np.zeros_like(piece) for piece in input]
            else:
                self.gradinput = np.zeros_like(input)
        self.select_slice(self.gradinput)[:] = grad_output
        return self.gradinput