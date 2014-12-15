from .base_layer import BaseLayer
import numpy as np

class ActivationLayer(BaseLayer):
    """
    An activation layer takes a neuron as input
    and using this it defines the gradient
    with respect to its input.
    """
    def __init__(self, neuron):
        BaseLayer.__init__(self)
        self.activation_function = neuron.activation_function
        self.error               = neuron.error
        self.dydz                = neuron.dydz
        self.dEdy                = neuron.dEdy
        self.gradients           = []

    def activate(self, x, out=None):
        """
        Operations of an activation layer are elementwise,
        and thus the out shape is defined from the in shape,
        x.
        """
        if out is None:
            self._activation = self.activation_function(x[0],out)
            return self._activation
        else:
            return self.activation_function(x[0],out)

    def update_grad_input(self, input, output, grad_output):
        """
        here we take the neuron's update method for the gradient
        which is usually a function of its output.
        this is a form of code smell, but for a neural network
        module this is habitutal
        """
        if self.gradinput is None:
            self.gradinput = np.zeros_like(input)
        self.gradinput += grad_output * self.dydz(output)
        return self.gradinput