from .layer import Layer
from .recurrent_layer import RecurrentLayer
import numpy as np
REAL = np.float32

class RecurrentGatedLayer(RecurrentLayer):
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
        RecurrentLayer.__init__(self, *args, **kwargs)

    def _random_weight_matrix(self):
        return Layer._random_weight_matrix(self)

    def _random_weight_tensor(self):
        return Layer._random_weight_tensor(self)

    def prepare_timestep_input(self, input):
        return input

    def backpropagate(self, signal):
        """
        Get local error responsability using
        the derivative of error with respect
        to output times the derivative of the
        local parameters dy / dz
        """
        if self.step == -1:
            # No initial hidden state here, so we can't backprop to it.
            # reset step:
            self.step = 0
            
            for layer in self._backward_layers:
                layer.backpropagate(self.dEdz)
        else:
            # signal backwards is given by taking weight matrix
            # with signal with derivative
            # take beginning part since remainder is attributable
            # to observation
            self._dEdy = signal[:, 0:self.output_size] * self.dydz(self._activation[self.step])
            
            # given we know the error signal at this stage,
            # constitute the local error responsability dEdz
            # and mark the updates to the weights:
            self.backpropagate_dEdy()
            
            self.step -=1
            
            return self.backpropagate(self.dEdz)

    def layer_input(self):
        """
        Input to layer is identity of activation
        of inout layers.
        
        Activation dimensions are:
           1. time step
           2. which stream (for batch training)
           3. dimensions of observation
        """
        return self._backward_layers[0]._activation[self.step]