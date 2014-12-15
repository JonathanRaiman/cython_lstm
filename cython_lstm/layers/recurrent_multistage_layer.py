from .recurrent_layer import RecurrentLayer

import numpy as np
REAL = np.float32

class RecurrentMultiStageLayer(RecurrentLayer):
    """
    Wrapper around multiple temporal layers.
    Handles the flow for forward and backward
    propagation of signal.
    
    """

    def __init__(self, layers, dtype=REAL):
        
        self.dtype            = dtype
        self._dEdy            = None
        self.dEdz             = None
        self.tensor           = False
        self._forward_layers  = []
        self._backward_layer  = None
        
        self._internal_layers = layers
        self._output_layer    = self._internal_layers[-1]
        
        # collect these using the layers:
        self.params           = []
        self.gradients        = []

    def create_weights(self):
        pass

    def reset_weights(self):
        for layer in self._internal_layers:
            self.reset_weights()

    def clear(self):
        """

        Clears the activation and the local
        error responsibility for this layer,
        and clears the internal layers too.

        """
        self.step              = 0
        self._dEdy             = None
        self.dEdz              = None
        for layer in self._internal_layers:
            layer.clear()

    def activation(self):
        return self._output_layer._activation[self.step]

    def activate_forward_layers(self):
        """
        Pass the last timestep activation forward
        """
        for layer in self._forward_layers:
            layer.activate(self.activation())

    def layer_input(self):
        return self._backward_layer._activation[self.step]

    def error_activate(self, target):
        raise NotImplementedError("Not implemented")

    def backpropagate_one_step(self, signal):
        pass

    def backpropagate(self, signal):
        # todo : generalize so that all layers use backprop one step
        # within backpropagate
        if self.step == -1:
            self.step = 0
            self._backward_layer.backpropagate(self.dEdz)
        else:
            for layer in reversed(self._internal_layers):
                layer.backpropagate_one_step(signal)
            self.step -= 1
            self.backpropagate(self.dEdz)

    def allocate_activation(self, timesteps, streams):
        for layer in self._internal_layers:
            layer.allocate_activation(timesteps, streams)

    def forward_propagate(self, input):
        self._internal_layers[0].activate(input)
