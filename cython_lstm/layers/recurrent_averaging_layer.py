from .recurrent_layer import RecurrentLayer, REAL
import numpy as np

class RecurrentAveragingLayer(RecurrentLayer):

    def __init__(self, a_layer, bc_layer, dtype = REAL):
        """
        Strange layer used for multi stage recurrence.
        """
        self.step                     = 0
        self._temporal_forward_layers = []
        
        self.dimensionless            = False
        self._backward_layer          = None
        self._forward_layers          = []
        self.input_size               = bc_layer.output_size
        self.output_size              = bc_layer.output_size
        
        self._a_layer                 = a_layer
        self._bc_layer                = bc_layer
        self._activation              = None
        self.tensor                   = False
        self.dtype                    = dtype
        self.create_weights()

    def create_weights(self):
        self.params           = []
        self.gradients        = []

        self._initial_hidden_state = self._zero_initial_state()
        self._initial_hidden_state_diff = np.zeros_like(self._initial_hidden_state)
        
        self.params.append(self._initial_hidden_state)
        self.gradients.append(self._initial_hidden_state_diff)

    def backpropagate_one_step(self, signal):
        """
        Get local error responsability using
        the derivative of error with respect
        to output times the derivative of the
        local parameters dy / dz

        Derivative for a * b + (1-a) * c
        for a is : b - c
        for b is : a
        for c is : 1-a

        """
        t         = self.step
        self.dEda = signal * (self._bc_layer._activation[t] - self._bc_layer._activation[t - 1])
        self.dEdb = signal * self._a_layer._activation[t]
        self.dEdc = signal * (1. - self._a_layer._activation[t])
        self.step -=1

    def backpropagate(self, signal):
        raise NotImplementedError("Cannot backpropagate multiple timesteps using this layer.")

    def forward_propagate(self, input):
        """
        Average b and c using a:
        out = a * b + (1 - a) * c
        """
        t = self.step
        hidden = self._activation[t-1] if t > 0 else np.tile(self._initial_hidden_state, (input.shape[0], 1))
        self._activation[t] = (
                self._a_layer._activation[t] * input +
            (1-self._a_layer._activation[t]) * hidden)
        return self._activation[t]

    def error_activate(self, target):
        raise NotImplementedError("Cannot error activate multiple timesteps using this layer.")

    def __repr__(self):
        return "<" + self.__class__.__name__ + " " + str({"activation": "a * b + (1 - a) * c", "input_size": "%d + %d" % (self._a_layer.output_size, self._bc_layer.output_size), "output_size": self._bc_layer.output_size})+">"
