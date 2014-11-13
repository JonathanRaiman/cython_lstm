from .recurrent_layer import RecurrentLayer, REAL


class RecurrentAveragingLayer(RecurrentLayer):

    def __init__(self, a_layer, bc_layer, dtype = REAL):
        """
        Strange layer used for multi stage recurrence.
        """
        self.step = 0
        self._temporal_backward_layers = []
        self._temporal_forward_layers  = []

        self.params           = []
        self.gradients        = []
        self._backward_layers = []
        self._forward_layers  = []
        self.input_size       = bc_layer.output_size
        self.output_size      = self.input_size

        # order these are added matters (unfortunately)
        a_layer.connect_to(self, temporal = True)
        bc_layer.connect_to(self, temporal = True)
        self._a_layer         = a_layer
        self._bc_layer        = bc_layer
        self._activation      = None
        self.tensor           = False
        self.dtype            = dtype

    def prepare_timestep_input(self, input):
        return input

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
        past = self._bc_layer._activation[t-1] if t > 0 else self._bc_layer._initial_hidden_state
        
        self._activation[t] = (
                self._a_layer._activation[t] * self._bc_layer._activation[t] +
            (1-self._a_layer._activation[t]) * past)
        return self._activation[t]

    def error_activate(self, target):
        raise NotImplementedError("Cannot error activate multiple timesteps using this layer.")

    def __repr__(self):
        return "<" + self.__class__.__name__ + " " + str({"activation": "a * b + (1 - a) * c", "input_size": "%d + %d" % (self._a_layer.output_size, self._bc_layer.output_size), "output_size": self._bc_layer.output_size})+">"
