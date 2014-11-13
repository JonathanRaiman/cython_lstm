"""
Recurrent Neural Network Layer
------------------------------

Missing: LSTM, Recursive Gated, Language Models, Hierachical Softmax

"""
from .layer import Layer, quadratic_form
import numpy as np
REAL = np.float32

class RecurrentLayer(Layer):
    """
    Recurrent Neural net layer with a linear activation,
    with backpropagation through time implemented
    for an error in the future.
    
    """
    def __init__(self, *args, **kwargs):
        self.step = 0
        self._temporal_backward_layers = []
        self._temporal_forward_layers  = []
        Layer.__init__(self, *args, **kwargs)
        # connect to self

    def connect_to(self, layer, temporal = False, **kwargs):
        if temporal:
            self.connect_through_time(layer)
        else:
            Layer.connect_to(self, layer, **kwargs)

    def connect_through_time(self, layer):
        self._temporal_forward_layers.append(layer)
        layer.add_temporal_backward_layer(self)

    def add_temporal_backward_layer(self, layer):
        self._temporal_backward_layers.append(layer)
            
    def activation(self):
        return self._activation[self.step]

    def allocate_activation(self, input):
        print(self.__class__.__name__ + " is allocating memory for its activations.")
        timesteps = input.shape[0]
        self._activation = np.zeros([timesteps, input.shape[1], self.output_size] , dtype=self.dtype)

    def allocate_temporal_forward_layers(self, input):
        for layer in self._temporal_forward_layers:
            layer.allocate_activation(input)

    def activate_timestep(self, input):
        if self.step < input.shape[0]:
            self.forward_propagate(input[self.step])
            self.step += 1
            for layer in self._temporal_forward_layers:
                layer.activate_timestep(self._activation)
            
    def activate(self, input):
        """
        Activate a recurrent neural layer
        by advancing a step for each of the
        dimensions in the first axis of the
        data.
        """
        # run net forward using input
        self.allocate_activation(input)
        self.allocate_temporal_forward_layers(input)

        self.step = 0
        self.activate_timestep(input)

        # transfer activation as input to next layers:
        self.activate_forward_layers()
            
    def layer_input(self):
        """
        Input is sum of activations of backward
        layers.
        
        Activation dimensions are:
           1. time step
           2. which stream (for batch training)
           3. dimensions of observation
        """
        return self.prepare_timestep_input(self._backward_layers[0]._activation[self.step])
        
    def prepare_timestep_input(self, observation):
        """
        Concatenate previous hidden state with observable
        """
        
        if self.step == 0:
            # repeat initial hidden state as many times as the data is observed
            hidden = np.tile(self._initial_hidden_state, (observation.shape[0],1))
        else:
            # previous hidden state is concatenated with the current observation:
            hidden = self._temporal_backward_layers[0]._activation[self.step-1]
          
        return np.concatenate([
                hidden, # repeated hidden state
                observation # timestep data observation
                ], axis=-1)
    
    def clear(self):
        """
        Clears the activation and the local
        error responsibility for this layer
        """
        self.step              = 0
        self._activation       = None
        self._dEdy             = None
        self.dEdz              = None
        self._weight_matrix_diff.fill(0)
        self._bias_units_diff.fill(0)
        
    def create_weights(self):
        """
        Randomly initialize the weights for this recurrent layer
        with gaussian noise with std 1 / input size.
        Weights have size corresponding to output:
            visible + hidden => hidden
            
        """
        self._weight_matrix = self._random_weight_matrix()
        self._weight_matrix_diff = np.zeros_like(self._weight_matrix)
        
        self._bias_units = self._random_bias_units()
        self._bias_units_diff = np.zeros_like(self._bias_units)
        
        self._initial_hidden_state = self._zero_initial_state()
        self._initial_hidden_state_diff = np.zeros_like(self._initial_hidden_state)
        
        self.params    = [self._weight_matrix, self._bias_units, self._initial_hidden_state]
        self.gradients = [self._weight_matrix_diff, self._bias_units_diff, self._initial_hidden_state_diff]
        
    def _zero_initial_state(self):
        return np.zeros(self.output_size, dtype=self.dtype)
        
    def reset_weights(self):
        """
        Reset to random weights this
        layer
        """
        self.clear()
        self._weight_matrix += self._random_weight_matrix()
        self._bias_units += self._random_bias_units()
        self._initial_hidden_state.fill(0)
        
    def _random_weight_matrix(self):
        return (1. / (self.input_size + self.output_size) *
            np.random.standard_normal([
                self.output_size,
                self.input_size + self.output_size])
        ).astype(self.dtype)

    def _random_weight_tensor(self):
        return (1. / (self.input_size + self.output_size) *
            np.random.standard_normal([
                self.output_size,
                self.input_size + self.output_size,
                self.input_size + self.output_size])
        ).astype(self.dtype)
    
    def forward_propagate(self, input):
        """
        use the weights and the activation function
        to react to the input
        
        TODO: use the `out' parameter of numpy dot to
        gain speed on memory allocation.
        
        Inputs
        ------
        
        inputs ndarray : the input data
        
        Outputs
        -------
        
        activation ndarray : the activation for this input
        
        """
        x = self.prepare_timestep_input(input)
        if self.tensor:
            self._activation[self.step] = self.activation_function(
                quadratic_form(self._weight_tensor, x) +
                np.dot(x, self._weight_matrix.T) +
                self._bias_units )
        else:
            self._activation[self.step] = self.activation_function(
                np.dot(x, self._weight_matrix.T) +
                self._bias_units )
        return self._activation[self.step]

    def backpropagate(self, signal):
        """
        Get local error responsability using
        the derivative of error with respect
        to output times the derivative of the
        local parameters dy / dz
        """
        if self.step == -1:
            # signal backwards is transmitted to initial hidden state:
            # only use top part of dE/dy to get hidden state error:
            self._initial_hidden_state_diff += signal[:, 0:self.output_size].T.sum(axis=-1)
            
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