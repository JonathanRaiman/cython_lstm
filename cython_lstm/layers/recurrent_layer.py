"""
Recurrent Neural Network Layer
------------------------------

Missing: LSTM, Recursive Gated, Language Models, Hierachical Softmax

"""
from .temporal_layer import TemporalLayer, quadratic_form
from .layer import Layer
import numpy as np
REAL = np.float32

class RecurrentLayer(TemporalLayer):
    """
    Recurrent Neural net layer with a linear activation,
    with backpropagation through time implemented
    for an error in the future.
    
    """

    def activate_timestep(self, input):
        if self.step < input.shape[0]:
            self.forward_propagate(input[self.step])
            self.step += 1
            for layer in self._temporal_forward_layers:
                #print("(%d) %s => %s" % (self.step, self.__class__.__name__, layer.__class__.__name__))
                layer.activate_timestep(self._activation)
            
    def layer_input(self):
        """
        Input is sum of activations of backward
        layers.
        
        Activation dimensions are:
           1. time step
           2. which stream (for batch training)
           3. dimensions of observation
        """
        # what was given as an input:
        observation = self._temporal_backward_layers[0]._activation[self.step]

        if self.step == 0:
            # repeat initial hidden state as many times as the data is observed
            hidden = np.tile(self._initial_hidden_state, (observation.shape[0], 1))
        else:
            # previous hidden state is concatenated with the current observation:
            hidden = self._activation[self.step-1]
        return np.concatenate([
                hidden, # repeated hidden state
                observation # timestep data observation
                ], axis=1)
        
    def create_weights(self):
        """
        Randomly initialize the weights for this recurrent layer
        with gaussian noise with std 1 / input size.
        Weights have size corresponding to output:
            visible + hidden => hidden
            
        """
        Layer.create_weights(self)
        
        self._initial_hidden_state = self._zero_initial_state()
        self._initial_hidden_state_diff = np.zeros_like(self._initial_hidden_state)
        
        self.params.append(self._initial_hidden_state)
        self.gradients.append(self._initial_hidden_state_diff)
        
    def reset_weights(self):
        """
        Reset to random weights this
        layer
        """
        Layer.reset_weights()
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
        t = self.step
        hidden = self._activation[t-1] if t > 0 else np.tile(self._initial_hidden_state, (input.shape[0], 1))
        x = np.concatenate([hidden, input], axis=1)
        if self.tensor:
            self._activation[t] = self.activation_function(
                quadratic_form(self._weight_tensor, x) +
                np.dot(self._weight_matrix, x.T).T +
                self._bias_units )
        else:
            self._activation[t] = self.activation_function(
                np.dot(self._weight_matrix, x.T).T +
                self._bias_units )
        return self._activation[t]

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