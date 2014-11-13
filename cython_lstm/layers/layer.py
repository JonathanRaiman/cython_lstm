"""
Neural Network Layer
--------------------

Missing: Dropout, Language Models, Hierachical Softmax

Note: it appears that Dropout is a weightless layer.
Layer should be generalized to the weighless case.

"""
from ..cython_utils import vector_outer_product, tensor_delta_down_with_output
from ..neuron import Neuron
import numpy as np
REAL = np.float32

def quadratic_form(tensor, x):
    return (np.dot(tensor, x.T) * x.T).sum(axis=1).T

def quadratic_form_gradient(error, x):
    return (vector_outer_product(x, x)[:,:,:, np.newaxis] * error).transpose((3,1,0,2))

class Layer():
    """
    Create a feedforward layer with identity activation
    """
    def __init__(self,
                 input_size = 10,
                 output_size = None,
                 tensor = False,
                 neuron = Neuron,
                 dtype=REAL):
        self.dtype               = dtype
        self.activation_function = neuron.activation_function
        self.error               = neuron.error
        self.dydz                = neuron.dydz
        self.dEdy                = neuron.dEdy
        self._weight_matrix      = None
        self._bias_units         = None
        self.input_size          = input_size
        self.output_size         = output_size
        
        self._forward_layers     = []
        self._backward_layers    = []
        
        self.tensor              = tensor
        
        self._activation         = None
        self._dEdy               = None
        self.dEdz                = None
        
        self._weight_matrix_diff = None
        self._bias_units_diff    = None
        
        if self.tensor:
            self._weight_tensor      = None
            self._weight_tensor_diff = None
        
        if self.input_size is not None and self.output_size is not None:
            self.create_weights()
        
    def activate_forward_layers(self):
        for layer in self._forward_layers:
            layer.activate(self.activation())
        
    def activate(self, input):
        # run net forward using input
        self.forward_propagate(input)
        # transfer activation as input to next layers:
        self.activate_forward_layers()
        
    def backpropagate_dEdy(self):
        """
        Backpropagate error signal to the weights
        and prepare for lower layers to use by getting 
        dEdz.
        """
        
        self.dEdz = np.dot(self._dEdy, self._weight_matrix)
        
        # can be a costlyish operation if requires addition
        # of hidden state vector:
        layer_input = self.layer_input()
        
        # updates to weight matrix are given by outer
        # product of signal with input:
        self._weight_matrix_diff += vector_outer_product(self._dEdy, layer_input).sum(axis=-1)
        
        # updates to bias units are given by signal
        self._bias_units_diff += self._dEdy.T.sum(axis=-1)
        
        if self.tensor:
            # propagate signal backwards through the tensor:
            tensor_delta_down_with_output(self._weight_tensor, self._dEdy, layer_input, self.dEdz)
            # obtain gradient for a tensor:
            self._weight_tensor_diff += quadratic_form_gradient(self._dEdy, layer_input).sum(axis=-1)
        
    def backpropagate(self, signal):
        """
        Get local error responsability using
        the derivative of error with respect
        to output times the derivative of the
        local parameters dy / dz
        """
        # signal backwards is given by taking weight matrix
        # with signal with derivative
        self._dEdy = signal * self.dydz(self._activation)
        
        # given we know the error signal at this stage,
        # constitute the local error responsability dEdz
        # and mark the updates to the weights:
        self.backpropagate_dEdy()
        
        # Send dEdz backwards as new error signal
        for layer in self._backward_layers:
            layer.backpropagate(self.dEdz)
        
    def layer_input(self):
        """
        Input is sum of activations of backward
        layers.
        """
        if len(self._backward_layers) > 1:
            return sum(layer.activation() for layer in self._backward_layers)
        else:
            return self._backward_layers[0].activation()
        
    def activation(self):
        return self._activation
        
    def error_activate(self, target):
        """
        Start the backpropagation using a target
        by getting the initial error responsability
        as dE / dy = y - t
        
        dEdW is then provided for the backward layers
        iteratively:
        dE / dW  = (dy_l / dW) * (...) * (dy_l / dy) * (dE / dy)
        """
        
        # get the error here
        self.backpropagate(self.dEdy(self.activation(),target))
        
    def clear(self):
        """
        Clears the activation and the local
        error responsibility for this layer
        """
        self._activation       = None
        self._dEdy             = None
        self.dEdz              = None
        self._weight_matrix_diff.fill(0)
        self._bias_units_diff.fill(0)
        if self.tensor:
            self._weight_tensor_diff.fill(0)
        
    def reset_weights(self):
        """
        Reset to random weights this
        layer
        """
        self.clear()
        self._weight_matrix += self._random_weight_matrix()
        self._bias_units += self._random_bias_units()
        if self.tensor:
            self._weight_tensor += self._random_weight_tensor()
            
    def _random_weight_tensor(self):
        return (
            (1. / self.input_size) *
            np.random.standard_normal([
                self.output_size,
                self.input_size,
                self.input_size])
        ).astype(self.dtype)
        
    def _random_weight_matrix(self):
        return (
            (1. / self.input_size) *
            np.random.standard_normal([
                self.output_size,
                self.input_size])
        ).astype(self.dtype)
    
    def _random_bias_units(self):
        return (
            (1. / self.input_size) *
            np.random.standard_normal(self.output_size)
        ).astype(self.dtype)
        
    def create_weights(self):
        """
        Randomly initialize the weights for this layer
        with gaussian noise with std 1 / input size
        """
        self._weight_matrix = self._random_weight_matrix()
        self._weight_matrix_diff = np.zeros_like(self._weight_matrix)
        
        self._bias_units = self._random_bias_units()
        self._bias_units_diff = np.zeros_like(self._bias_units)
        
        
        self.params    = [self._weight_matrix, self._bias_units]
        self.gradients = [self._weight_matrix_diff, self._bias_units_diff]
        
        if self.tensor:
            self._weight_tensor = self._random_weight_tensor()
            self._weight_tensor_diff = np.zeros_like(self._weight_tensor)
            self.params.append(self._weight_tensor)
            self.gradients.append(self._weight_tensor_diff)

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
        if self.tensor:
            self._activation = self.activation_function( quadratic_form(self._weight_tensor, input) + np.dot(input, self._weight_matrix.T) + self._bias_units)
        else:
            self._activation = self.activation_function( np.dot(input, self._weight_matrix.T) + self._bias_units)
        return self._activation
    
    def _connect_layer(self, layer):
        """
        Adds the layer to the forward and
        backward lists of layers to connect
        the graph.
        """
        self._forward_layers.append(layer)
        layer.add_backward_layer(self)
        
    def add_backward_layer(self, layer):
        """
        Connect a layer to the antecedents
        of this layer in the graph.
        """
        self._backward_layers.append(layer)
        
    def remove_backward_layer(self, layer):
        self._backward_layers.remove(layer)
        
    def connect_to(self, layer, trust = False):
        """
        Connect two layers together.
        """
        if trust:
            self._connect_layer(layer)
        else:
            if self.output_size == None:
                self.output_size = layer.input_size
                self.create_weights()
                self._connect_layer(layer)
            elif layer.input_size == self.output_size:
                self._connect_layer(layer)
            else:
                raise BaseException("Current layer's output size does not match input size for next layer")
            
    def __repr__(self):
        return "<" + self.__class__.__name__ + " " + str({"activation": self.activation_function.__doc__, "input_size": self.input_size, "output_size": self.output_size})+">"

