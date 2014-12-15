from .base_layer import BaseLayer
import numpy as np
from ..cython_utils import vector_outer_product, tensor_delta_down_with_output

def quadratic_form(tensor, x):
    return (np.dot(tensor, x.T) * x.T).sum(axis=1).T

def quadratic_form_gradient(error, x):
    return (vector_outer_product(x, x)[:,:,:, np.newaxis] * error).transpose((3,1,0,2))

class LinearLayer(BaseLayer):
    """
    A layer that takes an input and performs
    an affine, or tensor based, transformation
    of its inputs (e.g. using a matrix, a bias
    vector, and optionally a tensor).
    This is a feedforward layer in the traditional
    sense.
    No activation function is used here.
    """
    def __init__(self, input_size, output_size, tensor = False):
        """
        Initialize a layer that projects its input into another
        dimension.

        Inputs
        ------

        input_size  int                : the size of the input
                                         dimensions
        output_size int                : the size of the output
                                         dimensions
        tensor      boolean (optional) : whether to use a bilinear
                                         form in the projection of
                                         the input.

        Note: You may also opt to have a tensor also perform
        a transformation on the input. However this adds as many
        matrices as there are output dimensions, so this can
        become a very costly operation quickly.

        """
        BaseLayer.__init__(self)
        self.input_size  = input_size
        self.output_size = output_size
        self.tensor      = tensor
        self.create_weights()
        
    def random_weight_matrix(self):
        return ( (1. / self.input_size) * np.random.standard_normal([ self.output_size, self.input_size]) ).astype(self.dtype)
    
    def random_bias_units(self):
        return ( (1. / self.input_size) * np.random.standard_normal( self.output_size) ).astype(self.dtype)

    def random_weight_tensor(self):
        return ( (1. / self.input_size) * np.random.standard_normal([ self.output_size, self.input_size, self.input_size]) ).astype(self.dtype)
        
    def create_weights(self):
        """
        Randomly initialize the weights for this layer
        with gaussian noise with std 1 / input size
        """
        self.weight_matrix = self.random_weight_matrix()
        self.weight_matrix_diff = np.zeros_like(self.weight_matrix)
        
        self.bias_units = self.random_bias_units()
        self.bias_units_diff = np.zeros_like(self.bias_units)
        
        
        self.params    = [self.weight_matrix, self.bias_units]
        self.gradients = [self.weight_matrix_diff, self.bias_units_diff]
        
        if self.tensor:
            self.weight_tensor = self.random_weight_tensor()
            self.weight_tensor_diff = np.zeros_like(self.weight_tensor)
            self.params.append(self.weight_tensor)
            self.gradients.append(self.weight_tensor_diff)

    def reset_weights(self):
        """
        Reset to random weights this layer.
        """
        self.clear()
        self.weight_matrix.fill(0)
        self.bias_vector.fill(0)
        self.weight_matrix += self.random_weight_matrix()
        self.bias_vector   += self.random_bias_units()
        if self.tensor:
            self.weight_tensor.fill(0)
            self.weight_tensor += self.random_weight_tensor()

    def activate(self, x, out=None):
        """
        Projects the input into the output dimension.

        Inputs
        ------

        x list<ndarray> : the input to this layer.

        Outputs
        -------
        
        activation ndarray : the activation for this input
        
        """
        if out is None:
            if self.tensor:
                self._activation = quadratic_form(self.weight_tensor, x[0]) + np.dot(x[0], self.weight_matrix.T) + self.bias_units
            else:
                self._activation = np.dot(x[0], self.weight_matrix.T) + self.bias_units
            return self._activation
        else:
            if self.tensor:
                np.dot(x[0], self.weight_matrix.T, out=out)
                out += self.bias_units
                out += quadratic_form(self.weight_tensor, x[0])
                return out
            else:
                np.dot(x[0], self.weight_matrix.T, out=out)
                out += self.bias_units
                return out

    def update_grad_input(self, input, output, grad_output):
        """
        Here we use the input and the output of this layer to
        get the gradient of this layer. Usually the output of
        a linear layer is not needed, but because a second
        order method (a bilinear form, using a tensor) is
        possible, the output of this layer can be useful for
        backpropagation.
        """
        if self.gradinput is None:
            self.gradinput = np.zeros_like(input)

        self.gradinput += np.dot(grad_output, self.weight_matrix)
        
        # updates to weight matrix are given by outer
        # product of signal with input:
        self.weight_matrix_diff += vector_outer_product(grad_output, input).sum(axis=-1)
        
        # updates to bias units are given by signal
        self.bias_units_diff += grad_output.T.sum(axis=-1)
        
        if self.tensor:
            # propagate signal backwards through the tensor:
            tensor_delta_down_with_output(self.weight_tensor, grad_output, input, self.gradinput)
            # obtain gradient for a tensor:
            self.weight_tensor_diff += quadratic_form_gradient(grad_output, input).sum(axis=-1)

        return self.gradinput