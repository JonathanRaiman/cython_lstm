"""
Neural Network Temporal Layer
-----------------------------

"""
"""
Recurrent Neural Network Layer
------------------------------

Missing: LSTM, Recursive Gated, Language Models, Hierachical Softmax

"""
from .layer import Layer, quadratic_form
import numpy as np
REAL = np.float32

class TemporalLayer(Layer):
    """
    Recurrent Neural net layer with a linear activation,
    with backpropagation through time implemented
    for an error in the future, and no hidden activation.
    
    """

    def connect_to(self, layer, temporal = False, **kwargs):
        if temporal:
            self.connect_through_time(layer)
        else:
            Layer.connect_to(self, layer, **kwargs)

    def connect_through_time(self, layer):
        self._temporal_forward_layers.append(layer)
        layer.add_backward_layer(self)
            
    def activation(self):
        return self._activation[self.step]

    def allocate_activation(self, timesteps, streams):
        #print(self.__class__.__name__ + " is allocating memory for its activations.")
        self._activation = np.zeros([timesteps, streams, self.output_size] , dtype=self.dtype)

    def activate_timestep(self, input):
        if self.step < input.shape[0]:
            self.forward_propagate(input[self.step])
            self.step += 1
            for layer in self._temporal_forward_layers:
                #print("(%d) %s => %s" % (self.step, self.__class__.__name__, layer.__class__.__name__))
                layer.activate_timestep(self._activation)

    def recursive_activate_timestep(self, input):
        self.activate_timestep(input)
        if self.step < input.shape[0]:
            self.recursive_activate_timestep(input)
            
    def activate(self, input):
        """
        Activate a recurrent neural layer
        by advancing a step for each of the
        dimensions in the first axis of the
        data.

        """

        self.step = 0
        self.recursive_activate_timestep(input)
        self.step -= 1

        # transfer activation as input to next layers:
        print("Activating forward layers from %s" % (self.__class__.__name__,))
        self.activate_forward_layers()
            
    def layer_input(self):
        """
        Input is sum of activations of backward
        layers.

        """
        return self._backward_layer._activation[self.step]
    
    def forward_propagate(self, x):
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
            self._activation[self.step] = self.activation_function(
                quadratic_form(self._weight_tensor, x) +
                np.dot(self._weight_matrix, x.T).T +
                self._bias_units )
        else:
            self._activation[self.step] = self.activation_function(
                np.dot(self._weight_matrix, x.T).T + self._bias_units )
        return self._activation[self.step]

    def backpropagate(self, signal):
        """
        Get local error responsability using
        the derivative of error with respect
        to output times the derivative of the
        local parameters dy / dz
        """
        if self.step == -1:
            # reset step:
            self.step = 0
            
            self._backward_layer.backpropagate(self.dEdz)
        else:
            # signal backwards is given by taking weight matrix
            # with signal with derivative
            # take beginning part since remainder is attributable
            # to observation
            self._dEdy = signal[:,:] * self.dydz(self._activation[self.step])
            
            # given we know the error signal at this stage,
            # constitute the local error responsability dEdz
            # and mark the updates to the weights:
            self.backpropagate_dEdy()
            
            self.step -=1
            
            return self.backpropagate(self.dEdz)