import numpy as np
REAL = np.float32

class Network():
    """
    Create a simple neural network
    """
    def __init__(self, dtype=REAL):
        self.dtype = dtype
        self.layers = []
        self._output_layer = None
        self._input_layer = None
        self._data_layer = DataLayer()
        
    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()
        
    def add_layer(self, layer, input=False, output = False):
        self.layers.append(layer)
        if input:
            self._input_layer = layer
            # connect the data layer to the first network
            # layer:
            if self._data_layer._forward_layer is not None:
                self._data_layer._forward_layer.remove_backward_layer(self._data_layer)
            
            self._data_layer._forward_layer = layer
            layer.add_backward_layer(self._data_layer)
        if output:
            self._output_layer = layer
        
    def remove_layer(self, layer):
        self.layers.remove(layer)

    def allocate_activation(self, timesteps, streams):
        for layer in self.layers:
            layer.allocate_activation(timesteps, streams)
        
    def activate(self, input):
        input = np.asarray(input, dtype=self.dtype)
        if input.ndim == 1:
            input = input.reshape(1,-1)
        self.allocate_activation(input.shape[0], input.shape[1])
        self._data_layer.activate(input)
        return self._output_layer._activation
        
    def backpropagate(self, target):
        if not target.dtype.kind == 'i':
            target = np.asarray(target, dtype=self.dtype)
            if target.ndim == 1:
                target = target.reshape(1,-1)
        self._output_layer.error_activate(target)
        
    def error(self, target):
        return self._output_layer.error(self._output_layer.activation(), target)
        
    def clear(self):
        """
        Resets the state of the layers in this
        neural network
        """
        for layer in self.layers:
            layer.clear()
            
    def get_parameters(self):
        """
        Collect the parameters of the net into a list.
        """
        
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.params)
        return parameters
    
    def get_gradients(self):
        """
        Collect the gradients of the net into a list.
        """
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.gradients)
        return gradients
        
    def __repr__(self):
        if len(self.layers) == 1:
            return str({"layer":self.layers[0]})
        else:
            return str({
        "layers": self.layers,
        "output_layer": self._output_layer.activation_function.__doc__,
        "input_layer": self._input_layer.activation_function.__doc__
            })

class DataLayer():
    def __init__(self):
        self._activation = None
        self._forward_layer = None
        
    def activate(self, input):
        self._activation = input
        self._forward_layer.activate(self._activation)
        
    def activation(self):
        return self._activation
        
    def backpropagate(self, signal):
        """
        Backpropagating through data layer
        may be useful for language models,
        but for fixed dataset this is not
        meaningful.
        """
        pass