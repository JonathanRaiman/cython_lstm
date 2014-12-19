import numpy as np
from .error import MSE
from .layers.connectible_layer import ConnectibleLayer
from .topology import topological_sort, Node

class Network():
    """
    Create a simple neural network
    """
    def __init__(self, metric = MSE, dtype=np.float32):
        self.dtype = dtype
        self.layers = []
        assert(hasattr(metric, 'error') and hasattr(metric, 'dEdy')), "Metric must implement error and error derivative."
        self.error_metric = metric
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
            
            self._data_layer.connect_to(layer)
            # self._data_layer.children.append(layer)
            # layer.add_backward_layer(self._data_layer)
        if output:
            self._output_layer = layer
        
    def remove_layer(self, layer):
        self.layers.remove(layer)

    def allocate_activation(self, timesteps, streams):
        for layer in self.layers:
            if hasattr(layer, 'allocate_activation'):
                layer.allocate_activation(timesteps, streams)
        
    def activate(self, input):
        """
        Activate takes the input to the network
        and dispatches it forward starting from
        the data layer (the lowest input).

        Note: in the future this activation procedure
        will instead rely on the topological sort of the
        nodes in graph, and multiple inputs will be usable.
        """
        input = np.asarray(input, dtype=self.dtype)
        if input.ndim == 1:
            input = input.reshape(1,-1)
        # deprecated method:
        self.allocate_activation(input.shape[0], input.shape[1])
        # activate first layer
        last_layer = self._data_layer
        out = input
        while last_layer is not None:
            out = last_layer.activate([out])
            last_layer = last_layer._forward_layer
        
        return self._output_layer._activation

    def topsort(self):
        """
        Return the input layers to the network (the
        data layers) and use those as roots for a
        topological sort of the network.
        """
        mapping = Node.to_layer_mapping([self._data_layer])
        roots = [value
            for key, value in mapping.items()
            if isinstance(key, DataLayer)]
        return topological_sort(roots)
        
    def backpropagate(self, target):
        if not target.dtype.kind == 'i':
            target = np.asarray(target, dtype=self.dtype)
            if target.ndim == 1:
                target = target.reshape(1,-1)
        if hasattr(self._output_layer, 'error_activate'):
            self._output_layer.error_activate(target)
        else:
            self.manual_backpropagation(target)


    def manual_backpropagation(self, target):
        last_layer = self._output_layer
        # special error is used here to initialize the backpropagation
        # procedure
        error_signal = self.error_metric.dEdy(last_layer._activation, target)
        while last_layer is not None and last_layer is not self._data_layer:
            # we pass down the current error signal
            error_signal = last_layer.update_grad_input(
                last_layer._backward_layer._activation,
                last_layer._activation,
                error_signal)
            # step down by a layer.
            last_layer = last_layer._backward_layer

    def set_error(self, metric):
        """
        Set the error metric that should
        be used. The choices are:
            * MSE
            * BinaryCrossEntropy
            * TanhBinayCrossEntropy
            * CategoricalCrossEntropy
        """
        assert(hasattr(metric, 'error') and hasattr(metric, 'dEdy')), "Metric must implement error and error derivative."
        self.error_metric = metric
        
    def error(self, target):
        return self.error_metric.error(self._output_layer._activation, target)
        
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
        "output_layer": self._output_layer.activation_function.__doc__ if hasattr(self._output_layer, 'activation_function') else '',
        "input_layer": self._input_layer.activation_function.__doc__ if hasattr(self._input_layer, 'activation_function') else ''
            })

class DataLayer(ConnectibleLayer):
    def __init__(self):
        self.parents = []
        self.children = []
        self._activation = None
        self._forward_layer = None
        
    def activate(self, input):
        self._activation = input[0]
        return self._activation

    def activation(self):
        return self._activation
        
    def backpropagate(self, signal):
        pass