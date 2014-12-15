class BaseLayer(object):

    def __init__(self):
        self._forward_layers  = []
        self._backward_layer  = None

    def allocate_activation(self, timesteps, streams):
        """
        Each layer must handle temporal memory allocation
        based on its needs. e.g. certain layers are memory
        less and they do not need to store their
        activations to compute a gradient, while others
        need to, like a Dropout layer that masks the input
        in a unique way each time.

        """
        raise NotImplementedError()

    def _connect_layer(self, layer):
        """
        Adds the layer to the forward and
        backward lists of layers to connect
        the graph.
        """
        self._forward_layers.append(layer)
        layer.add_backward_layer(self)
        
    def activate_forward_layers(self):
        for layer in self._forward_layers:
            layer.activate(self.activation())

    def add_backward_layer(self, layer):
        """
        Connect a layer to the antecedents
        of this layer in the graph.
        """
        if self._backward_layer is not None:
            self._backward_layer.remove_forward_layer(self)
        self._backward_layer = layer

    def remove_forward_layer(self, layer):
        self._forward_layers.remove(layer)
        
    def remove_backward_layer(self, layer):
        if self._backward_layer is layer:
            self._backward_layer = None

    def connect_to(self, layer, temporal = False, trust = False):
        """
        Connect two layers together

        """
        if temporal:
            self.connect_through_time(layer)
        else:
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

    def connect_through_time(self, layer):
        self._temporal_forward_layers.append(layer)
        layer.add_backward_layer(self)

    def __repr__(self):
        return "<" + self.__class__.__name__ + " " + str({"activation": self.activation_function.__doc__ if hasattr(self, 'activation_function') else '', "input_size": self.input_size, "output_size": self.output_size})+">"