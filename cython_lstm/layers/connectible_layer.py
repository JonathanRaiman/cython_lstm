class ConnectibleLayer(object):
    """
    This class only handles layer
    interconnections. Only knowledge
    about a supposed graph is assumed here.

    Note: a layer can have multiple output
    layers, but only a single input layer.
    Use a layer stack to have multiple inputs
    go to a single input (backprop through
    structure essentially).

    """
    def __init__(self):
        self._forward_layer  = None
        self._backward_layer = None

    def add_backward_layer(self, layer):
        """
        Connect a layer to the antecedents
        of this layer in the graph.
        """
        if self._backward_layer is not None: self._backward_layer.remove_forward_layer(self)
        self._backward_layer = layer

    def remove_forward_layer(self, layer):
        """
        Remove a layer from the forward layers
        of this layer. Stops the propagation
        of activations in the graph here.

        """
        if self._forward_layer is layer: self._forward_layer = None
        
    def remove_backward_layer(self, layer):
        """
        Remove a layer from the antecedents
        of this layer.
        """
        if self._backward_layer is layer: self._backward_layer = None

    def connect_to(self, layer):
        """
        Adds the layer to the forward and
        backward lists of layers to connect
        the graph.
        """
        self._forward_layer = layer
        layer.add_backward_layer(self)