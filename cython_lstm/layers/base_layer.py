from .connectible_layer import ConnectibleLayer

class BaseLayer(ConnectibleLayer):
    """
    Base Neural Network Layer, defines the key 
    methods that need to be implemented. Is not
    useful until inherited from."""

    def __init__(self, dtype='float32'):
        ConnectibleLayer.__init__(self)
        self.dtype       = dtype
        self._activation = None
        self.gradinput   = None
        self.params      = []

    def clear(self):
        self.gradinput = None
        for grad in self.gradients:
            grad.fill(0)

    def activate(self, x, out=None):
        """
        Each layer must have an activate method
        that can output to the out parameter
        or returns its out allocated output.
        """
        raise NotImplementedError

    def update_grad_input(self, input, output, grad_output):
        """
        Each layer must have an update grad input method
        that takes care of updating its gradients and passing
        those down.
        """
        raise NotImplementedError
        
    def activate_forward_layers(self):
        # first activate forward.
        for layer in self._forward_layers:
            layer.activate(self.activation())
        # then pass the message onwards:
        for layer in self._forward_layers:
            layer.activate_forward_layers()

    def __repr__(self):
        return "<" + self.__class__.__name__ + " " + str({"activation": self.activation_function.__doc__ if hasattr(self, 'activation_function') else '', "input_size": self.input_size if hasattr(self, 'input_size') else '', "output_size": self.output_size if hasattr(self, 'output_size') else ''})+">"