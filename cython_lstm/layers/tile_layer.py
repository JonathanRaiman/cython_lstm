from .base_layer import BaseLayer

class TileLayer(BaseLayer):
    """
    Repeats an input sequence to all its outgoing
    layers. Useful for repeating a data layer
    to multiple nodes that need to listen to it.
    """

    def forward_propagate(self, x):
        return [layer.forward_propagate(x) for layer in self._temporal_forward_layers]

    def clear(self):
        self.step        = 0
        self._activation = None

    def allocate_activation(self, *args):
        pass

    @property
    def input_size(self):
        return self._internal_layers[0].input_size if len(self._temporal_forward_layers) > 0 else None

    @property
    def output_size(self):
        return self._internal_layers[0].input_size if len(self._temporal_forward_layers) > 0 else None

    def activate(self, input):
        """
        For each layer within this repeat
        layer, the input is sent.

        """
        self._activation = input

        self.step = 0
        self.recursive_activate_timestep(input)

        # transfer activation as input to next layers:
        self.activate_forward_layers()

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