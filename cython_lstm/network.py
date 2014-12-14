import numpy as np, networkx as nx, matplotlib.pyplot as plt
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
        
    def create_network_graph(self):
        """
        Create a networkx representation
        of the current layer arrangement
        for this network.
        """
        graph        = nx.DiGraph()
        nodes_pos    = {}
        nodes_label  = {}
        input_units  = []
        output_units = []
        outputted_units = []
        hidden_units = []
        
        layer_index = {}
        
        # add the units per layer
        for layer_i, layer in enumerate(self.layers):
            layer_index[layer] = layer_i

            # add the nodes for this layer
            for node_i in range(layer.input_size):
                node = "%d_%d" % (layer_i, node_i)  # simple encoding of a node
                graph.add_node(node)
                nodes_pos[node] = (layer_i, float(layer.input_size) / 2.0 - float(node_i))

                if layer == self._input_layer:
                    # label for input layer
                    input_units.append(node)
                    nodes_label[node] = r"$X_{%d}$" % node_i
                elif layer == self._output_layer:
                    # label for output layer
                    output_units.append(node)
                    nodes_label[node] = r"$Y_{%d}$" % node_i
                else:
                    # hidden layer
                    hidden_units.append(node)
                    nodes_label[node] = r"$Z_{%d, %d}$" % (layer_i, node_i)
                    
            if layer is self._output_layer:
                for i in range(layer.output_size):
                    node = "%d_%d" % (len(self.layers), i)  # simple encoding of a node
                    graph.add_node(node)
                    outputted_units.append(node)
                    nodes_label[node] = r"$O_{%d}$" % i
                    nodes_pos[node] = (len(self.layers), layer.output_size / 2.0 - i)
        
        for layer_i, layer in enumerate(self.layers):
            for forward_layer in layer._forward_layers:
                graph.add_edges_from([
                    ("%d_%d" % (layer_i, k), "%d_%d" % (layer_index[forward_layer], l))
                    for k in range(layer.input_size) for l in range(forward_layer.input_size)
                ])
            if layer is self._output_layer:
                # map to fictional output nodes for net
                graph.add_edges_from([
                    ("%d_%d" % (layer_i, k), "%d_%d" % (len(self.layers), l))
                    for k in range(layer.input_size) for l in range(layer.output_size)
                ])
        
        return (graph, nodes_pos, nodes_label, input_units, output_units, outputted_units, hidden_units)
        
    def draw(self, ax = None):
        """ Draw the neural network
        """
        if ax is None:
            ax = plt.figure(figsize=(10, 6)).add_subplot(1, 1, 1)
            
        graph, nodes_pos, nodes_label, input_units, output_units, outputted_units, hidden_units = self.create_network_graph()
            
        nx.draw_networkx_edges(graph, pos=nodes_pos, alpha=0.7, ax=ax)
        nx.draw_networkx_nodes(graph, nodelist=input_units,
                               pos=nodes_pos, ax=ax,
                               node_color='#66FFFF', node_size=700)
        nx.draw_networkx_nodes(graph, nodelist=hidden_units,
                               pos=nodes_pos, ax=ax,
                               node_color='#CCCCCC', node_size=900)
        nx.draw_networkx_nodes(graph, nodelist=output_units,
                               pos=nodes_pos, ax=ax,
                               node_color='#FFFF99', node_size=700)
        nx.draw_networkx_labels(graph, labels=nodes_label,
                                pos=nodes_pos, font_size=14, ax=ax)
        
        nx.draw_networkx_nodes(graph, nodelist=outputted_units,
                       pos=nodes_pos, ax=ax,
                       node_color='#f2276e', node_size=470)
        ax.axis('off');

        layer_sizes = [layer.input_size for layer in self.layers] + [self._output_layer.output_size]
        activation_func = [layer.activation_function.__doc__ for layer in self.layers] + ["Output"]
        max_heights_for_layer = [max([nodes_pos["%d_%d" % (k, node)][1]  for node in range(size)]) for k, size in enumerate(layer_sizes)]

        for i, layer_height in enumerate(max_heights_for_layer):
            ax.text(i, layer_height + 0.8, activation_func[i], horizontalalignment ='center');
        ax.axis('off')

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