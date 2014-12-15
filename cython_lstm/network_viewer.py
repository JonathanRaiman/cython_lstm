import networkx as nx
import matplotlib.pyplot as plt

def create_network_graph(network):
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
    for layer_i, layer in enumerate(network.layers):
        layer_index[layer] = layer_i

        # add the nodes for this layer
        for node_i in range(layer.input_size):
            node = "%d_%d" % (layer_i, node_i)  # simple encoding of a node
            graph.add_node(node)
            nodes_pos[node] = (layer_i, float(layer.input_size) / 2.0 - float(node_i))

            if layer == network._input_layer:
                # label for input layer
                input_units.append(node)
                nodes_label[node] = r"$X_{%d}$" % node_i
            elif layer == network._output_layer:
                # label for output layer
                output_units.append(node)
                nodes_label[node] = r"$Y_{%d}$" % node_i
            else:
                # hidden layer
                hidden_units.append(node)
                nodes_label[node] = r"$Z_{%d, %d}$" % (layer_i, node_i)
                
        if layer is network._output_layer:
            for i in range(layer.output_size):
                node = "%d_%d" % (len(network.layers), i)  # simple encoding of a node
                graph.add_node(node)
                outputted_units.append(node)
                nodes_label[node] = r"$O_{%d}$" % i
                nodes_pos[node] = (len(network.layers), layer.output_size / 2.0 - i)
    
    for layer_i, layer in enumerate(network.layers):
        for forward_layer in layer._forward_layers:
            graph.add_edges_from([
                ("%d_%d" % (layer_i, k), "%d_%d" % (layer_index[forward_layer], l))
                for k in range(layer.input_size) for l in range(forward_layer.input_size)
            ])
        if layer is network._output_layer:
            # map to fictional output nodes for net
            graph.add_edges_from([
                ("%d_%d" % (layer_i, k), "%d_%d" % (len(network.layers), l))
                for k in range(layer.input_size) for l in range(layer.output_size)
            ])
    
    return (graph, nodes_pos, nodes_label, input_units, output_units, outputted_units, hidden_units)
    
def draw(network, ax = None):
    """ Draw the neural network
    """
    if ax is None:
        ax = plt.figure(figsize=(10, 6)).add_subplot(1, 1, 1)
        
    graph, nodes_pos, nodes_label, input_units, output_units, outputted_units, hidden_units = create_network_graph(network)
        
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

    layer_sizes = [layer.input_size for layer in network.layers] + [network._output_layer.output_size]
    activation_func = [layer.activation_function.__doc__ for layer in network.layers] + ["Output"]
    max_heights_for_layer = [max([nodes_pos["%d_%d" % (k, node)][1]  for node in range(size)]) for k, size in enumerate(layer_sizes)]

    for i, layer_height in enumerate(max_heights_for_layer):
        ax.text(i, layer_height + 0.8, activation_func[i], horizontalalignment ='center');
    ax.axis('off')