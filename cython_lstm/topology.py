"""
Topology submodule handling topological
sorts for handling of forward and backward
propagation over arbitrary graphs (similar to
nn-graph for Torch).
"""
class Node(object):
    """
    Takes a layer (inheriting from BaseLayer)
    and a mapping function taking layers to
    Nodes, and recursively maps all layers
    into a graph.
    """
    def __init__(self, layer, mapping):
        mapping[layer] = self
        self.layer = layer
        self.parents  = []
        for parent in layer.parents:
            if parent in mapping:
                self.parents.append(mapping[parent])
            else:
                Node(parent, mapping)
                self.parents.append(mapping[parent])
        self.children = []
        for child in layer.children:
            if child in mapping:
                self.children.append(mapping[child])
            else:
                Node(child, mapping)
                self.children.append(mapping[child])

    @staticmethod
    def to_layer_mapping(layers):
        """
        Convert several layers to Nodes,
        recursively cross the graph starting at those
        layers and convert the remainder to nodes.
        Return the mapping dictionary.

        Inputs
        ------

        nodes list: the roots of the graph

        Outputs
        -------

        mapped_layers dict: a mapping from layer to Node.
        """
        mapped_layers = {}
        for layer in layers:
            if not layer in mapped_layers:
                mapped_layers[layer] = Node(layer, mapped_layers)
        return mapped_layers

    @staticmethod
    def to_nodes(layers):
        """
        Convert several layers to Nodes,
        recursively cross the graph starting at those
        layers and convert the remainder to nodes.
        Return the input layers as nodes.

        Inputs
        ------

        nodes list: the roots of the graph

        Outputs
        -------

        mapped_nodes list: Converted layers
                           into Nodes of the
                           calculation graph.
        """
        mapped_layers = Node.to_layer_mapping(layers)
        return [mapped_layers[layer] for layer in layers]

def topological_sort(nodes):
    """
    Find an ordering of nodes that respects
    all dependencies in the graph.

    Inputs
    ------

    nodes list: the roots of the graph

    Outputs
    -------

    L list: an ordering of the nodes in the graph
            such that all dependencies are respected
            by proceeding in the provided order.

    Note: this function does not check whether underlying
    graph is a directed acyclic graph. Since cycles are
    not detected, certain nodes may never be called, and
    hence the behavior in those cases will differ from
    user-intended.
    """
    L = []
    S = nodes
    while len(S) > 0:
        node = S.pop()
        L.append(node)
        children = [m for m in node.children]
        for child in children:
            # remove edge from graph
            del node.children[node.children.index(child)]
            del child.parents[child.parents.index(node)]

            # if this was the last connection
            # then we can add this node to
            # the nodes with no incoming connections:
            # S
            if len(child.parents) == 0:
                S.append(child)
    return L