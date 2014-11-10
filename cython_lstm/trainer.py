class Trainer():
    """
    Dead simple training algorithm using backpropagation
    for a, artificial neural network.
    
    """
    def __init__(self, network, alpha = 0.035):
        # should freeze the structure of the network or have
        # robust method of linking to the elements inside
        self.network = network
        self.parameters = network.get_parameters()
        self.gradients = network.get_gradients()
        self._alpha = alpha
        
    def train(self, input, output):
        
        # reset network activations
        self.network.clear()
        
        # run data through network
        self.network.activate(input)
        
        # backpropagate error through net:
        self.network.backpropagate(output)
        
        # collect cost:
        cost = self.network.error(output).sum()
        
        # update weights:
        for gparam, param in zip(self.gradients, self.parameters):
            param -= (self._alpha * gparam)
            
        return cost