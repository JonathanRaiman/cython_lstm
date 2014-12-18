import numpy as np

class Trainer():
    """
    Get the updates for a gradient descent optimizer using
    SGD, AdaDelta, or AdaGrad.

    Call train with a pair of input output to perform
    gradient descent on a model.
    
    Parameters are updated in place.

    Inputs
    ------

    model    Network         : The model to optimizer.
                               Must have the methods:
                               * `get_parameters()`,
                               * `get_gradients()`
                               * `clear()`,
                               * `activate(input)`, 
                               * `backpropagate(output)`,
                               * `error(output)`.
    lr       float           : base learning rate for
                               adagrad and SGD
    eps      float           : numerical stability value
                               to not divide by zero
                               sometimes
    rho      float           : adadelta hyperparameter.
    method   str             : 'adagrad', 'adadelta', or 'sgd'.

    """
    def __init__(self, model,
        eps=1e-6,
        rho=0.95,
        lr = 0.01,
        max_norm=5.0,
        method = "adadelta"):
        # should freeze the structure of the network or have
        # robust method of linking to the elements inside
        self.model      = model
        self._method    = method
        self.parameters = model.get_parameters()
        self.gradients  = model.get_gradients()
        self.lr         = lr
        self.rho        = rho
        self.eps        = eps

        if method == "adadelta":
            self.gsums   = [np.zeros_like(param) for param in self.parameters]
            self.xsums   = [np.zeros_like(param) for param in self.parameters]
            self._grad_update = self.adadelta_update
        elif method == "adagrad":
            self.gsums   = [np.zeros_like(param) for param in self.parameters]
            self._grad_update = self.adagrad_update
        else:
            self._grad_update = self.sgd_update

    @property
    def method(self):
        return self._method

    def adadelta_update(self):
        for gparam, param, gsum, xsum in zip(self.gradients, self.parameters, self.gsums, self.xsums):
            gsum[:] = (self.rho * gsum + (1. - self.rho) * (gparam **2)).astype(param.dtype, False)
            dparam = -np.sqrt((xsum + self.eps) / (gsum + self.eps)) * gparam
            xsum[:] = (self.rho * xsum + (1. - self.rho) * (dparam **2)).astype(param.dtype, False)
            param += dparam
    def adagrad_update(self):
        for gparam, param, gsum in zip(self.gradients, self.parameters, self.gsums):
            gsum[:] =  (gsum + (gparam ** 2)).astype(param.dtype, False)
            param -=  self.lr * (gparam / (np.sqrt(gsum + self.eps)))

    def sgd_update(self):
        for gparam, param in zip(self.gradients, self.parameters):
            param -= (self.lr * gparam)


    def train(self, input, output):
        
        # reset model activations
        self.model.clear()
        
        # run data through model
        self.model.activate(input)
        
        # backpropagate error through net:
        self.model.backpropagate(output)
        
        # collect cost:
        cost = self.model.error(output).sum()
        
        # update weights:
        self._grad_update()
            
        return cost