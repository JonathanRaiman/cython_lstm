import numpy as np
try:
    import scipy.special
    sigmoid = scipy.special.expit
except ImportError:
    def sigmoid(x, out=None):
        """
        Sigmoid implemented using proxy
        ufuncs from numpy.
        """
        return np.divide(1., 1. + np.exp(-x), out, dtype=x.dtype)

def softmax(x, out=None):
    layer_max = x.max(axis=-1)
    exped_distributions = np.exp(x.T - layer_max)
    total_distribution = exped_distributions.sum(axis=0)
    if out is None:
        return (exped_distributions / total_distribution).T
    else:
        out[:] = (exped_distributions / total_distribution).T
        return out

def softmax_unsafe(x):
    exped_distributions = np.exp(x.T)
    total_distribution = exped_distributions.sum(axis=0)
    return (exped_distributions / total_distribution).T

class Neuron():
    @staticmethod
    def activation_function(x, out=None):
        """Identity"""
        if out is None:
            return x
        else:
            out[:] = x
            return out
    
    @staticmethod
    def dydz(x):
        return np.float64(1.0).astype(x.dtype)

class RectifierNeuron(Neuron):
    @staticmethod
    def activation_function(x, out=None):
        """Rectifier"""
        return np.fmax(0,x, out)
    
    @staticmethod
    def dydz(x):
        return np.sign(x)
    
class LogisticNeuron(Neuron):
    @staticmethod
    def activation_function(x, out=None):
        """Sigmoid"""
        return sigmoid(x,out)
    
    @staticmethod
    def dydz(x):
        """
        Sigmoid derivative
        d/dx 1/ ( 1 + e^-x)  = d/dx sig(x) = sig(x) - sig(x)^2
        """
        return x - x**2
    
class TanhNeuron(Neuron):
    @staticmethod
    def activation_function(x, out=None):
        """Tanh"""
        return np.tanh(x, out)
    
    @staticmethod
    def dydz(x):
        """
        hyperbolic tangent Derivative
        """
        return 1.0 - x**2

class SoftmaxNeuron(Neuron):
    @staticmethod
    def activation_function(x, out=None):
        """Softmax"""
        return softmax(x, out)