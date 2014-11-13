import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    layer_max = x.max(axis=-1)
    exped_distributions = np.exp(x.T - layer_max)
    total_distribution = exped_distributions.sum(axis=0)
    return (exped_distributions / total_distribution).T

def softmax_unsafe(x):
    exped_distributions = np.exp(x.T)
    total_distribution = exped_distributions.sum(axis=0)
    return (exped_distributions / total_distribution).T
    
def softmax_error_one_hot(x, t):
    return -np.log(x[np.arange(0,t.shape[0]), t]).sum()

class Neuron():
    @staticmethod
    def activation_function(x):
        """Identity"""
        return x
    
    @staticmethod
    def dydz(x):
        return 1.0

    @staticmethod
    def dEdy(y, t):
        return y - t

    @staticmethod
    def error(y, target):
        """
        Mean squared error : 1/2 (y-t)^2
        """
        return 0.5 * (y - target)**2

class RectifierNeuron(Neuron):
    @staticmethod
    def activation_function(x):
        """Rectifier"""
        return np.fmax(0,x)
    
    @staticmethod
    def dydz(x):
        return np.sign(x)
    
class LogisticNeuron(Neuron):
    @staticmethod
    def activation_function(x):
        """Sigmoid"""
        return sigmoid(x)
    
    @staticmethod
    def dydz(x):
        """
        Sigmoid derivative
        d/dx 1/ ( 1 + e^-x)  = d/dx sig(x) = sig(x) - sig(x)^2
        """
        return x - x**2

    @staticmethod
    def error(y, target):
        """
        Binary Cross entropy error:
        D_{KL}(p || q) = sum_i { (1-p_i) * log(1 - q_i) -p_i log (q_i) }
        """
        return -(target * np.log(y) + (1.0 - target) * np.log(1.0 - y))
    
class TanhNeuron(Neuron):
    @staticmethod
    def activation_function(x):
        """Tanh"""
        return np.tanh(x)
    
    @staticmethod
    def dydz(x):
        """
        hyperbolic tangent Derivative
        """
        return 1.0 - x**2

    @staticmethod
    def error(y, target):
        """
        Cross entropy error (we reshape tanh activation into
        a sigmoid like activation and then apply cross entropy
        criterion to it)
        
        """
        resized_activation = (y + 1.0) / 2.0
        return -(target * np.log(resized_activation) + (1.0 - target) * np.log(1.0 - resized_activation))
    
class SoftmaxNeuron(Neuron):
    @staticmethod
    def activation_function(x):
        """Softmax"""
        return softmax(x)

    @staticmethod
    def dEdy(y, t):
        dEdy = y.copy()
        dEdy[np.arange(0,t.shape[0]), t] -= 1.
        return dEdy

    @staticmethod
    def error(y, target):
        """
        Cross entropy error (we reshape tanh activation into
        a sigmoid like activation and then apply cross entropy
        criterion to it) (One hot outputs)
        
        """
        return softmax_error_one_hot(y, target)