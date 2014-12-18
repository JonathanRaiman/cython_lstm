import numpy as np

class Error(object):
    pass

def softmax_error_one_hot(x, t):
    return -np.log(x[np.arange(0,t.shape[0]), t]).sum()

class MSE(Error):
    @staticmethod
    def dEdy(y, t):
        return y - t

    @staticmethod
    def error(y, target):
        """
        Mean squared error : 1/2 (y-t)^2
        """
        return 0.5 * (y - target)**2

class BinaryCrossEntropy(MSE):
    @staticmethod
    def error(y, target):
        """
        Binary Cross entropy error:
        D_{KL}(p || q) = sum_i { (1-p_i) * log(1 - q_i) -p_i log (q_i) }
        """
        return -(target * np.log(y) + (1.0 - target) * np.log1p(-y.clip(max=0.99999999)))

class TanhBinayCrossEntropy(MSE):
    @staticmethod
    def error(y, target):
        """
        Cross entropy error (we reshape tanh activation into
        a sigmoid like activation and then apply cross entropy
        criterion to it) 
        """
        resized_activation = (y + 1.0) / 2.0
        return -(target * np.log(resized_activation) + (1.0 - target) * np.log1p(- resized_activation))

class CategoricalCrossEntropy(MSE):
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