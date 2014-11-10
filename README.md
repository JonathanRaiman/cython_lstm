Cython LSTM
-----------

@author Jonathan Raiman
@date 3rd November 2014

See the current implementation [on this notebook](http://nbviewer.ipython.org/github/JonathanRaiman/cython_lstm/blob/master/Cython%20LSTM.ipynb).

## Capabilities:

* Multi Layer Perceptrons

* Backprop over the network

* Tanh, Logistic, Softmax, Rectifier, Linear activations

* Recurrent Neural Networks (Hidden states only, no memory)

* Backprop through time

* Draw graph of network using matplotlib ([see notebook](http://nbviewer.ipython.org/github/JonathanRaiman/cython_lstm/blob/master/Cython%20LSTM.ipynb#drawing-the-network))

* Training using SGD or batch gradient descent

* Tensor networks (quadratic form connecting layers)

### Key design goals 

* are to mimic simplicity and practicaly of Pynnet and Cybrain / Pybrain.

* Model connections using matrices not explicit connections (to get vector algebra involved)

* Construct and run million parameter models for LSTM and RNN type models

* Be able to run AdaGrad / RMSprop on gradients easily

#### Icing on the cake

* Support dtype float32, float64 (currently float32), and int32 / int64 for indices

* BackProp through structure

* Variable input size indices for RNN (so batches of different sequence sizes can be run adjacent to one another -- currently difficult given numpy array size restrictions)

* Language Models / Hiearchical Softmax parameters

* Have an interface for Theano variables if needed (avoid compilation times and make everything cythonish)