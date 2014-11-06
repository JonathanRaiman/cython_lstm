Cython LSTM
-----------
@author Jonathan Raiman

See the current implementation [on this notebook](http://nbviewer.ipython.org/github/JonathanRaiman/cython_lstm/blob/master/Cython%20LSTM.ipynb).

## Capabilities:

* Multi Layer Perceptrons

* Backprop over the network

* Tanh, Logistic, Linear activations

* Recurrent Neural Networks (Hidden states only, no memory)

* Backprop through time

* Output graph of network to matplotlib ([see notebook](http://nbviewer.ipython.org/github/JonathanRaiman/cython_lstm/blob/master/Cython%20LSTM.ipynb))



### Key design goals 

* are to mimic simplicity and practicaly of Pynnet and Cybrain / Pybrain.

* Model connections using matrices not explicit connections (to get vector algebra involved)

* Be able to construct and run million parameter models for LSTM and RNN type models

* Be able to do AdaGrad / RMSprop on gradients easily

#### Icing on the cake

* Support dtype float32, float64 (currently float32), and int32 / int64 for indices

* BackProp through structure

* Variable input size indices for RNN

* Language Models / Hiearchical Softmax parameters

* Have an interface for Theano variables if needed (avoid compilation times and make everything cythonish)