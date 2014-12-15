# from .temporal_layer import TemporalLayer

# class ComputationGraph(object):
# 	"""
# 	ComputationGraph object controls the flow
# 	of activations in the network and detects
# 	disconnected components.

# 	TODO: find clever way of ensuring temporal
# 		  datasets are iterated through, and
# 		  regular program flow is preserved.

# 	IDEA: implement a special LoopLayer that
# 	takes as input data and iterates over its
# 	first dimension. It has an internal set
# 	of nodes. Each node receives the loop
# 	data and the timestep.

# 	The Loop can be connected forwards and
# 	backwards with other ordinary layers,
# 	such as the slice layer.

# 	The slice layer then takes the output
# 	of the loop layer at a specific time
# 	point and passes that forward. A
# 	Softmax layer placed on the end of this
# 	sequence will then receive the specific
# 	output at a given timestep. Backprop
# 	can then be implemented in many different
# 	ways. The easiest is to pass errors to
# 	the loop and let the loop figure how
# 	to enable each inner element to do
# 	backprop on its own.

# 	Easy fix for now is to remove all temporal
# 	information, and make all nodes temporally
# 	aware. The loop layer is a priority.

# 	"""

# 	def __init__(self, network):
# 		self.network = network

# 	def trace_computation(self):
# 		temporal_trace = issubclass(type(self.network.input_layer), TemporalLayer)

# 		if temporal_trace:
# 			# test 1 timestep, 1 stream of data, with correct input size.
# 			test_value = np.zeros([1, 1, self.network.input_layer.input_size], dtype=self.network.input_layer.dtype)
# 			self.network.allocate_activation(1, 1)
# 		else:
# 			# test 1 stream of data, with correct input size.
# 			test_value = np.zeros([1, self.network.input_layer.input_size], dtype=self.network.input_layer.dtype)

# 		self.network.activate(test_value)

# 		post_activation_layers = []

# 		for layer in self.network.layers:
# 			if layer._activation is None:
# 				post_activation_layers.append(layer)

# 		pass