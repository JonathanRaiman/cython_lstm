from .base_layer import BaseLayer
import numpy as np

class LoopLayer(BaseLayer):

	def __init__(self, input_size, internal_layer):
		BaseLayer.__init__(self)
		self.input_size     = input_size
		self.internal_layer = internal_layer
		self.layers = []
		last_layer = self.internal_layer
		while last_layer is not None:
			self.layers.append(last_layer)
			last_layer = last_layer._forward_layer

	def activation_holder(self, timesteps, streams):
		"""
		Construct an empty array of arrays for holding the result
		of the internal computations at every timesteps.

		For now this computation is done without recurrence built in,
		e.g. layers do not get to see the past.

		Note: We'll upgrade this in a minute.
		"""
		holder = []
		output_size = self.input_size
		for layer in self.layers:
			if hasattr(layer, 'output_size'):
				output_size = layer.output_size
			holder.append(np.zeros([timesteps, streams, output_size], dtype=layer.dtype))

		return holder

	def activate(self, x, out = None):
		"""
		Activate through time by taking
		slices along the input's first dimensions.

		Implementation Note: for now let's not worry
		too much about memory and speed and consider
		that the input x is a Python list with different
		useful pieces for the computation passed in in
		an order reasonable for the internal layer (and
		this responsability is on the user, but not on
		the loop layer).

		"""

		timesteps = x[0].shape[0]
		streams   = x[0].shape[1]

		if out is None:
			self._activation = self.activation_holder(timesteps, streams)
			out = self._activation

		for t in range(timesteps):
			# this way of doing things is a bit clumsy.
			last_layer = self.internal_layer
			input = x[0][t]
			for layer, holder in zip(self.layers, out):
				input = last_layer.activate([input], out = holder[t])

		return out

	def update_grad_input(self, input, output, grad_output):
		"""
		Here we take the gradient with respect to all time points. The error
		signal is the grad_output. The input is the original input provided
		to the loop.

		"""
		# go backwards in time
		timesteps = input.shape[0]
		if self.gradinput is None:
			self.gradinput = np.zeros_like(input)
		for t in reversed(range(timesteps)):
			self.gradinput[t] += self.internal_layer.update_grad_input(input[t], [act[t] for act in output], grad_output[t])

		return self.gradinput