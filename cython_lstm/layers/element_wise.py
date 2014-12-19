from .base_layer import BaseLayer
import numpy as np

class ElementWise(BaseLayer):
	"""
	Sum the elements of both parent layers
	element wise. The gradient is equally
	shared among the parents.

	Note: untested
	"""
	def __init__(self, a, b):
		BaseLayer.__init__(self)
		self.a = a
		self.b = b
		# inform the topological sort
		# of this dependency
		self.parents.append(a)
		self.parents.append(b)
		a.children.append(self)
		b.children.append(self)

class ElementWiseSum(ElementWise):
	def activate(self, x, out=None):
		"""
		Activate by passing the list of activations
		for both parents.
		"""
		if out is not None:
			out[:] = x[0] + x[1]
		else:
			return x[0] + x[1]
	def update_grad_input(self, input, output, grad_output):
		num_singletons = [len(grad_output.shape) - len(x.shape) for x in input]

		self.gradinput = [np.sum(grad_output,
			axis=tuple(range(num_singletons[i])),
			keepdims=False) if num_singletons[i] > 0 else grad_output
			for i in len(input)]
		return self.gradinput
class ElementWiseSub(ElementWise):
	def activate(self, x, out=None):
		if out is not None:
			out[:] = x[0] - x[1]
		else:
			return x[0] - x[1]
	def update_grad_input(self, input, output, grad_output):
		num_singletons = [len(grad_output.shape) - len(x.shape) for x in input]

		self.gradinput = [np.sum(grad_output,
			axis=tuple(range(num_singletons[i])),
			keepdims=False) if num_singletons[i] > 0 else grad_output
			for i in len(input)]
		# this one has a negative gradient:
		self.gradinput[1] *= -1
		return self.gradinput
class ElementWiseProd(ElementWise):
	def activate(self, x, out=None):
		if out is not None:
			out[:] = x[0] * x[1]
		else:
			return x[0] * x[1]
	def update_grad_input(self, input, output, grad_output):
		raise NotImplementedError()
		# self.gradinput = map(np.ones_like, input)
		# self.gradinput[1] *= -1
		# for grad, grad_out in zip(self.gradinput, grad_output):
		# 	grad *= grad_out
		# return self.gradinput