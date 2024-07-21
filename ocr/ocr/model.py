import numpy as np


class DummyModel:

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return np.random.random((inputs.shape[0], 256))
