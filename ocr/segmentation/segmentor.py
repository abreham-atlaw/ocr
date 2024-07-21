from abc import ABC, abstractmethod
import typing

import numpy as np


class Segmentor(ABC):

	@abstractmethod
	def get_bounds(self, image: np.ndarray) -> typing.List[typing.Tuple[int, int, int, int]]:
		pass

	def segment(self, image: np.ndarray) -> typing.List[np.ndarray]:
		bounds = self.get_bounds(image)
		return [image[b[1]:b[3], b[0]:b[2]] for b in bounds]
