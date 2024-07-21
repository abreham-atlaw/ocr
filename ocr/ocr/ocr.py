from abc import abstractmethod, ABC

import numpy as np


class OCR(ABC):

	@abstractmethod
	def transform(self, image: np.ndarray) -> str:
		pass
