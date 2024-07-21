import os

import numpy as np
from PIL import Image

from ocr.config import RES_DIR


class ImageUtils:

	@staticmethod
	def load(path: str) -> np.ndarray:
		return np.array(Image.open(path))

	@staticmethod
	def load_from_res(path: str) -> np.ndarray:
		return ImageUtils.load(os.path.join(RES_DIR, path))

	@staticmethod
	def display(image: np.ndarray):
		Image.fromarray(image).show()
