from abc import ABC, abstractmethod
import typing

import numpy as np
from PIL import Image


class Segmentor(ABC):

	def __init__(self, image_shape = (128, 128)):
		self.__image_shape = image_shape

	@abstractmethod
	def get_bounds(self, image: np.ndarray) -> typing.List[typing.Tuple[int, int, int, int]]:
		pass

	def __scale(self, image: np.ndarray) -> np.ndarray:
		target_height, target_width = self.__image_shape
		image_height, image_width = image.shape[:2]

		if image_height > target_height or image_width > target_width:
			scale_factor = min(target_height / image_height, target_width / image_width)
			new_size = (int(image_width * scale_factor), int(image_height * scale_factor))

			pil_image = Image.fromarray(image)
			pil_image = pil_image.resize(new_size, Image.LANCZOS)
			image = np.array(pil_image)
			image_height, image_width = image.shape[:2]

		pad_height = max(target_height - image_height, 0)
		pad_width = max(target_width - image_width, 0)

		top = pad_height // 2
		bottom = pad_height - top
		left = pad_width // 2
		right = pad_width - left

		if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
			padded_image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=255)
		elif len(image.shape) == 3 and image.shape[2] == 4:  # Color image with alpha channel
			padded_image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=255)
		else:
			padded_image = np.pad(image, ((top, bottom), (left, right)), mode='constant', constant_values=255)

		return padded_image

	def segment(self, image: np.ndarray) -> typing.List[np.ndarray]:
		bounds = self.get_bounds(image)
		images = [image[b[1]:b[3], b[0]:b[2]] for b in bounds]
		images = [
			self.__scale(image)
			for image in images
		]
		return images
