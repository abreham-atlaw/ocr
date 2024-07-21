import typing

import numpy as np
from PIL import Image


class OCRModel:

	def __init__(self, model: 'keras.Model', word_list: typing.List[str], image_shape: typing.Tuple[int, int] = (128, 128)):
		self.__model = model
		self.__words = word_list
		self.__image_shape = image_shape

	def __pad_image(self, image: np.ndarray) -> np.ndarray:
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

	def predict(self, images: typing.List[np.ndarray]) -> typing.List[str]:
		images = [
			np.expand_dims(self.__pad_image(image), axis=0)
			for image in images
		]
		images = np.concatenate(images, axis=0)
		idxs = np.argmax(self.__model.predict(images), axis=1)
		return [self.__words[idx] for idx in idxs]
