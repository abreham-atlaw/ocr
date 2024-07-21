import typing

import numpy as np
import pytesseract
from pytesseract import Output
from ocr.segmentation import Segmentor


class TesseractSegmentor(Segmentor):

	def get_bounds(self, image: np.ndarray) -> typing.List[typing.Tuple[int, int, int, int]]:
		details = pytesseract.image_to_data(image, output_type=Output.DICT)
		bounds = []
		for i in range(len(details['level'])):
			if details['level'][i] == 5:
				x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
				bounds.append((x, y, x + w, y + h))
		return bounds
