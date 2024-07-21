import numpy as np

from ocr.ocr.ocr import OCR
from ocr.ocr.ocr_model import OCRModel
from ocr.segmentation import Segmentor


class SegmentationOCR(OCR):

	def __init__(self, segmentor: Segmentor, model: OCRModel):
		self.__segmentor = segmentor
		self.__model = model

	def transform(self, image: np.ndarray) -> str:
		words_images = self.__segmentor.segment(image)
		return " ".join(self.__model.predict(words_images))
