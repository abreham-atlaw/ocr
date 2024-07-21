import json

from ocr.config import CLS_MODEL_PATH, WORDLIST_PATH
from ocr.ocr import OCRModel, SegmentationOCR, OCR
from ocr.ocr.model import DummyModel
from ocr.segmentation import Segmentor, TesseractSegmentor


class CoreProviders:

	__cls_model = None
	__ocr_model = None

	@staticmethod
	def provide_ocr() -> OCR:
		return SegmentationOCR(CoreProviders.provide_segmentor(), CoreProviders.provide_ocr_model())

	@staticmethod
	def provide_segmentor() -> Segmentor:
		return TesseractSegmentor()

	@staticmethod
	def provide_ocr_model() -> OCRModel:
		if CoreProviders.__ocr_model is None:
			CoreProviders.__ocr_model = OCRModel(CoreProviders.provide_cls_model(), CoreProviders.provide_word_list())
		return CoreProviders.__ocr_model

	@staticmethod
	def provide_cls_model() -> 'keras.Model':
		if CoreProviders.__cls_model is None:
			# CoreProviders.__cls_model = keras.models.load_model(CLS_MODEL_PATH)
			CoreProviders.__cls_model = DummyModel()
		return CoreProviders.__cls_model

	@staticmethod
	def provide_word_list():
		with open(WORDLIST_PATH) as file:
			return json.load(file)
