import os.path
import unittest

from ocr.config import RES_DIR
from ocr.segmentation import TesseractSegmentor
from ocr.utils.image_loader import ImageUtils


class TesseractSegmentorTest(unittest.TestCase):

	def test_functionality(self):

		segmentor = TesseractSegmentor()
		image = ImageUtils.load_from_res("test/ocr/segmentation/img.png")
		words_images = segmentor.segment(image)
		self.assertNotEqual(len(words_images), 0)

		for word_image in words_images:
			ImageUtils.display(word_image)
