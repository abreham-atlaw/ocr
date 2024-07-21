import unittest

from ocr.di.core_providers import CoreProviders
from ocr.utils.image_loader import ImageUtils


class SegmentationOCRTest(unittest.TestCase):

	def test_functionality(self):
		ocr = CoreProviders.provide_ocr()
		image = ImageUtils.load_from_res("test/ocr/segmentation/img.png")
		sentence = ocr.transform(image)
		self.assertIsNotNone(sentence)
