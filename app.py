from flask import Flask, request, jsonify
from ocr.di.core_providers import CoreProviders
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
ocr = CoreProviders.provide_ocr()


@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
	if 'file' not in request.files:
		return jsonify({"error": "No file part"}), 400
	file = request.files['file']
	if file.filename == '':
		return jsonify({"error": "No selected file"}), 400

	try:
		# Convert file to an image
		image = Image.open(io.BytesIO(file.read()))
		image = np.array(image)

		# Perform OCR
		sentence = ocr.transform(image)
		return jsonify({"text": sentence})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=8888, debug=True)
