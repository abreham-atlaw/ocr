import os.path

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RES_DIR = os.path.join(BASE_DIR, "res")

CLS_MODEL_PATH = os.path.join(RES_DIR, "ocr/ocr/model.keras")
WORDLIST_PATH = os.path.join(RES_DIR, "ocr/ocr/words.json")
