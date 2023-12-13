import numpy as np

from src.config import Config
from src.barcodes.container import Container


def test_not_fail(barcodes_container: Container, barcode_only_image_numpy: np.ndarray):
    """ all methods raw check """
    ocr_model = barcodes_container.barcodes_ocr()
    ocr_model.inference(image=barcode_only_image_numpy)
    ocr_model.extract_text(image=barcode_only_image_numpy)


def test_inference(barcodes_container: Container, barcode_only_image_numpy: np.ndarray, config: Config):
    """ vocab size and confidences intervals """
    ocr_model = barcodes_container.barcodes_ocr()
    labels, confidences = ocr_model.inference(image=barcode_only_image_numpy)

    assert len(config.ocr.vocab) == labels.max() - labels.min()
    assert confidences.min() >= 0 and confidences.max() <= 1


def test_text(barcodes_container: Container, barcode_only_image_numpy: np.ndarray, config: Config):
    """ extracted text vocab """
    ocr_model = barcodes_container.barcodes_ocr()
    text = ocr_model.extract_text(image=barcode_only_image_numpy)

    assert all([letter in config.ocr.vocab for letter in text])
