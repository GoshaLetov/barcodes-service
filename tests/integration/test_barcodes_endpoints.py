import cv2
import numpy as np

from fastapi.testclient import TestClient
from http import HTTPStatus


def test_health(client: TestClient):
    response = client.get(url='/barcodes/health')
    assert response.status_code == HTTPStatus.OK


def test_extract_barcode(client: TestClient, barcode_image_bytes: bytes):
    """ find credentials on obvious barcode image """
    files = {'image': barcode_image_bytes}

    response = client.post(url='/barcodes/extract', files=files)
    assert response.status_code == HTTPStatus.OK

    inferred_barcodes = response.json().get('barcodes')
    assert isinstance(inferred_barcodes, list)

    assert len(inferred_barcodes) == 1 and isinstance(inferred_barcodes[0], dict)


def test_draw_barcode(client: TestClient, barcode_image_bytes: bytes, barcode_image_numpy: np.ndarray):
    """ compare image shapes after draw """
    files = {'image': barcode_image_bytes}

    response = client.post(url='/barcodes/draw', files=files)
    assert response.status_code == HTTPStatus.OK

    image = cv2.imdecode(buf=np.frombuffer(buffer=response.read(), dtype=np.uint8), flags=cv2.IMREAD_COLOR)

    assert barcode_image_numpy.shape == image.shape
