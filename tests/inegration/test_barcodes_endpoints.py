from fastapi.testclient import TestClient
from http import HTTPStatus


def test_inference_barcode(client: TestClient, barcode_image_bytes: bytes) -> None:
    files = {'image': barcode_image_bytes}
    response = client.post(url='/barcodes/inference', files=files)
    assert response.status_code == HTTPStatus.OK

    inferred_barcodes = response.json().get('barcodes')
    assert isinstance(inferred_barcodes, list)

    # TODO: check


def test_inference_fake(client: TestClient, fake_image_bytes: bytes) -> None:
    files = {'image': fake_image_bytes}
    response = client.post(url='/barcodes/inference', files=files)
    assert response.status_code == HTTPStatus.OK

    inferred_barcodes = response.json().get('barcodes')
    assert isinstance(inferred_barcodes, list)
    assert inferred_barcodes == []

    # TODO: check


def test_draw_barcode(client: TestClient, barcode_image_bytes: bytes):
    files = {'image': barcode_image_bytes}
    # TODO: check
