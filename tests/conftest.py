import cv2
import pytest

from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import barcodes
from src.constants import PROJECT_PATH
from src.config import Config

TESTS_DIR = Path(__file__).parent


@pytest.fixture(scope='function')
def barcode_image_bytes():
    """ Barcode image: bytes """
    with open(TESTS_DIR / 'images' / 'barcode.jpg', 'rb') as image:
        yield image


@pytest.fixture
def barcode_image_numpy():
    """ Barcode image: ndarray """
    return cv2.cvtColor(
        src=cv2.imread(filename=str(TESTS_DIR / 'images' / 'barcode.jpg')),
        code=cv2.COLOR_BGR2RGB,
    )


@pytest.fixture
def barcode_only_image_numpy():
    """ Barcode only image: ndarray """
    return cv2.cvtColor(
        src=cv2.imread(filename=str(TESTS_DIR / 'images' / 'barcode_only.png')),
        code=cv2.COLOR_BGR2RGB,
    )


@pytest.fixture(scope='function')
def barcode_two_image_numpy():
    """ Barcode only image: ndarray """
    return cv2.cvtColor(
        src=cv2.imread(filename=str(TESTS_DIR / 'images' / 'barcode_two.jpg')),
        code=cv2.COLOR_BGR2RGB,
    )


@pytest.fixture(scope='function')
def fake_image_numpy():
    """ Fake image: numpy """
    return cv2.cvtColor(
        src=cv2.imread(filename=str(TESTS_DIR / 'images' / 'fake.jpg')),
        code=cv2.COLOR_BGR2RGB,
    )


@pytest.fixture(scope='session')
def config():
    """ Application config """
    return Config.from_yaml(path=PROJECT_PATH / 'config.yaml')


@pytest.fixture
def barcodes_container(config):
    """ Barcodes container """
    container = barcodes.Container()
    container.config.from_dict(options=config)
    return container


@pytest.fixture
def wired_planet_container(barcodes_container):
    """ Wired barcodes container """
    barcodes_container.wire([barcodes.routes])
    yield barcodes_container
    barcodes_container.unwire()


@pytest.fixture
def test_app(wired_planet_container):
    """ Fast API Test app """
    app = FastAPI()
    app.include_router(router=barcodes.router)
    return app


@pytest.fixture
def client(test_app):
    """ Test client """
    return TestClient(app=test_app)
