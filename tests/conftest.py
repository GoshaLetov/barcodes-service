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
    with open(TESTS_DIR / 'images' / 'barcode.jpg', 'rb') as image:
        yield image


@pytest.fixture(scope='function')
def fake_image_bytes():
    with open(TESTS_DIR / 'images' / 'fake.jpg', 'rb') as image:
        yield image


@pytest.fixture
def barcode_image_numpy():
    return cv2.cvtColor(
        src=cv2.imread(filename=str(TESTS_DIR / 'images' / 'barcode.jpg')),
        code=cv2.COLOR_BGR2RGB,
    )


@pytest.fixture
def fake_image_numpy():
    return cv2.cvtColor(
        src=cv2.imread(filename=str(TESTS_DIR / 'images' / 'fake.jpg')),
        code=cv2.COLOR_BGR2RGB,
    )


@pytest.fixture(scope='session')
def config():
    return Config.from_yaml(path=PROJECT_PATH / 'config.yaml')


@pytest.fixture
def planet_container(config):
    container = barcodes.Container()
    container.config.from_dict(options=config)
    return container


@pytest.fixture
def wired_planet_container(config):
    container = barcodes.Container()
    container.config.from_dict(config)
    container.wire([barcodes.routes])
    yield container
    container.unwire()


@pytest.fixture
def test_app(wired_planet_container):
    app = FastAPI()
    app.include_router(router=barcodes.router)
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
