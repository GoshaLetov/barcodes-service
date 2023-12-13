import numpy as np

from src.barcodes.container import Container
from src.barcodes.services import BaseBarCodeOCRModel, BaseBarCodeSegmentationModel


class FakeBarCodeSegmentationModel(BaseBarCodeSegmentationModel):

    def __init__(self, empty_output: bool = False):
        self.empty_output = empty_output

    def inference(self, image: np.ndarray) -> np.ndarray:
        return np.ndarray(shape=[1, 1, 1], dtype=np.uint8)

    def extract_bounding_box(self, image: np.ndarray) -> list[dict[str, int]]:
        if self.empty_output:
            return []
        return [{'x_min': 1, 'x_max': 1, 'y_min': 1, 'y_max': 1}]


class FakeBarCodeOCRModel(BaseBarCodeOCRModel):

    def __init__(self, empty_output: bool = False):
        self.empty_output = empty_output

    def inference(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.ndarray(shape=[1, 1], dtype=np.uint8), np.ndarray(shape=[1, 1], dtype=np.float16)

    def extract_text(self, image: np.ndarray) -> str:
        if self.empty_output:
            return ''
        return '111111111111'


def test_not_fail(barcodes_container: Container, barcode_image_numpy: np.ndarray):
    """ all methods raw check """
    with barcodes_container.reset_singletons():
        with (
            barcodes_container.barcodes_segmentation.override(FakeBarCodeSegmentationModel()),
            barcodes_container.barcodes_ocr.override(FakeBarCodeOCRModel()),
        ):
            analyzer = barcodes_container.analyzer()
            analyzer.inference(image=barcode_image_numpy)
            analyzer.draw(image=barcode_image_numpy)


def test_inference_values(barcodes_container: Container, barcode_image_numpy: np.ndarray):
    """ pre-defined output check """
    with barcodes_container.reset_singletons():
        with (
            barcodes_container.barcodes_segmentation.override(FakeBarCodeSegmentationModel()),
            barcodes_container.barcodes_ocr.override(FakeBarCodeOCRModel()),
        ):
            analyzer = barcodes_container.analyzer()
            barcodes = analyzer.inference(image=barcode_image_numpy)

            assert isinstance(barcodes, list)

            barcode = barcodes[0]
            assert barcode.get('bbox') == {'x_min': 1, 'x_max': 1, 'y_min': 1, 'y_max': 1}
            assert barcode.get('value') == '111111111111'


def test_draw_do_not_mutate_fake(barcodes_container: Container, fake_image_numpy: np.ndarray):
    """ draw do not mutate if no barcodes on image """
    with barcodes_container.reset_singletons():
        with (
            barcodes_container.barcodes_segmentation.override(FakeBarCodeSegmentationModel(empty_output=True)),
            barcodes_container.barcodes_ocr.override(FakeBarCodeOCRModel(empty_output=True)),
        ):
            analyzer = barcodes_container.analyzer()

            image = analyzer.draw(image=fake_image_numpy)
            assert np.allclose(fake_image_numpy, image)


def test_draw_do_mutate_barcode(barcodes_container: Container, barcode_image_numpy: np.ndarray):
    """ draw do mutate if barcode on image """
    with barcodes_container.reset_singletons():
        with (
            barcodes_container.barcodes_segmentation.override(FakeBarCodeSegmentationModel()),
            barcodes_container.barcodes_ocr.override(FakeBarCodeOCRModel()),
        ):
            analyzer = barcodes_container.analyzer()

            image = analyzer.draw(image=barcode_image_numpy)
            assert not np.allclose(barcode_image_numpy, image)
