import numpy as np

from src.barcodes.container import Container


def test_not_fail(barcodes_container: Container, barcode_image_numpy: np.ndarray):
    """ all methods raw check """
    segmentation_model = barcodes_container.barcodes_segmentation()
    segmentation_model.inference(image=barcode_image_numpy)
    segmentation_model.extract_bounding_box(image=barcode_image_numpy)


def test_inference(barcodes_container: Container, barcode_image_numpy: np.ndarray):
    """ mask correct shape """
    segmentation_model = barcodes_container.barcodes_segmentation()
    mask = segmentation_model.inference(image=barcode_image_numpy)

    assert barcode_image_numpy.shape[:2] == mask.shape


def test_bounding_box(barcodes_container: Container, barcode_image_numpy: np.ndarray):
    """ bounding box output """
    segmentation_model = barcodes_container.barcodes_segmentation()

    bounding_boxes = segmentation_model.extract_bounding_box(image=barcode_image_numpy)
    assert isinstance(bounding_boxes, list)

    bounding_box = bounding_boxes[0]
    assert isinstance(bounding_box, dict)

    assert sorted(list(bounding_box.keys())) == sorted(['x_min', 'x_max', 'y_min', 'y_max'])
    assert all([isinstance(value, int) for value in bounding_box.values()])


def test_fake_image(barcodes_container: Container, fake_image_numpy: np.ndarray):
    """ simple fake image false positive check """
    segmentation_model = barcodes_container.barcodes_segmentation()
    bounding_boxes = segmentation_model.extract_bounding_box(image=fake_image_numpy)
    assert bounding_boxes == []


def test_two_barcodes(barcodes_container: Container, barcode_two_image_numpy: np.ndarray):
    """ several barcode on one image """
    segmentation_model = barcodes_container.barcodes_segmentation()

    bounding_boxes = segmentation_model.extract_bounding_box(image=barcode_two_image_numpy)
    assert len(bounding_boxes) > 1
