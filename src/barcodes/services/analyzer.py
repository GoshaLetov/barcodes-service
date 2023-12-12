import numpy as np
from abc import ABC, abstractmethod

from src.barcodes.services import BaseBarCodeOCRModel, BaseBarCodeSegmentationModel


class BaseBarCodesAnalyzer(ABC):

    @abstractmethod
    def inference(self, image: np.ndarray) -> list[dict[str, int]]:
        ...


class ONNXBarCodesAnalyzer(BaseBarCodesAnalyzer):
    def __init__(self, ocr: BaseBarCodeOCRModel, segmentation: BaseBarCodeSegmentationModel):
        self._ocr = ocr
        self._segmentation = segmentation

    def inference(self, image: np.ndarray) -> list[dict[str, int]]:
        bounding_boxes, barcodes = self._segmentation.extract_bounding_box(image=image), []
        for bounding_box in bounding_boxes:
            x_min = bounding_box.get('x_min')
            x_max = bounding_box.get('x_max')
            y_min = bounding_box.get('y_min')
            y_max = bounding_box.get('y_max')
            barcodes.append({
                'bbox': bounding_box,
                'value': self._ocr.extract_text(image=image[y_min:y_max, x_min:x_max])
            })
        return barcodes
