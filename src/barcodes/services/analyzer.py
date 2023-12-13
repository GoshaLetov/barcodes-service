import numpy as np
import cv2
from abc import ABC, abstractmethod

from src.barcodes.services import BaseBarCodeOCRModel, BaseBarCodeSegmentationModel


class BaseBarCodesAnalyzer(ABC):

    @abstractmethod
    def inference(self, image: np.ndarray) -> list[dict[str, int]]:
        ...

    @abstractmethod
    def draw(self, image: np.ndarray) -> np.ndarray:
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

            crop = image[x_min:x_max, y_min:y_max]

            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, rotateCode=2)

            value = ''.join(self._ocr.extract_text(image=crop))

            if len(value) >= 8:
                barcodes.append({'bbox': bounding_box, 'value': value})

        return barcodes

    def draw(self, image: np.ndarray) -> np.ndarray:
        image, barcodes = image.copy(), self.inference(image=image)
        for barcode in barcodes:
            x_min = barcode.get('bbox', {}).get('x_min')
            x_max = barcode.get('bbox', {}).get('x_max')
            y_min = barcode.get('bbox', {}).get('y_min')
            y_max = barcode.get('bbox', {}).get('y_max')
            value = ' '.join(barcode.get('value', ''))
            cv2.rectangle(image, pt1=[y_min, x_min], pt2=[y_max, x_max], color=[0, 255, 0], thickness=2)
            cv2.putText(image, text=value, org=[y_min, x_min - 30], fontFace=0, fontScale=1, color=[0, 255, 0])
        return image
