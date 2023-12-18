import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List
from src.barcodes.services import BaseBarCodeOCRModel, BaseBarCodeSegmentationModel
from src.barcodes.schemas import BarCodeCredentials


class BaseBarCodesAnalyzer(ABC):

    @abstractmethod
    def inference(self, image: np.ndarray) -> List[BarCodeCredentials]:
        ...

    @abstractmethod
    def draw(self, image: np.ndarray) -> np.ndarray:
        ...


class ONNXBarCodesAnalyzer(BaseBarCodesAnalyzer):
    def __init__(self, ocr: BaseBarCodeOCRModel, segmentation: BaseBarCodeSegmentationModel):
        self._ocr = ocr
        self._segmentation = segmentation

    def inference(self, image: np.ndarray) -> List[BarCodeCredentials]:
        bounding_boxes, barcodes = self._segmentation.extract_bounding_box(image=image), []
        for bounding_box in bounding_boxes:
            crop = image[bounding_box.x_min:bounding_box.x_max, bounding_box.y_min:bounding_box.y_max]  # noqa: WPS221
            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, rotateCode=2)
            barcodes.append(BarCodeCredentials(
                bbox=bounding_box,
                value=self._ocr.extract_text(image=crop),
            ))
        return barcodes

    def draw(self, image: np.ndarray) -> np.ndarray:
        image, barcodes = image.copy(), self.inference(image=image)
        for barcode in barcodes:
            cv2.rectangle(
                img=image,
                pt1=[barcode.bbox.y_min, barcode.bbox.x_min],
                pt2=[barcode.bbox.y_max, barcode.bbox.x_max],
                color=[0, 255, 0],
                thickness=2,
            )
            cv2.putText(
                img=image,
                text=barcode.value,
                org=[barcode.bbox.y_min, barcode.bbox.x_min - 30],
                fontFace=0,
                fontScale=1,
                color=[0, 255, 0],
            )
        return image
