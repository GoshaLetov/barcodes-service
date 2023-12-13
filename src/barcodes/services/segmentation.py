import albumentations
import numpy as np
from abc import ABC, abstractmethod
from cv2 import resize
from scipy.special import expit
from skimage.measure import label, regionprops
from onnxruntime import InferenceSession
from src.config import SEGConfig
from src.constants import SEG_MODEL_PATH, SEG_MIN_AREA


class BaseBarCodeSegmentationModel(ABC):

    @abstractmethod
    def extract_bounding_box(self, image: np.ndarray) -> list[dict[str, int]]:
        ...

    @abstractmethod
    def inference(self, image: np.ndarray) -> np.ndarray:
        ...


class ONNXBarCodeSegmentationModel(BaseBarCodeSegmentationModel):

    def __init__(self, config: SEGConfig):
        self._config = config

        self._model = InferenceSession(
            path_or_bytes=SEG_MODEL_PATH / config.onnx,
            providers=[config.provider],
        )

        self._transform = albumentations.Compose([
            albumentations.Resize(width=config.width, height=config.width),
            albumentations.Normalize(),
        ])

    def extract_bounding_box(self, image: np.ndarray) -> list[dict[str, int]]:
        bounding_boxes = []
        for prop in regionprops(label(self.inference(image=image))):
            x_min = prop.bbox[0]
            x_max = prop.bbox[2]
            y_min = prop.bbox[1]
            y_max = prop.bbox[3]

            if (y_max - y_min) * (x_max - x_min) >= SEG_MIN_AREA:
                bounding_boxes.append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                })
        return bounding_boxes

    def inference(self, image: np.ndarray) -> np.ndarray:
        tensor = self._transform(image=image).get('image').transpose(2, 0, 1)

        mask = self._model.run(output_names=None, input_feed={'input': [tensor]})
        mask = expit(resize(src=mask[0][0, 0], dsize=[image.shape[1], image.shape[0]])) > 0.5

        return mask.astype(int)
