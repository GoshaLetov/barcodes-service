import albumentations

from albumentations.pytorch import ToTensorV2

import numpy as np

from abc import ABC, abstractmethod
from cv2 import resize
from scipy.special import expit
from skimage.measure import label, regionprops
from onnxruntime import InferenceSession
from src.config import SEGConfig
from src.constants import SEG_MODEL_PATH


class BaseBarCodeSegmentationModel(ABC):

    @abstractmethod
    def extract_mask(self, image: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def extract_bounding_box(self, image: np.ndarray) -> list[dict[str, int]]:
        ...

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
            ToTensorV2(),
        ])

    def extract_mask(self, image: np.ndarray) -> np.ndarray:
        return self.inference(image=image)

    def extract_bounding_box(self, image: np.ndarray) -> list[dict[str, int]]:
        return [{
            'x_min': prop.bbox[0],
            'x_max': prop.bbox[2],
            'y_min': prop.bbox[1],
            'y_max': prop.bbox[3],
        } for prop in regionprops(label(self.inference(image=image)))]

    def inference(self, image: np.ndarray) -> np.ndarray:
        tensor = self._transform(image=image).get('image')

        mask = self._model.run(output_names=None, input_feed={'input': [tensor]})
        mask = expit(resize(src=mask[0][0, 0], dsize=[image.shape[1], image.shape[0]])) > 0.5

        return mask.astype(int)
