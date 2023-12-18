import albumentations
import cv2
import itertools
import operator
import numpy as np

from typing import List, Union, Dict
from abc import ABC, abstractmethod
from scipy.special import softmax
from onnxruntime import InferenceSession
from src.config import OCRConfig
from src.constants import OCR_MODEL_PATH


class BaseBarCodeOCRModel(ABC):

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> str:
        ...

    @abstractmethod
    def inference(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class ONNXBarCodeOCRModel(BaseBarCodeOCRModel):
    def __init__(self, config: OCRConfig):
        self._config = config

        self._model = InferenceSession(path_or_bytes=OCR_MODEL_PATH / config.onnx)
        self._transform = albumentations.Compose([
            PadResizeOCR(
                target_height=config.height,
                target_width=config.width,
                mode='left',
            ),
            albumentations.Normalize(),
            TextEncode(vocab=config.vocab, target_text_size=config.text_size),
        ])
        self._vocab = config.vocab

    def extract_text(self, image: np.ndarray) -> str:
        labels, _ = _decode(*self.inference(image=image))
        return _labels_to_strings(labels=labels, vocab=self._vocab)

    def inference(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = self._transform(
            image=image,
            text='',
        ).get('image').transpose(2, 0, 1)
        logits = self._model.run(output_names=None, input_feed={'input': [tensor]})
        probas = softmax(x=logits[0].transpose(1, 0, 2))
        return probas.argmax(axis=2).ravel(), probas.max(axis=2).ravel()  # noqa: WPS221


class PadResizeOCR(albumentations.BasicTransform):
    """
    Приводит к нужному размеру с сохранением отношения сторон, добавляет padding если нужно.
    """

    def __init__(self, target_width, target_height, value: int = 0, mode: str = 'random'):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        self.value = value
        self.mode = mode

        assert self.mode in {'random', 'left', 'center'}

    def __call__(self, force_apply: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        image = kwargs['image'].copy()

        tmp_w = min(int(image.shape[1] * (self.target_height / image.shape[0])), self.target_width)  # noqa: WPS221
        image = cv2.resize(image, (tmp_w, self.target_height))

        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            if self.mode == 'random':
                pad_left = np.random.randint(dw)
            elif self.mode == 'left':
                pad_left = 0
            else:
                pad_left = dw // 2

            pad_right = dw - pad_left

            image = cv2.copyMakeBorder(
                src=image,
                top=0,
                bottom=0,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, ),
            )

        kwargs['image'] = image
        return kwargs


class TextEncode(albumentations.BasicTransform):
    """
    Кодирует исходный текст.
    """

    def __init__(self, vocab: Union[str, List[str]], target_text_size: int):
        super().__init__()
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.target_text_size = target_text_size

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        source_text = kwargs['text'].strip()

        processed_text = [
            self.vocab.index(char) + 1
            for char in source_text if char in self.vocab
        ]
        pad_width = 0, self.target_text_size - len(processed_text)
        processed_text = np.pad(processed_text, pad_width, mode='constant')

        kwargs['text'] = processed_text.astype(int)

        return kwargs


def _decode(
    labels: np.ndarray,
    confidences: np.ndarray,
) -> tuple[list[int], list[float]]:
    result_labels, result_confidences = [], []
    for label, confidence in itertools.groupby(zip(labels, confidences), operator.itemgetter(0)):  # noqa: WPS221
        if label > 0:
            result_labels.append(label)
            result_confidences.append(max(list(zip(*confidence))[1]))
    return result_labels, result_confidences


def _labels_to_strings(labels: list[int], vocab: str) -> str:
    return ''.join(
        vocab[label - 1] if label > 0 else '_'
        for label in labels
    )
