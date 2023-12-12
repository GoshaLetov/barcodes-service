import albumentations
import torch
import cv2
import itertools
import operator

from typing import List, Union, Dict
from albumentations.pytorch import ToTensorV2

import numpy as np

from abc import ABC, abstractmethod
from scipy.special import softmax
from onnxruntime import InferenceSession
from src.config import OCRConfig
from src.constants import OCR_MODEL_PATH


class BaseBarCodeOCRModel(ABC):

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> np.ndarray:
        ...

    def inference(self, image: np.ndarray) -> torch.Tensor:
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
            ToTensorV2(),
        ])
        self._vocab = config.vocab

    def extract_text(self, image: np.ndarray) -> list[str]:
        labels, _ = self._decode(*self.inference(image=image))
        return self._labels_to_strings(labels=labels, vocab=self._vocab)

    def inference(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = self._transform(image=image, text='').get('image')
        logits = self._model.run(output_names=None, input_feed={'input': [tensor]})
        probas = softmax(x=logits[0].transpose(1, 0, 2))
        return probas.argmax(axis=2), probas.max(axis=2)

    @staticmethod
    def _decode(labels: np.ndarray, confidences: np.ndarray) -> tuple[list[list[int]], list[np.ndarray]]:
        result_labels, result_confidences = [], []
        for label, confidence in zip(labels, confidences):
            result_one_labels, result_one_confidences = [], []
            for l, group in itertools.groupby(zip(label, confidence), operator.itemgetter(0)):
                if l > 0:
                    result_one_labels.append(l)
                    result_one_confidences.append(max(list(zip(*group))[1]))
            result_labels.append(result_one_labels)
            result_confidences.append(np.array(result_one_confidences))
        return result_labels, result_confidences

    @staticmethod
    def _labels_to_strings(labels: list[list[int]], vocab: str) -> list[str]:
        strings = []
        for single_str_labels in labels:
            try:
                output_str = ''.join(
                    vocab[char_index - 1] if char_index > 0 else '_' for char_index in single_str_labels)
                strings.append(output_str)
            except IndexError:
                strings.append('Error')
        return strings


class PadResizeOCR(albumentations.BasicTransform):
    """
    Приводит к нужному размеру с сохранением отношения сторон, если нужно добавляет падинги.
    """

    def __init__(self, target_width, target_height, value: int = 0, mode: str = 'random'):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        self.value = value
        self.mode = mode

        assert self.mode in {'random', 'left', 'center'}

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        image = kwargs['image'].copy()

        h, w = image.shape[:2]

        tmp_w = min(int(w * (self.target_height / h)), self.target_width)
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

            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=0)

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

        processed_text = [self.vocab.index(x) + 1 for x in source_text if x in self.vocab]
        processed_text = np.pad(
            processed_text,
            (0, self.target_text_size - len(processed_text)),
            mode='constant',
        )
        processed_text = torch.IntTensor(processed_text)

        kwargs['text'] = processed_text

        return kwargs
