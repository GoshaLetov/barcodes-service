from omegaconf import OmegaConf
from pydantic import BaseModel


class OCRConfig(BaseModel):
    onnx: str
    provider: str
    width: int
    height: int
    vocab: str
    text_size: int


class SEGConfig(BaseModel):
    onnx: str
    provider: str
    width: int
    height: int
    threshold: float


class Config(BaseModel):
    ocr: OCRConfig
    seg: SEGConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
