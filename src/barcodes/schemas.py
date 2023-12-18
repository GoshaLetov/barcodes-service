from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class BarCodeCredentials(BaseModel):
    bbox: BoundingBox
    value: str
