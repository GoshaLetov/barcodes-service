import cv2
import numpy as np

from io import BytesIO
from PIL import Image
from http import HTTPStatus
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from dependency_injector.wiring import Provide, inject

from src.barcodes.services import BaseBarCodesAnalyzer
from src.barcodes.container import Container
from src.barcodes.schemas import BarCodeCredentialsList, BarCodeCredentials, BoundingBox

router = APIRouter(prefix='/barcodes', tags=['barcodes'])


def _bytes2image(image: UploadFile):

    content_type = image.content_type
    if content_type not in {'image/jpeg', 'image/png'}:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='Invalid file type')

    buffer = np.frombuffer(buffer=image.file.read(), dtype=np.uint8)

    return cv2.imdecode(buf=buffer, flags=cv2.IMREAD_COLOR)


@router.post(
    path='/extract',
    response_model=BarCodeCredentialsList,
    description='Run OCR Pipeline on given image and return credentials',
)
@inject
def predict(
    image: UploadFile = File(
        title='BarCodeImageInput',
        alias='image',
        description='Image for inference.',
    ),
    analyzer: BaseBarCodesAnalyzer = Depends(Provide[Container.analyzer]),
) -> BarCodeCredentialsList:
    return BarCodeCredentialsList(barcodes=[
        BarCodeCredentials(
            bbox=BoundingBox(**barcode.get('bbox', {})),
            value=barcode.get('value', ''),
        )
        for barcode in analyzer.inference(image=_bytes2image(image=image))
    ])


@router.post(
    path='/draw',
    description='Run OCR Pipeline on given image and return image with bboxes and texts',
)
@inject
def crop(
        image: UploadFile = File(
            title='BarCodeImageInput',
            alias='image',
            description='Image for inference.',
        ),
        analyzer: BaseBarCodesAnalyzer = Depends(Provide[Container.analyzer]),
) -> StreamingResponse:
    io = BytesIO()
    image = Image.fromarray(analyzer.draw(image=_bytes2image(image=image)))
    image.save(io, format='JPEG')
    io.seek(0)
    return StreamingResponse(content=io, media_type='image/jpeg')
