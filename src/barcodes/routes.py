from io import BytesIO
from PIL import Image
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse, Response
from dependency_injector.wiring import Provide, inject
from src.io import bytes2image
from src.barcodes.services import BaseBarCodesAnalyzer
from src.barcodes.container import Container
from src.barcodes.schemas import BarCodeCredentials, BoundingBox

router = APIRouter(prefix='/barcodes', tags=['barcodes'])


@router.post(
    path='/extract',
    response_model=List[BarCodeCredentials],
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
) -> List[BarCodeCredentials]:
    return analyzer.inference(image=bytes2image(image=image))


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
    image = Image.fromarray(analyzer.draw(image=bytes2image(image=image)))
    image.save(io, format='JPEG')
    io.seek(0)
    return StreamingResponse(content=io, media_type='image/jpeg')


@router.get('/health')
def health():
    return Response(content='OK')
