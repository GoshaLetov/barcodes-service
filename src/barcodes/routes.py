from io import BytesIO
from PIL import Image
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse, Response
from dependency_injector.wiring import Provide, inject
from src.io import bytes2image
from src.barcodes.services import BaseBarCodesAnalyzer
from src.barcodes.container import Container
from src.barcodes.schemas import BarCodeCredentialsList, BarCodeCredentials, BoundingBox

router = APIRouter(prefix='/barcodes', tags=['barcodes'])


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
        for barcode in analyzer.inference(image=bytes2image(image=image))
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
    image = Image.fromarray(analyzer.draw(image=bytes2image(image=image)))
    image.save(io, format='JPEG')
    io.seek(0)
    return StreamingResponse(content=io, media_type='image/jpeg')


@router.get('/health')
def health():
    return Response(content='OK')
