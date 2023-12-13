import cv2
import numpy as np
from http import HTTPStatus
from fastapi import UploadFile, HTTPException


def bytes2image(image: UploadFile):

    content_type = image.content_type
    if content_type not in {'image/jpeg', 'image/png'}:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='Invalid file type')

    buffer = np.frombuffer(buffer=image.file.read(), dtype=np.uint8)

    return cv2.cvtColor(cv2.imdecode(buf=buffer, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
