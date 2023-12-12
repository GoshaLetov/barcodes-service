import uvicorn

from fastapi import FastAPI
from src import barcodes
from src.config import Config
from src.constants import PROJECT_PATH


def main() -> FastAPI:
    app = FastAPI()
    config = Config.from_yaml(path=PROJECT_PATH / 'config.yaml')

    container = barcodes.Container()
    container.config.from_dict(options=config)
    container.wire(modules=[barcodes.routes])

    app.include_router(router=barcodes.router)

    return app


if __name__ == '__main__':
    uvicorn.run(main())
