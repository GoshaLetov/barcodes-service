from dependency_injector import containers, providers
from src.barcodes.services import ONNXBarCodesAnalyzer, ONNXBarCodeSegmentationModel, ONNXBarCodeOCRModel
from src.config import Config


class Container(containers.DeclarativeContainer):
    config: Config = providers.Configuration()

    barcodes_segmentation = providers.Singleton(
        ONNXBarCodeSegmentationModel,
        config=config.seg,
    )

    barcodes_ocr = providers.Singleton(
        ONNXBarCodeOCRModel,
        config=config.ocr,
    )

    analyzer = providers.Singleton(
        ONNXBarCodesAnalyzer,
        segmentation=barcodes_segmentation,
        ocr=barcodes_ocr,
    )
