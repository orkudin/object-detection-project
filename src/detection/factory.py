import logging
import os
from .yolo import YOLODetector
from .rknn import RKNNDetector

def build_detector(config: dict, logger: logging.Logger):
    """
    Фабрика детекторов. 
    Гарантирует SOLID: Приложение не обязано знать, какой конкретно код запускается,
    оно лишь опирается на контракт BaseDetector.
    """
    weights = config.get("weights_path", "")
    
    # Автоопределение NPU по расширению (Rockchip NPU)
    if weights.endswith(".rknn") or config.get("model_type") == "rknn":
        logger.info("Фабрика выбрала RKNNDetector (Edge NPU Mode)")
        return RKNNDetector(config, logger)
        
    # По умолчанию используем PyTorch/YOLO (GPU/CPU Mode)
    logger.info("Фабрика выбрала YOLODetector (PC GPU/CPU Mode)")
    return YOLODetector(config, logger)
