from ultralytics import YOLO
import torch
import gc

class UAVDetector:
    def __init__(self, config, logger):
        self.logger = logger
        self.device = config.get("device", "cpu")
        weights = config.get("weights_path", "yolo11s.pt")
        
        self.logger.info(f"YOLO11 Detector init: weights={weights}, device={self.device}")
        self.model = YOLO(weights)
        
        self.conf_thresh = config.get("confidence_threshold", 0.35)
        self.iou_thresh = config.get("iou_threshold", 0.45)
        self.classes = config.get("classes_of_interest", None)
        
    def track_frame(self, frame, tracker_yaml_path):
        """
        Выполняет детекцию и трекинг кадра. 
        При OOM освобождает GPU.
        """
        try:
            results = self.model.track(
                frame, 
                persist=True, 
                tracker=tracker_yaml_path,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
            return results
        except torch.cuda.OutOfMemoryError:
            self.logger.error("OOM в Detector. Очистка кэша GPU.")
            torch.cuda.empty_cache()
            gc.collect()
            return None
        except Exception as e:
            self.logger.error(f"Ошибка при обработке кадра: {e}")
            return None
