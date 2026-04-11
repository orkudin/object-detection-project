from ultralytics import YOLO
import torch
import gc
from .base import BaseDetector

class YOLODetector(BaseDetector):
    """
    Стандартный детектор на базе PyTorch / Ultralytics (для ПК: CPU/GPU).
    """
    def __init__(self, config, logger):
        self.logger = logger
        self.device = config.get("device", "cpu")
        weights = config.get("weights_path", "yolo11s.pt")
        
        self.logger.info(f"YOLO Detector init: weights={weights}, device={self.device}")
        # Явное указание task='detect' спасает от сбоев парсинга масок (Segmentation) 
        # при работе с кастомными ONNX графами
        self.model = YOLO(weights, task="detect")
        
        self.conf_thresh = config.get("confidence_threshold", 0.35)
        self.iou_thresh = config.get("iou_threshold", 0.45)
        self.classes = config.get("classes_of_interest", None)
        
    def track_frame(self, frame, tracker_yaml_path: str) -> list[dict]:
        """
        Возвращает стандартизированный список [ {track_id, class_id, bbox} ].
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
            
            tracks_data = []
            if results and len(results) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                cls = results[0].boxes.cls.int().cpu().numpy()
                
                for b, t_id, c_id in zip(boxes, ids, cls):
                    if self.classes is None or int(c_id) in self.classes:
                        tracks_data.append({
                            "track_id": int(t_id), 
                            "class_id": int(c_id), 
                            "bbox": b.tolist()
                        })
            return tracks_data
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("OOM в YOLODetector. Очистка кэша GPU.")
            torch.cuda.empty_cache()
            gc.collect()
            return []
        except Exception as e:
            self.logger.error(f"Ошибка при обработке кадра YOLODetector: {e}")
            return []
