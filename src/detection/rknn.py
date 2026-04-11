import cv2
import numpy as np
from .base import BaseDetector

class RKNNDetector(BaseDetector):
    """
    Аппаратный детектор для NPU Rockchip (Radxa 4D / RK3576).
    """
    def __init__(self, config, logger):
        self.logger = logger
        self.weights = config.get("weights_path", "yolo11s.rknn")
        self.conf_thresh = config.get("confidence_threshold", 0.35)
        self.classes = config.get("classes_of_interest", None)
        
        # Динамический импорт для совместимости (чтобы код на ПК не падал без rknnlite)
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            self.logger.info(f"Инициализация RKNNLite на Rockchip NPU. Загрузка: {self.weights}")
            
            ret = self.rknn.load_rknn(self.weights)
            if ret != 0:
                self.logger.error("Ошибка загрузки RKNN модели!")
                
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                self.logger.error("Ошибка инициализации RKNN NPU Runtime!")
                
        except ImportError:
            self.logger.warning("Библиотека `rknnlite` не найдена! RKNN Инференс недоступен на архитектуре x86/Windows.")
            self.rknn = None
            
        # Так как RKNN не умеет сам трекать, мы загрузим программный трекер Bytetrack
        from ultralytics.trackers.byte_tracker import BYTETracker
        # Имитируем аргументы, нужные для ByteTracker
        from types import SimpleNamespace
        tracker_args = SimpleNamespace(
            track_high_thresh=0.5, track_low_thresh=0.1, 
            new_track_thresh=0.6, track_buffer=30, match_thresh=0.8,
            fuse_score=True
        )
        self.tracker = BYTETracker(tracker_args)
        self.logger.info("RKNN-Детектор успешно скомпонован с программным ByteTracker на CPU.")

    def _preprocess(self, frame):
        # YOLO требует 640x640 RGB (в зависимости от экспорта)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        # RKNN обычно ожидает HWC uint8 если включен pre_compile
        return img

    def _postprocess(self, outputs, orig_w, orig_h):
        """
        Упрощенный парсинг вывода NPU. 
        Зависит от того, включен ли NMS при конвертации в .rknn!
        """
        # Пока возвращаем заглушку, чтобы проект не падал и архитектура была Solid.
        # Для реальной работы сюда вставляется yolo_postprocess (NMS + Anchor decode)
        return []

    def track_frame(self, frame, tracker_yaml_path: str) -> list[dict]:
        """
        Реализация интерфейса абстрактного Детектора.
        1. Прогоняет кадр через NPU (RKNN)
        2. Прогоняет BBox через CPU (ByteTracker)
        """
        orig_h, orig_w = frame.shape[:2]
        
        if self.rknn is None:
            # Fallback (Отказоустойчивость), если запускают .rknn на ПК
            self.logger.debug("Пропуск кадра: RKNN не поддерживается на текущей ОС.")
            return []
            
        # 1. NPU Inference
        img = self._preprocess(frame)
        outputs = self.rknn.inference(inputs=[img])
        
        # 2. Декодирование тензоров NPU в numpy Bounding Boxes
        raw_detections = self._postprocess(outputs, orig_w, orig_h)
        
        if len(raw_detections) == 0:
            return []
            
        # 3. CPU Tracking
        # Формат для BYTETracker: np.ndarray [x1, y1, x2, y2, conf, cls]
        # tracks = self.tracker.update(np.array(raw_detections), frame)
        
        # 4. Форматирование результата согласно Архитектуре
        tracks_data = [] # ...
        
        return tracks_data
