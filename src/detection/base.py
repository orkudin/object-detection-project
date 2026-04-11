from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Абстрактный класс детектора (Нарушение SOLID: Dependency Inversion).
    Конвейер зависит от этого интерфейса, а не от конкретных (YOLO/RKNN).
    """
    
    @abstractmethod
    def track_frame(self, frame, tracker_yaml_path: str) -> list[dict]:
        """
        Принимает:
        - frame: BGR numpy изображение
        - tracker_yaml_path: Путь к конфигурации трекера
        
        Возвращает:
        Список словарей формата `[{"track_id": 1, "class_id": 2, "bbox": [x, y, w, h]}, ...]`
        Это полностью изолирует конвейер от внутренних объектов библиотек.
        """
        pass
