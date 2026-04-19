from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Абстрактный класс детектора (Нарушение SOLID: Dependency Inversion).
    Конвейер зависит от этого интерфейса, а не от конкретных (YOLO/RKNN).
    """
    
    @abstractmethod
    def track_frame(self, frame, tracker_yaml_path: str) -> list[dict]:
        """
        Выполняет инференс на одном кадре для обнаружения и отслеживания локализованных объектов.
        
        Абстрактный метод, обязывающий все конкретные классы детекторов (YOLO, RKNN, и др.)
        реализовывать единый интерфейс взаимодействия для соблюдения принципов SOLID (DIP).
        
        Args:
            frame (numpy.ndarray): Кадр видеопотока в формате BGR.
            tracker_yaml_path (str): Путь к файлу конфигурации системы мультитрекинга.
        
        Returns:
            list[dict]: Стандартизированный список обнаруженных объектов. 
                Формат: `{"track_id": int, "class_id": int, "bbox": [x_center, y_center, width, height]}`.
        """
        pass

    @property
    def class_names(self) -> dict:
        """Возвращает словарь классов (ID -> Название), если поддерживается моделью."""
        return {}
