import os
import yaml
import tempfile

class UAVTrackerConfig:
    """
    Класс для динамической генерации конфигурации трекера
    в соответствии с ZERO HARDCODE POLICY.
    Ultralytics принимает yaml конфиги для трекеров (bytetrack, botsort).
    """
    def __init__(self, config, logger):
        self.logger = logger
        
        # Динамическое чтение всех параметров трекера из конфига
        self.tracker_cfg = config.copy()
        
        # Трансляция ключа 'algorithm' в формат, ожидаемый Ultralytics
        if "algorithm" in self.tracker_cfg:
            self.tracker_cfg["tracker_type"] = self.tracker_cfg.pop("algorithm")
            
        # Гарантируем наличие параметра fuse_score для новейших версий YOLO11
        if "fuse_score" not in self.tracker_cfg:
            self.tracker_cfg["fuse_score"] = True
            
        # Если выбран BoT-SORT, подгружаем базовые параметры, 
        # если пользователь не указал их явно в vision_pipeline.yaml
        if self.tracker_cfg.get("tracker_type") == "botsort":
            self.tracker_cfg.setdefault("gmc_method", "sparseOptFlow")
            self.tracker_cfg.setdefault("proximity_thresh", 0.5)
            self.tracker_cfg.setdefault("appearance_thresh", 0.25)
            self.tracker_cfg.setdefault("with_reid", False)
        else:
            self.tracker_cfg.setdefault("gsi_th", 0.5)
        
        # Генерация временного конфигурационного файла
        fd, self.temp_yaml = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, 'w') as f:
            yaml.dump(self.tracker_cfg, f)
            
        self.logger.info(f"Трекер успешно сконфигурирован (Алгоритм: {self.tracker_cfg['tracker_type']}). Временный файл: {self.temp_yaml}")

    def get_config_path(self):
        return self.temp_yaml
    
    def cleanup(self):
        if os.path.exists(self.temp_yaml):
            try:
                os.remove(self.temp_yaml)
            except OSError:
                pass
