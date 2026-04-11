import unittest
import os
import yaml
import numpy as np

# Настройка PYTHONPATH для тестов
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.planning.planner import UAVPlanner
from src.utils.logger import get_system_logger

class TestUAVPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.logger = get_system_logger()
        cls.config = {
            "mode": "adaptive",
            "grid_step_meters": 10,
            "coverage_area": [0, 0, 100, 100],
            "safe_altitude": 50,
            "priority_weights": {
                "frequent_detection": 1.5,
                "time_since_last_visit": 1.2
            }
        }

    def test_config_validation(self):
        """Тест 1: Валидация конфига (Zero Hardcode check)"""
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
        self.assertTrue(os.path.exists(cfg_path), "Файл default.yaml не найден")
        
        with open(cfg_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            
        self.assertIn('detector', full_config)
        self.assertIn('tracker', full_config)
        self.assertIn('planner', full_config)
        # Проверяем отсутствие хардкода
        self.assertGreater(len(full_config['detector']['classes_of_interest']), 0)
        
    def test_planner_route(self):
        """Тест 2: Корректность расчёта маршрута (Адаптивный навигатор)"""
        planner = UAVPlanner(self.config, self.logger)
        
        # Симулируем 0 кадров
        planner.update_state([], frame_w=1920, frame_h=1080)
        
        # Симулируем обнаружение объектов в центре
        mock_tracks = [{'bbox': [960, 540, 50, 50]}]
        for _ in range(5):
            planner.update_state(mock_tracks, frame_w=1920, frame_h=1080)
            
        # Удостоверимся, что приоритет ячейки в центре матрицы вырос!
        center_row, center_col = planner.grid_height // 2, planner.grid_width // 2
        self.assertGreater(planner.heatmap[center_row, center_col], 0.1)
        
        # Форсируем генерацию следующей адаптивной точки
        planner.current_wp_idx = 100
        next_wp = planner.get_next_waypoint()
        
        # Точка генерируется корректно из X, Y, Z
        self.assertEqual(len(next_wp), 3) 
        self.assertGreater(next_wp[0], 0)

    def test_tracking_stability_and_edge_cases(self):
        """Тест 3: Краевые случаи детектора (Битое/Пустое видео)"""
        from src.detection.factory import build_detector
        
        det_config = {
            "model_type": "yolo11",
            "weights_path": "yolo11n.pt", 
            "device": "cpu", 
            "confidence_threshold": 0.2
        }
        
        try:
            # Детектор не должен падать с OOM при инициализации
            detector = build_detector(det_config, self.logger)
            
            # Симулируем пустой черный (битый) кадр
            empty_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Тестируем тот самый баг Ultralytics, который мы пропатчили
            # Метод не должен упасть от Negative Dimension (-14)
            res = detector.track_frame(empty_frame, tracker_yaml_path="bytetrack.yaml")
            self.assertTrue(True, "Детектор успешно восстановился при битом кадре")
        except Exception as e:
            self.fail(f"Детекционно-трекинговый пайплайн упал (Crash) на сбойном кадре: {e}")

if __name__ == '__main__':
    unittest.main()
