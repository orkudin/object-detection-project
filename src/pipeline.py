import argparse
import yaml
import cv2
import time
import os

from utils.logger import get_system_logger, TelemetryLogger
from detection.factory import build_detector
from tracking.tracker import UAVTrackerConfig
from planning.planner import UAVPlanner
from utils.visualizer import Visualizer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run(args):
    config = load_config(args.config)
    
    # Инициализация системного логгера
    sys_logger = get_system_logger()
    sys_logger.info(f"Запуск конвейера БПЛА. Конфиг: {args.config}")
    
    # Переопределение трекера из аргументов командной строки, если задано
    if args.tracker:
        config['tracker']['algorithm'] = args.tracker
        sys_logger.info(f"Алгоритм трекера переопределен пользователем: {args.tracker}")
        
    # Переопределение весов модели, если задано
    if args.weights:
        config['detector']['weights_path'] = args.weights
        sys_logger.info(f"Веса детектора переопределены пользователем: {args.weights}")
    
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    # Формат телеметрии берется из конфига
    telemetry_path = "data/output/telemetry." + config['pipeline']['telemetry_format']
    telemetry = TelemetryLogger(telemetry_path)
    
    # Инициализация модулей через Фабрики (Zero Hardcode & SOLID)
    detector = build_detector(config['detector'], sys_logger)
    tracker_cfg = UAVTrackerConfig(config['tracker'], sys_logger)
    planner = UAVPlanner(config['planner'], sys_logger)
    visualizer = Visualizer(sys_logger)
    
    # Загрузка видео или live-потока
    video_source = int(args.video) if args.video.isdigit() else args.video
    sys_logger.info(f"Медиа источник: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        sys_logger.error(f"Не удалось открыть видео {args.video}.")
        sys_logger.info("Пожалуйста, запустите скачивание: python data/download_test_video.py")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    frame_id = 0
    sys_logger.info("Начало конвейерной обработки кадров...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # 1. Детекция и Трекинг (Изолировано через абстрактный интерфейс list[dict])
        tracks_data = detector.track_frame(frame, tracker_cfg.get_config_path())
        
        # 2. Навигатор: Планирование маршрута (Адаптивное обновление)
        planner.update_state(tracks_data, width, height)
        current_wp = planner.get_current_waypoint()
        next_wp = planner.get_next_waypoint()
        
        process_fps = 1.0 / (time.time() - start_time + 1e-6)
        
        # 4. Логирование телеметрии
        telemetry.log_state(frame_id, process_fps, tracks_data, current_wp, next_wp)
        
        # 5. Визуализация OSD
        disp_frame = visualizer.draw(frame, tracks_data, current_wp, next_wp, process_fps, planner)
        out.write(disp_frame)
        
        # Вывод на экран в реальном времени (если запущен локально)
        if not args.headless:
            try:
                cv2.imshow("UAV-CV Navigator [Live]", disp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    sys_logger.info("Вызвано ручное прерывание сеанса (q). Остановка конвейера.")
                    break
            except Exception:
                # Игнорируем ошибку отрисовки
                pass
        
        frame_id += 1
        # Имитация продвижения БПЛА по маршруту каждую секунду
        if frame_id % int(fps) == 0:
            planner.current_wp_idx += 1
            
    # Уборка мусора и временных файлов
    tracker_cfg.cleanup()
    cap.release()
    out.release()
    if not args.headless:
        cv2.destroyAllWindows()
    sys_logger.info(f"Обработка успешно завершена. Результат сохранен в {args.output_video}")
    sys_logger.info(f"Телеметрия записана в {telemetry_path}")


def run_pipeline_yield(video_source, config_overrides=None):
    """
    Генераторная версия конвейера для интеграции с Gradio Web UI.
    Возвращает обработанные кадры в формате RGB вместо показа в cv2.imshow().
    """
    # Поддержка вызова из разных директорий
    import os
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'default.yaml')
    config = load_config(cfg_path)
    if config_overrides:
        if "weights" in config_overrides: config['detector']['weights_path'] = config_overrides['weights']
        if "tracker" in config_overrides: config['tracker']['algorithm'] = config_overrides['tracker']
        if "conf" in config_overrides: config['detector']['confidence_threshold'] = config_overrides['conf']
        if "grid_step" in config_overrides: config['planner']['grid_step_meters'] = config_overrides['grid_step']

    sys_logger = get_system_logger()
    sys_logger.info(f"Запуск Web UI конвейера. Источник: {video_source}")
    
    # Инициализация через Фабрику (чтобы избежать OOM при переключении моделей веб-интерфейсом)
    detector = build_detector(config['detector'], sys_logger)
    tracker_cfg = UAVTrackerConfig(config['tracker'], sys_logger)
    planner = UAVPlanner(config['planner'], sys_logger)
    visualizer = Visualizer(sys_logger)
    
    # Обработка источников
    src = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        sys_logger.error("Не удалось прочитать видео поток в Web UI.")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        # Архитектурно-независимая обработка кадра
        tracks_data = detector.track_frame(frame, tracker_cfg.get_config_path())
                    
        planner.update_state(tracks_data, width, height)
        process_fps = 1.0 / (time.time() - start_time + 1e-6)
        
        disp_frame = visualizer.draw(frame, tracks_data, planner.get_current_waypoint(), planner.get_next_waypoint(), process_fps, planner)
        
        frame_id += 1
        if frame_id % int(max(1, fps)) == 0:
            planner.current_wp_idx += 1
            
        # Gradio Image component expects RGB numpy arrays
        yield cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        
    tracker_cfg.cleanup()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Авиационный конвейер детекции БПЛА")
    parser.add_argument("--video", type=str, default="data/test_video_2.mp4", help="Путь к видео")
    parser.add_argument("--config", type=str, default="configs/vision_pipeline.yaml", help="Путь к YAML конфигу")
    parser.add_argument("--output_video", type=str, default="data/output/result.mp4", help="Путь сохранения результата")
    parser.add_argument("--mode", type=str, default="full", help="Режим (full)")
    parser.add_argument("--tracker", type=str, default=None, choices=["bytetrack", "botsort"], help="Переопределить алгоритм трекера (по умолчанию из конфига)")
    parser.add_argument("--weights", type=str, default=None, help="Переопределить путь к весам модели (по умолчанию из конфига)")
    parser.add_argument("--headless", action="store_true", help="Обязательно для серверов/Colab: отключить показ окон")
    
    args = parser.parse_args()
    run(args)
