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
    """
    Запускает основной CLI-конвейер обработки видео с БПЛА.
    
    Выполняет инициализацию системных модулей (детектора, трекера, планировщика),
    последовательно обрабатывает кадры видеопотока и сохраняет результаты 
    (OSD-видео и JSONL-телеметрию).
    
    Args:
        args (argparse.Namespace): Аргументы командной строки, включая пути 
                                   к медиа, конфигурации, и флаги запуска.
                                   
    Returns:
        None
    """
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
    
    output_video_path = args.output_video if args.output_video is not None else config.get('pipeline', {}).get('output_video', 'data/output/result.mp4')
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    # Формат телеметрии берется из конфига
    telemetry_path = "data/output/telemetry." + config['pipeline']['telemetry_format']
    telemetry = TelemetryLogger(telemetry_path)
    
    # Инициализация модулей через Фабрики (Zero Hardcode & SOLID)
    detector = build_detector(config['detector'], sys_logger)
    tracker_cfg = UAVTrackerConfig(config['tracker'], sys_logger)
    planner = UAVPlanner(config['planner'], sys_logger)
    visualizer = Visualizer(sys_logger)
    
    # Загрузка видео или live-потока
    video_source_arg = args.video if args.video is not None else config.get('pipeline', {}).get('source_video', '0')
    video_source = int(video_source_arg) if str(video_source_arg).isdigit() else video_source_arg
    sys_logger.info(f"Медиа источник: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        sys_logger.error(f"Не удалось открыть видео {video_source}.")
        sys_logger.info("Пожалуйста, запустите скачивание: python data/download_test_video.py")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_id = 0
    sys_logger.info("Начало конвейерной обработки кадров...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # 1. Детекция и Трекинг (Изолировано через абстрактный интерфейс list[dict])
        tracks_data = detector.track_frame(frame, tracker_cfg.get_config_path())
        
        # --- Оптимизация: Фильтрация объектов по зоне ВПП (ROI) ---
        roi_cfg = config['detector'].get('runway_roi', None)
        if roi_cfg is not None:
            import numpy as np
            roi_poly = np.array([[int(p[0]*width), int(p[1]*height)] for p in roi_cfg], np.int32)
            filtered_tracks = []
            for tr in tracks_data:
                # Проверка попадания центра рамки в полигон ВПП
                if cv2.pointPolygonTest(roi_poly, (int(tr['bbox'][0]), int(tr['bbox'][1])), False) >= 0:
                    filtered_tracks.append(tr)
            tracks_data = filtered_tracks
        # ----------------------------------------------------------
        
        # 2. Навигатор: Планирование маршрута (Адаптивное обновление)
        planner.update_state(tracks_data, width, height)
        current_wp = planner.get_current_waypoint()
        next_wp = planner.get_next_waypoint()
        
        process_fps = 1.0 / (time.time() - start_time + 1e-6)
        
        # 4. Логирование телеметрии
        telemetry.log_state(frame_id, process_fps, tracks_data, current_wp, next_wp)
        
        # 5. Визуализация OSD
        class_names = getattr(detector, 'class_names', {})
        
        if args.no_osd:
            show_osd_flag = False
        else:
            show_osd_flag = config.get('ui', {}).get('show_osd_default', True)
            
        if args.no_target_wp:
            show_target_wp_flag = False
        else:
            show_target_wp_flag = config.get('ui', {}).get('show_target_wp_default', True)
            
        disp_frame = visualizer.draw(frame, tracks_data, current_wp, next_wp, process_fps, planner, class_names, roi_cfg, show_osd=show_osd_flag, show_target_wp=show_target_wp_flag)
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
    sys_logger.info(f"Обработка успешно завершена. Результат сохранен в {output_video_path}")
    sys_logger.info(f"Телеметрия записана в {telemetry_path}")


def run_pipeline_yield(video_source, config_overrides=None):
    """
    Генераторная версия конвейера для интеграции с Gradio Web UI.
    Возвращает обработанные кадры в формате RGB вместо показа в cv2.imshow().
    
    Эта функция изолирует состояние пайплайна для веб-сессий и адаптирует 
    вывод для потоковой трансляции в браузер.
    
    Args:
        video_source (str | int): Путь к медиафайлу, RSTP-потоку или индекс USB-камеры.
        config_overrides (dict, optional): Словарь переопределенных параметров 
            (веса, пороги, шаг сетки), заданных пользователем через UI. По умолчанию None.
            
    Yields:
        numpy.ndarray: Обработанный кадр (RGB) с нанесенной OSD-телеметрией для 
                       отображения потока реального времени в веб-интерфейсе.
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
        
        # --- Оптимизация: Фильтрация объектов по зоне ВПП (ROI) ---
        roi_cfg = config['detector'].get('runway_roi', None)
        if roi_cfg is not None:
            import numpy as np
            roi_poly = np.array([[int(p[0]*width), int(p[1]*height)] for p in roi_cfg], np.int32)
            filtered_tracks = []
            for tr in tracks_data:
                if cv2.pointPolygonTest(roi_poly, (int(tr['bbox'][0]), int(tr['bbox'][1])), False) >= 0:
                    filtered_tracks.append(tr)
            tracks_data = filtered_tracks
        # ----------------------------------------------------------
                    
        planner.update_state(tracks_data, width, height)
        process_fps = 1.0 / (time.time() - start_time + 1e-6)
        
        class_names = getattr(detector, 'class_names', {})
        show_osd_flag = config_overrides.get("show_osd", True) if config_overrides else True
        show_target_wp_flag = config_overrides.get("show_target_wp", True) if config_overrides else True
        disp_frame = visualizer.draw(frame, tracks_data, planner.get_current_waypoint(), planner.get_next_waypoint(), process_fps, planner, class_names, roi_cfg, show_osd=show_osd_flag, show_target_wp=show_target_wp_flag)
        
        frame_id += 1
        if frame_id % int(max(1, fps)) == 0:
            planner.current_wp_idx += 1
            
        # Gradio Image component expects RGB numpy arrays
        yield cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        
    tracker_cfg.cleanup()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Авиационный конвейер детекции БПЛА")
    parser.add_argument("--video", type=str, default=None, help="Путь к видео (по умолчанию из конфига)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Путь к YAML конфигу")
    parser.add_argument("--output_video", type=str, default=None, help="Путь сохранения результата (по умолчанию из конфига)")
    parser.add_argument("--mode", type=str, default=None, help="Режим (по умолчанию из конфига)")
    parser.add_argument("--tracker", type=str, default=None, choices=["bytetrack", "botsort"], help="Переопределить алгоритм трекера (по умолчанию из конфига)")
    parser.add_argument("--weights", type=str, default=None, help="Переопределить путь к весам модели (по умолчанию из конфига)")
    parser.add_argument("--headless", action="store_true", help="Обязательно для серверов/Colab: отключить показ окон")
    parser.add_argument("--no-osd", action="store_true", help="Отключить вывод телеметрии (FPS/Координаты) поверх видео")
    parser.add_argument("--no-target-wp", action="store_true", help="Отключить вывод прицела следующей точки маршрута (Target WP)")
    
    args = parser.parse_args()
    run(args)
