import logging

def get_system_logger(name="UAV_System"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

import json
import os

class TelemetryLogger:
    def __init__(self, output_path="data/output/telemetry.jsonl"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # Очищаем или создаем файл
        open(self.output_path, 'w').close()
        
    def log_state(self, frame_id, fps, tracks, current_wp, next_wp):
        """
        tracks: список словарей {'track_id': id, 'class_id': cls, 'bbox': [x, y, w, h]}
        """
        payload = {
            "frame_id": frame_id,
            "fps": round(fps, 2),
            "detections": tracks,
            "current_waypoint": current_wp,
            "next_planned_waypoint": next_wp
        }
        with open(self.output_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
