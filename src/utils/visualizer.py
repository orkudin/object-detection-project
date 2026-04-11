import cv2

class Visualizer:
    def __init__(self, logger):
        self.logger = logger
        
    def draw(self, frame, tracks, current_wp, next_wp, fps, planner=None):
        disp = frame.copy()
        h, w = disp.shape[:2]
        
        # Инфо панель
        cv2.rectangle(disp, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.putText(disp, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        mode_text = planner.mode.upper() if planner else "UNKNOWN"
        cv2.putText(disp, f"MODE: {mode_text}", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255) if mode_text == 'ADAPTIVE' else (255, 255, 255), 2)
        
        cv2.putText(disp, f"Cur WP: {current_wp}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(disp, f"Next WP: {next_wp}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Отрисовка работы НАВИГАТОРА (прицел следующей точки на экране)
        if planner is not None and next_wp is not None:
            nx, ny, _ = next_wp
            c_xmin, c_ymin, c_xmax, c_ymax = planner.coverage_area
            # Обратная проекция мировых координат в пиксели экрана
            px = int(w * (nx - c_xmin) / (c_xmax - c_xmin))
            py = int(h * (ny - c_ymin) / (c_ymax - c_ymin))
            
            # Защита от выхода за границы
            px = max(20, min(w-20, px))
            py = max(20, min(h-20, py))
            
            # Рисуем красный прицел (Crosshair)
            cv2.circle(disp, (px, py), 20, (0, 0, 255), 2)
            cv2.line(disp, (px - 30, py), (px + 30, py), (0, 0, 255), 2)
            cv2.line(disp, (px, py - 30), (px, py + 30), (0, 0, 255), 2)
            cv2.putText(disp, "TARGET WP", (px + 20, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # BBoxes и треки (с использованием профильтрованной телеметрии)
        if tracks is not None and len(tracks) > 0:
            for tr in tracks:
                x, y, w, h = tr['bbox']
                track_id = tr['track_id']
                cls_id = tr['class_id']
                
                # Конвертация xywh (центр_x, центр_y, ширина, высота) -> xyxy
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = f"ID:{track_id} C:{cls_id}"
                cv2.putText(disp, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        return disp
