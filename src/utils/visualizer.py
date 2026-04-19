import cv2

class Visualizer:
    def __init__(self, logger):
        self.logger = logger
        
    def draw(self, frame, tracks, current_wp, next_wp, fps, planner=None, class_names=None, runway_roi=None, show_osd=True, show_target_wp=True):
        disp = frame.copy()
        h, w = disp.shape[:2]
        
        # Отрисовка зоны ВПП (ROI)
        if runway_roi is not None:
            import numpy as np
            roi_poly = np.array([[int(p[0]*w), int(p[1]*h)] for p in runway_roi], np.int32)
            cv2.polylines(disp, [roi_poly], True, (0, 100, 0), 2)
            cv2.putText(disp, "RUNWAY ROI", (roi_poly[0][0], roi_poly[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
            
        # Инфо панель (скрываемая, уменьшенная)
        if show_osd:
            cv2.rectangle(disp, (10, 10), (320, 100), (0, 0, 0), -1)
            cv2.putText(disp, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            mode_text = planner.mode.upper() if planner else "UNKNOWN"
            cv2.putText(disp, f"MODE: {mode_text}", (140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255) if mode_text == 'ADAPTIVE' else (255, 255, 255), 2)
            
            cv2.putText(disp, f"Cur WP: {current_wp}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(disp, f"Next WP: {next_wp}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Отрисовка работы НАВИГАТОРА (прицел следующей точки на экране)
        if planner is not None and next_wp is not None and show_target_wp:
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
                x, y, w_box, h_box = tr['bbox']
                track_id = tr['track_id']
                cls_id = tr['class_id']
                
                # Конвертация xywh (центр_x, центр_y, ширина, высота) -> xyxy
                x1, y1 = int(x - w_box / 2), int(y - h_box / 2)
                x2, y2 = int(x + w_box / 2), int(y + h_box / 2)
                
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                # Вывод названия класса, если оно доступно
                if class_names and int(cls_id) in class_names:
                    label = f"{class_names[int(cls_id)].upper()} ID:{track_id}"
                else:
                    label = f"ID:{track_id} C:{cls_id}"
                    
                cv2.putText(disp, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        return disp
