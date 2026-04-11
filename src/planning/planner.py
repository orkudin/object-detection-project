import numpy as np

class UAVPlanner:
    def __init__(self, config, logger):
        self.logger = logger
        self.mode = config.get("mode", "static")
        self.grid_step = config.get("grid_step_meters", 10)
        self.coverage_area = config.get("coverage_area", [0, 0, 100, 100])
        self.safe_alt = config.get("safe_altitude", 50)
        
        # Загрузка весовых коэффициентов из YAML без хардкода
        self.weights = config.get("priority_weights", {"frequent_detection": 1.5, "time_since_last_visit": 1.2})
        self.w_det = self.weights.get("frequent_detection", 1.5)
        self.w_time = self.weights.get("time_since_last_visit", 1.2)
        
        self.logger.info(f"Навигатор запущен в режиме '{self.mode}'. Шаг сетки: {self.grid_step}м")
        
        self.current_wp_idx = 0
        self.waypoints = self._generate_static_route()
        
        # Инициализация матриц для Адаптивного режима (Phase 2)
        x_min, y_min, x_max, y_max = self.coverage_area
        self.grid_width = max(1, int((x_max - x_min) / self.grid_step))
        self.grid_height = max(1, int((y_max - y_min) / self.grid_step))
        
        # Сетка скоплений объектов
        self.heatmap = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        # Сетка времени, прошедшего с последнего визита
        self.age_map = np.ones((self.grid_height, self.grid_width), dtype=np.float32)
        
    def _generate_static_route(self):
        """Паттерн 'газонокосилка' (lawnmower) для базового покрытия территории"""
        x_min, y_min, x_max, y_max = self.coverage_area
        waypoints = []
        
        if self.mode == "adaptive":
            # В адаптивном режиме пропускаем газонокосилку. Даем точку старта в центре, 
            # чтобы навигатор сразу начал строить маршрут по 'горячим точкам' (людям/машинам)
            waypoints.append(((x_max - x_min)/2, (y_max - y_min)/2, self.safe_alt))
            return waypoints
            
        y = y_min
        going_right = True
        
        while y <= y_max:
            if going_right:
                waypoints.append((x_min, y, self.safe_alt))
                waypoints.append((x_max, y, self.safe_alt))
            else:
                waypoints.append((x_max, y, self.safe_alt))
                waypoints.append((x_min, y, self.safe_alt))
            going_right = not going_right
            y += self.grid_step
            
        return waypoints
        
    def _update_adaptive_waypoint(self):
        """
        Вычисляет самую "горячую" точку сетки на основе эвристики:
        P_ij = (W_det * D_ij) + (W_time * T_ij)
        """
        # Считаем матрицу приоритетов
        priority_matrix = (self.w_det * self.heatmap) + (self.w_time * self.age_map)
        
        # Ищем 2D-индекс ячейки с максимальным приоритетом
        max_idx = np.unravel_index(np.argmax(priority_matrix, axis=None), priority_matrix.shape)
        row, col = max_idx
        
        # Обнуляем возраст этой ячейки, так как мы сгенерировали маршрут в нее
        self.age_map[row, col] = 0.0
        
        # Перевод индексов сетки в реальные "географические" координаты БПЛА
        x_min, y_min, _, _ = self.coverage_area
        target_x = round(x_min + (col + 0.5) * self.grid_step, 2)
        target_y = round(y_min + (row + 0.5) * self.grid_step, 2)
        
        self.waypoints.append((target_x, target_y, self.safe_alt))
        self.logger.info(f"[{self.mode.upper()}] Сгенерирована адаптивная точка: X:{target_x} Y:{target_y}")
        
    def update_state(self, tracks, frame_w, frame_h):
        """
        Обновление тепловой карты на основе треков с учетом размерности кадра.
        """
        # Состариваем все зоны (увеличиваем приоритет непосещенных зон)
        self.age_map += 1.0
        
        if not tracks:
            # Охлаждение тепловой карты (в случае потери объектов со временем)
            self.heatmap *= 0.98
            return
            
        for tr in tracks:
            # Достаем x_center, y_center (относительно пикселей экрана)
            x_c, y_c, _, _ = tr['bbox']
            
            # Проекция из пиксельных координат в метрику coverage_area
            x_min, y_min, x_max, y_max = self.coverage_area
            norm_x = (x_c / frame_w) * (x_max - x_min)
            norm_y = (y_c / frame_h) * (y_max - y_min)
            
            # Перевод в индексы сетки (Cell Row / Col)
            grid_col = int(norm_x / self.grid_step)
            grid_row = int(norm_y / self.grid_step)
            
            # Защита от выхода за границы массива
            grid_col = min(max(grid_col, 0), self.grid_width - 1)
            grid_row = min(max(grid_row, 0), self.grid_height - 1)
            
            # Увеличиваем приоритет (градус тепловой ячейки)
            self.heatmap[grid_row, grid_col] += 1.0

        # Постепенное остывание тепловой карты (экспоненциальный спад)
        self.heatmap *= 0.98
        
    def get_current_waypoint(self):
        if not self.waypoints: return (0,0,self.safe_alt)
        return self.waypoints[min(self.current_wp_idx, len(self.waypoints)-1)]
        
    def get_next_waypoint(self):
        # Если мы достигли конца статического маршрута в адаптивном режиме, генерируем новую точку
        if self.mode == "adaptive" and (self.current_wp_idx + 1) >= len(self.waypoints):
            self._update_adaptive_waypoint()
            
        if not self.waypoints: return (0,0,self.safe_alt)
        return self.waypoints[min(self.current_wp_idx + 1, len(self.waypoints)-1)]
