# 🦅 UAV-CV-Navigator: Intelligent Aerial Object Detection & Tracking

Система обнаружения, отслеживания объектов и адаптивного планирования маршрута БПЛА.  
Разработано для мониторинга воздушного пространства и взлётно-посадочных полос (ВПП).

| | |
|---|---|
| **Статус** | Прототип (ТЗ от 13.04.2026) |
| **Целевая ОС** | Ubuntu 20/22/24 LTS, Windows 10/11 |
| **Python** | 3.10+ (рекомендуется 3.11) |
| **Поддерживаемые платформы** | PC (CPU/GPU), Radxa 4D (Rockchip RK3576 NPU) |

> Архитектура строится на принципах **SOLID**, **DRY** и **Zero Hardcode Policy**.  
> Все параметры — из YAML-конфигов. Бизнес-логика декомпозирована на независимые модули:  
> **Детекция → Трекинг → Планирование → Визуализация**.

---

## 📁 Структура проекта

```
object_detection/
├── configs/
│   └── default.yaml            # Единственный источник конфигурации
├── data/
│   └── output/                 # Результаты: result.mp4, telemetry.jsonl
├── models/                     # Веса моделей (.pt, .onnx, .rknn)
├── notebooks/
│   └── Colab_Pipeline_Runner.ipynb
├── src/
│   ├── detection/
│   │   ├── base.py             # Абстрактный интерфейс (BaseDetector)
│   │   ├── yolo.py             # YOLODetector (PyTorch / ONNX Runtime)
│   │   ├── rknn.py             # RKNNDetector (Rockchip NPU)
│   │   └── factory.py          # Фабрика детекторов (패턴 Factory)
│   ├── tracking/
│   │   └── tracker.py          # Конфигуратор трекеров (ByteTrack / BoT-SORT)
│   ├── planning/
│   │   └── planner.py          # Адаптивный маршрутизатор (Heatmap + Age Map)
│   ├── utils/
│   │   ├── logger.py           # Системный логгер + JSONL телеметрия
│   │   └── visualizer.py       # OSD-рендер (HUD, прицел, bbox)
│   ├── pipeline.py             # Главный конвейер (CLI + генератор для Web)
│   └── app.py                  # Gradio Web UI — Панель Оператора
├── tests/
│   └── test_pipeline.py        # Юнит-тесты (unittest)
├── requirements.txt
└── .gitignore
```

---

## 🚀 1. Установка

```bash
conda create -n object_detection_project python=3.11 -y
conda activate object_detection_project
pip install -r requirements.txt
```

Скачивание демо-видео *(опционально)*:
```bash
python data/download_test_video.py
```

---

## 🛠 2. Конфигурирование (Zero Hardcode)

Вся логика конфигурируется через `configs/default.yaml`:

| Секция | Параметры | Описание |
|--------|-----------|----------|
| `detector` | `weights_path`, `confidence_threshold`, `classes_of_interest` | Модель, порог уверенности, фильтр классов |
| `tracker` | `algorithm`, `track_buffer`, `match_thresh` | Алгоритм трекинга, буфер памяти, порог совпадения |
| `planner` | `mode`, `grid_step_meters`, `priority_weights` | Режим навигации (`static` / `adaptive`), шаг сетки, веса эвристики |

---

## 🖥 3. Запуск из терминала (CLI)

**Подготовка PYTHONPATH** *(требуется один раз за сессию)*:
```bash
# Linux / macOS / Colab
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Windows PowerShell
$env:PYTHONPATH += ";$(Get-Location)"
```

### Запуск по ТЗ (одной командой)
```bash
python src/pipeline.py --video data/test_video.mp4 --config configs/default.yaml --mode full
```

### Дополнительные флаги
| Флаг | Пример | Описание |
|------|--------|----------|
| `--weights` | `--weights models/mnv4_yolo11s.pt` | Переопределить модель без правки YAML |
| `--tracker` | `--tracker botsort` | Переключить трекер (компенсация тряски GMC) |
| `--video` | `--video 0` или `--video rtsp://...` | USB-камера или RTSP-поток дрона |
| `--headless` | `--headless` | Отключить окна (обязательно для серверов) |

> Нажмите `q` во время Live-отрисовки для грациозной остановки.

---

## 🌐 4. Веб-Интерфейс (Панель Оператора)

Запуск Gradio Web UI:
```bash
python src/app.py
```
Откройте в браузере: **http://127.0.0.1:7860**

**Функционал панели:**
- 📁 Загрузка видеофайла или ввод RTSP-ссылки
- 📷 Подключение USB-камеры
- 🧠 Динамический выбор модели из папки `models/` (`.pt` / `.onnx` / `.rknn`)
- 🎯 Переключение трекера (ByteTrack / BoT-SORT)
- 🔥 Ползунок Confidence Threshold
- 🗺️ Ползунок шага сетки навигатора
- 🚀 / 🛑 Кнопки запуска и остановки патрулирования

---

## ☁️ 5. Облачный запуск (Google Colab)

Файл `notebooks/Colab_Pipeline_Runner.ipynb` содержит готовый пайплайн:
1. Клонирование репозитория
2. Установка зависимостей
3. Запуск пайплайна в headless-режиме
4. Визуализация телеметрии (графики FPS и количества объектов)
5. Встроенный HTML5-видеоплеер (FFmpeg → H.264 транскодирование)

---

## 🏗 6. Архитектура (SOLID / Factory Pattern)

```
┌─────────────────────────────────────────────────┐
│                 pipeline.py                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│   │ Detector │→ │ Tracker  │→ │   Planner    │ │
│   │ (Factory)│  │(ByteTrack│  │  (Adaptive)  │ │
│   └────┬─────┘  │ /BoTSORT)│  └──────────────┘ │
│        │        └──────────┘                    │
│   ┌────┴─────────────────────┐                  │
│   │      BaseDetector        │ ← DIP (SOLID)    │
│   ├──────────┬───────────────┤                  │
│   │ YOLODet  │  RKNNDetector │                  │
│   │ (.pt/.ox)│  (.rknn, NPU) │                  │
│   └──────────┴───────────────┘                  │
└─────────────────────────────────────────────────┘
```

**Фабрика детекторов** (`src/detection/factory.py`) автоматически выбирает бэкенд:
- `.pt` / `.onnx` → `YOLODetector` (PyTorch / ONNX Runtime, PC)
- `.rknn` → `RKNNDetector` (rknnlite, Rockchip NPU, Edge)

---

## 🧪 7. Тестирование

```bash
python -m unittest discover tests
```

| Тест | Описание |
|------|----------|
| `test_config_validation` | Валидация YAML: отсутствие хардкода путей и параметров |
| `test_planner_route` | Проверка эвристики адаптивного навигатора на синтетических данных |
| `test_tracking_stability` | Стресс-тест детектора на чёрных кадрах и отсутствии объектов |

---

## 📊 8. Выходные артефакты

| Файл | Формат | Описание |
|------|--------|----------|
| `data/output/result.mp4` | MP4 (mp4v) | Видео с OSD: рамки, ID треков, прицел навигатора |
| `data/output/telemetry.jsonl` | JSON Lines | Покадровые метрики: `frame_id`, `fps`, `detections`, `waypoints` |

---

## 📦 9. Модели

Поместите файлы весов в папку `models/`. Web UI автоматически их обнаружит.

| Модель | Формат | Назначение |
|--------|--------|------------|
| `yolo11s.pt` | PyTorch | Базовая детекция (COCO, 80 классов) |
| `yolo11s.onnx` | ONNX | Кроссплатформенный инференс (CPU оптимизация) |
| `mnv4_yolo11s.pt` | PyTorch | Fine-tuned на аэросъёмку (MobileNetV4 backbone) |
| `mnv4_yolo11s.onnx` | ONNX | Экспорт MNv4 для Edge/Server |
| `*.rknn` | RKNN | Для аппаратного NPU Rockchip RK3576 (Radxa 4D) |

> ⚠️ Веса не коммитятся в Git (см. `.gitignore`). Стандартные `yolo11s.pt` скачиваются автоматически при первом запуске.

---

## 🏛 Лицензия и Авторство

Проект разработан в рамках научной стажировки МУИТ АГА (2026).
