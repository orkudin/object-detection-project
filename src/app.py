import os
import sys
import gradio as gr

# Обеспечиваем корректные импорты при запуске из корня проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_pipeline_yield

# Функция-обертка для запуска конвейера из Gradio
def process_video(video_file, stream_url, webcam_enabled, model_weights, tracker, conf_thresh, grid_step):
    # Определение источника (Веб-камера -> Файл -> RTSP)
    if webcam_enabled:
        video_source = 0
    elif video_file:
        video_source = video_file
    elif stream_url:
        video_source = stream_url
    else:
        raise gr.Error("Пожалуйста, выберите источник: Загрузите видео, вставьте RTSP или включите USB-камеру!")
        
    config_overrides = {
        # Принудительно направляем путь в папку models/
        "weights": os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", model_weights),
        "tracker": tracker,
        "conf": float(conf_thresh),
        "grid_step": int(grid_step)
    }
    
    # Генератор кадров `yield` для плавной Web-операнды
    for frame in run_pipeline_yield(video_source, config_overrides):
        yield frame


def get_available_models():
    """Сканирует папку models/ на наличие поддерживаемых весов"""
    # Путь к папке от корня проекта
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    supported_ext = ('.pt', '.onnx', '.rknn', '.engine')
    models = [f for f in os.listdir(models_dir) if f.endswith(supported_ext)]
    return sorted(models) if models else ["Модели не найдены. Поместите их в папку models/"]

def create_ui():
    theme = gr.themes.Soft(
        primary_hue="indigo", 
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )
    
    with gr.Blocks(theme=theme, title="БПЛА-Навигатор: Панель Оператора") as app:
        gr.Markdown(
            """
            # UAV-CV-Navigator  |  Панель оператора
            Конвейер обнаружения, отслеживания объектов и адаптивного планирования маршрута БПЛА.  
            Выберите источник видео, настройте параметры модели и запустите обработку.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Конфигурация")
                
                with gr.Tab("Локальное видео"):
                    video_input = gr.Video(label="Загрузите MP4 / AVI", sources=["upload"])
                with gr.Tab("RTSP / IP-камера"):
                    stream_input = gr.Textbox(label="RTSP URL", placeholder="rtsp://10.0.0.1:554/stream")
                with gr.Tab("USB-камера"):
                    webcam_cb = gr.Checkbox(label="Использовать системную веб-камеру (устройство 0)")
                
                available_models = get_available_models()
                model_dd = gr.Dropdown(
                    choices=available_models, 
                    value=available_models[0] if available_models else None,
                    label="Модель детектора"
                )
                
                tracker_dd = gr.Dropdown(
                    choices=["bytetrack", "botsort"], 
                    value="bytetrack", 
                    label="Алгоритм трекинга"
                )
                
                conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.05, label="Порог уверенности (Confidence)")
                grid_slider = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="Шаг сетки навигатора, м")
                
                with gr.Row():
                    run_btn = gr.Button("Запуск", variant="primary", scale=2)
                    stop_btn = gr.Button("Стоп", variant="stop", scale=1)
                
            with gr.Column(scale=2):
                gr.Markdown("### Видеопоток")
                # output_image будет обновляться кадр за кадром (Стриминг)
                output_image = gr.Image(label="OSD Telemetry Output", streaming=True)
                
        # Привязка функции к кнопке (Generator Mode)
        run_event = run_btn.click(
            fn=process_video,
            inputs=[video_input, stream_input, webcam_cb, model_dd, tracker_dd, conf_slider, grid_slider],
            outputs=[output_image]
        )
        
        # Кнопка остановки перехватывает и отменяет генератор событий
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

    return app

if __name__ == "__main__":
    ui = create_ui()
    # Запускаем локальный веб-сервер (по умолчанию 127.0.0.1, что избегает ошибки ERR_ADDRESS_INVALID в Windows)
    ui.launch(server_port=7860, share=False)
