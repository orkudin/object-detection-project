# %% [markdown]
# # UAV Telemetry Analysis (Промежуточный Этам 13.04)
# Этот скрипт демонстрирует, как загружать `telemetry.jsonl` в `pandas` для построения метрик.
# Скопируйте этот код в ячейки вашего Google Colab.

# %%
import pandas as pd
import json
import matplotlib.pyplot as plt

file_path = '../data/output/telemetry.jsonl'

# %% [markdown]
# ## 1. Загрузка данных
# %%
data = []
try:
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            
    df = pd.DataFrame(data)
    print(f"Загружено {len(df)} кадров телеметрии.")
    display(df.head())
except Exception as e:
    print("Ошибка загрузки:", e)

# %% [markdown]
# ## 2. Анализ производительности (FPS)
# %%
if 'df' in locals():
    plt.figure(figsize=(10, 4))
    plt.plot(df['frame_id'], df['fps'], color='green')
    plt.axhline(df['fps'].mean(), color='red', linestyle='--', label=f"Средний FPS: {df['fps'].mean():.2f}")
    plt.title("Производительность Pipeline (FPS)")
    plt.xlabel("Кадр")
    plt.ylabel("FPS")
    plt.legend()
    plt.grid()
    plt.show()

# %% [markdown]
# ## 3. Статистика по объектам
# %%
if 'df' in locals():
    # Подсчет количества треков в каждом кадре
    df['num_objects'] = df['detections'].apply(len)
    
    plt.figure(figsize=(10, 4))
    plt.plot(df['frame_id'], df['num_objects'])
    plt.title("Количество обнаруженных объектов на кадр")
    plt.xlabel("Кадр")
    plt.ylabel("Количество")
    plt.grid()
    plt.show()
