from ultralytics import YOLO
import os

# --- НАСТРОЙКИ ОБУЧЕНИЯ ---
DATA_YAML_PATH = 'conf_yolo.yaml' 
MODEL_WEIGHTS = 'yolov8n.pt' 
EPOCHS = 10
IMG_SIZE = 640

def train_and_report(data_yaml, model_weights, epochs, img_size):
    
    if not os.path.exists(data_yaml):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл конфигурации YAML не найден по пути: {data_yaml}")
        return

    # 1. Инициализация модели
    model = YOLO(model_weights)
    print(f"Начало обучения с использованием {model_weights} на {epochs} эпох...")

    # 2. Запуск обучения
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        workers=8 # Можно добавить количество потоков для ускорения
    )

    print("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")
    

    # 4. Информация о сохранении модели
    save_dir = model.trainer.save_dir
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    
    print(f"\nМодель сохранена в: {save_dir}")
    print(f"Лучшая модель: {best_model_path}")


if __name__ == '__main__':
    train_and_report(
        data_yaml=DATA_YAML_PATH,
        model_weights=MODEL_WEIGHTS,
        epochs=EPOCHS,
        img_size=IMG_SIZE
    )