from ultralytics import YOLO
import cv2
import os

# --- НАСТРОЙКИ ДЕТЕКЦИИ ---
MODEL_PATH = r'C:\Users\Auron\Downloads\codes\yolo\train\weights\best.pt'

CAMERA_INDEX = 0 
CONFIDENCE_THRESHOLD = 0.5 #вероятность для отрисовки

def run_live_detection(model_path, camera_index, confidence):
    
    if not os.path.exists(model_path):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Модель не найдена по пути: {model_path}")
        print("Проверьте, правильно ли указан путь к 'best.pt'.")
        return

    try:
        model = YOLO(model_path)
        print(f"Модель '{os.path.basename(model_path)}' успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    # 2. Инициализация веб-камеры
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть веб-камеру с индексом {camera_index}.")
        return

    print("\n--- Запуск детекции в реальном времени (Нажмите 'q' для выхода) ---")

    while True:
        # Чтение кадра с камеры
        ret, frame = cap.read()
        
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        # 3. Запуск предсказания
        results = model(frame, conf=confidence, verbose=False)
        # 4. Визуализация результатов
        annotated_frame = results[0].plot() 
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)

        if results[0].boxes:
            print("-" * 30)
            print(f"Обнаружено объектов: {len(results[0].boxes)}")
            for box in results[0].boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                # Получаем имя класса из словаря имен модели
                class_name = model.names[class_id]
                print(f"  Класс: {class_name}, Уверенность: {conf:.2f}")
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_detection(
        model_path=MODEL_PATH, 
        camera_index=CAMERA_INDEX, 
        confidence=CONFIDENCE_THRESHOLD
    )