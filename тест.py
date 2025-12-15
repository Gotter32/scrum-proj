import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("C:/Users/User/Desktop/Папка_с_папками/Учёба/Бакалавр/4.2/Зрение 3/mod3.h5")

classes = ['elephant', 'horse', 'lion', 'cat', 'dog', 'human']

def preprocess_frame(frame, img_size=(128, 128)):
    img = cv2.resize(frame, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Ошибка: не удалось открыть веб-камеру.")
    exit()

print("Запуск распознавания объектов. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра")
        break

    
    img = preprocess_frame(frame)

    preds = model.predict(img)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    label = classes[pred_class]

    text = f'Prediction: {label}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Распознавание объектов", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()