import os
import glob
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Параметры
IMG_SIZE = (128, 128)   # размер изображений для модели
BATCH_SIZE = 32
EPOCHS = 10

# Классы: первые 5 будут загружаться из 'train' и 'val', для 'human' – из отдельной папки
classes = ['elephant', 'horse', 'lion', 'cat', 'dog', 'human']
non_human_classes = classes[:-1]  # ['elephant', 'horse', 'lion', 'cat', 'dog']

def load_dataset_from_base(base_path, class_list, img_size, exts=('*.jpg', '*.jpeg', '*.png')):
    X = []
    y = []
    for idx, cls in enumerate(class_list):
        folder = os.path.join(base_path, cls)
        if not os.path.exists(folder):
            print(f"Папка {folder} не найдена, пропуск.")
            continue
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img)
                X.append(img_array)
                y.append(idx)  # индекс в списке class_list
            except Exception as e:
                print(f"Ошибка при загрузке {img_path}: {e}")
    return np.array(X), np.array(y)

def load_dataset_from_folder(folder, img_size, label, exts=('*.jpg', '*.jpeg', '*.png')):
    X = []
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img)
            X.append(img_array)
        except Exception as e:
            print(f"Ошибка при загрузке {img_path}: {e}")
    # Создаем массив меток для всех изображений (одинаковая метка label)
    y = np.full(len(X), label)
    return np.array(X), y

# Загрузка данных для не-human классов
print("Загрузка тренировочного датасета для не-human классов...")
X_train_main, y_train_main = load_dataset_from_base("train", non_human_classes, IMG_SIZE)
print(f"Тренировочных изображений (не-human): {len(X_train_main)}")

print("Загрузка валидационного датасета для не-human классов...")
X_val_main, y_val_main = load_dataset_from_base("val", non_human_classes, IMG_SIZE)
print(f"Валидационных изображений (не-human): {len(X_val_main)}")

# Загрузка датасета для класса 'human'
print("Загрузка датасета для класса 'human'...")
human_folder = "humans"  # путь к датасету human
if not os.path.exists(human_folder):
    print(f"Папка {human_folder} не найдена, пропуск датасета 'human'.")
    X_human = np.array([])
    y_human = np.array([])
else:
    # Для класса human используем индекс равный len(non_human_classes)
    human_label = len(non_human_classes)  # индекс для 'human'
    X_human, y_human = load_dataset_from_folder(human_folder, IMG_SIZE, human_label)
    print(f"Изображений для 'human': {len(X_human)}")
    # Делим на обучающую и валидационную выборки (80/20)
    X_human_train, X_human_val, y_human_train, y_human_val = train_test_split(
        X_human, y_human, test_size=0.2, random_state=42, stratify=y_human)
    print(f"Тренировочных изображений 'human': {len(X_human_train)}")
    print(f"Валидационных изображений 'human': {len(X_human_val)}")

# Объединяем данные не-human и human
if X_human.size > 0:
    X_train = np.concatenate([X_train_main, X_human_train], axis=0)
    y_train = np.concatenate([y_train_main, y_human_train], axis=0)
    X_val = np.concatenate([X_val_main, X_human_val], axis=0)
    y_val = np.concatenate([y_val_main, y_human_val], axis=0)
else:
    X_train, y_train = X_train_main, y_train_main
    X_val, y_val = X_val_main, y_val_main

print(f"Всего тренировочных изображений: {len(X_train)}")
print(f"Всего валидационных изображений: {len(X_val)}")

# Нормализация изображений
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# One-hot кодирование меток для всех классов
y_train_cat = to_categorical(y_train, num_classes=len(classes))
y_val_cat = to_categorical(y_val, num_classes=len(classes))

# Построение модели (простая CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Обучение модели
history = model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val_cat))

# Сохранение модели
model_save_path = "my_model_new.h5"
model.save(model_save_path)
print(f"Модель сохранена в файл: {model_save_path}")
