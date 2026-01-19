import os
import shutil
import random

def prepare_yolo_dataset(source_dir, val_split=0.2, project_name="yolo_project"):
   
    print("--- Запуск подготовки данных YOLO ---")
    image_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"Ошибка: В папке {source_dir} не найдено изображений.")
        return

    valid_pairs = []
    for img_file in image_files:
        # Определяем имя TXT-файла (заменяем расширение)
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + '.txt'
        
        if os.path.exists(os.path.join(source_dir, txt_file)):
            valid_pairs.append((img_file, txt_file))
        else:
            print(f"Предупреждение: Пропущен {img_file}, так как {txt_file} не найден.")

    if not valid_pairs:
        print("Ошибка: Не найдено ни одной пары (изображение + аннотация).")
        return

    random.shuffle(valid_pairs)
    
    num_total = len(valid_pairs)
    num_val = int(num_total * val_split)
    
    train_pairs = valid_pairs[:-num_val]
    val_pairs = valid_pairs[-num_val:]
    
    print(f"Всего пар найдено: {num_total}")
    print(f"Обучение (Train): {len(train_pairs)} | Валидация (Val): {len(val_pairs)}")

    base_path = os.path.join(os.getcwd(), project_name) # Создаем папку в текущей директории скрипта
    
    paths = {
        'img_train': os.path.join(base_path, 'images', 'train'),
        'lbl_train': os.path.join(base_path, 'labels', 'train'),
        'img_val': os.path.join(base_path, 'images', 'val'),
        'lbl_val': os.path.join(base_path, 'labels', 'val'),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    def copy_files(pairs, is_train):
        img_path = paths['img_train'] if is_train else paths['img_val']
        lbl_path = paths['lbl_train'] if is_train else paths['lbl_val']
        
        for img_file, txt_file in pairs:
            src_img = os.path.join(source_dir, img_file)
            src_lbl = os.path.join(source_dir, txt_file)
            
            shutil.copy(src_img, img_path)
            shutil.copy(src_lbl, lbl_path)

    copy_files(train_pairs, is_train=True)
    copy_files(val_pairs, is_train=False)
    
    print(f"\nУспешно! Данные скопированы в структуру YOLO в папку: {base_path}")
    print("Теперь вы готовы к созданию YAML и запуску обучения.")



SOURCE_DATA_FOLDER = r'C:\Users\Auron\Downloads\codes\yolo\dataset' # <-- ИЗМЕНИТЕ ЭТОТ ПУТЬ

if __name__ == '__main__':
    prepare_yolo_dataset(
        source_dir=SOURCE_DATA_FOLDER,
        val_split=0.15, # 15% данных пойдут на валидацию
        project_name="Custom_Object_Detector"
    )