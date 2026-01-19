import os
import re

def increment_first_number_in_file(filepath):
    """
    Открывает файл, находит первое число в содержимом, увеличивает его на 1,
    и перезаписывает файл с измененным значением, не трогая остальной контент.
    """
    try:
        # 1. Чтение всего содержимого файла
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. Поиск первого числа
        match = re.search(r'\d+', content)

        if not match:
            print(f"  [SKIP] В файле {filepath} не найдено чисел.")
            return

        found_number_str = match.group(0)
        original_number = int(found_number_str)
        new_number = original_number + 1
        new_number_str = str(new_number)

        # 3. Замена первого вхождения числа в строке
        # count=1 гарантирует замену только первого найденного числа
        new_content = re.sub(
            r'\d+',
            new_number_str,
            content,
            count=1
        )

        # 4. Запись измененного содержимого обратно в файл
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  [OK] {filepath}: {original_number} -> {new_number}")

    except Exception as e:
        print(f"  [ERROR] Не удалось обработать файл {filepath}. Ошибка: {e}")


def process_files_in_directory(directory_path, file_extension=".txt"):
    """
    Перебирает все файлы с указанным расширением в директории и применяет 
    функцию increment_first_number_in_file к каждому из них.
    """
    print(f"Начинаем обработку файлов в директории: {directory_path}")
    
    file_count = 0
    
    # os.walk лучше, если нужно перебирать подпапки, но os.listdir проще для одной папки
    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            filepath = os.path.join(directory_path, filename)
            
            # Пропускаем, если это не файл (хотя endswith должен отсеять папки)
            if os.path.isfile(filepath):
                increment_first_number_in_file(filepath)
                file_count += 1
    
    print(f"\nОбработка завершена. Изменено {file_count} файлов.")


# --- НАСТРОЙКИ ---

# 1. Укажите путь к папке, где лежат ваши TXT файлы
TARGET_DIRECTORY = r'C:\Users\Auron\Downloads\codes\yolo\test'  # ИЗМЕНИТЕ ЭТОТ ПУТЬ


EXTENSION = ".txt" 

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Проверка, существует ли указанная папка
    if not os.path.isdir(TARGET_DIRECTORY):
        print(f"ОШИБКА: Указанный каталог не существует: {TARGET_DIRECTORY}")
    else:
        process_files_in_directory(TARGET_DIRECTORY, EXTENSION)