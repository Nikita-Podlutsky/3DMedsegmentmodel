import os
from pathlib import Path
# Импортируем готовые функции и конфигурацию из вашего data_units.py
from data_units import get_prepared_synthstrip_dataset, SYNTHSTRIP_PROCESSING_CONFIG

# Путь к вашей папке с распакованным датасетом
dataset_dir = r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_data_v1.5"

# Путь, куда вы хотите сохранить собранный H5 файл
h5_cache_path = r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_prepared.h5"

print("--- Запуск процесса восстановления H5 датасета ---")
print(f"Исходная директория: {dataset_dir}")
print(f"Целевой H5 файл: {h5_cache_path}\n")

# Проверим, существует ли папка перед запуском
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Директория {dataset_dir} не найдена! Проверьте путь.")

# Запускаем сборку
h5_file = get_prepared_synthstrip_dataset(
    dataset_dir=dataset_dir,
    h5_cache_path=h5_cache_path,
    config=SYNTHSTRIP_PROCESSING_CONFIG,
    force_create=True  # Форсируем создание заново
)

print(f"\n[+] Успех! Сборка завершена. Файл сохранен по пути: {h5_file}")