#run_synthstrip_wsl.py
import os
import subprocess
from pathlib import Path
from nilearn import plotting





"""
Скрипт для выполнения skull-stripping (удаления черепа) на МРТ-изображениях с помощью утилиты mri_synthstrip из пакета FreeSurfer.

Ключевая особенность скрипта - возможность запускать Linux-инструмент
FreeSurfer непосредственно из среды Windows через подсистему Windows для Linux (WSL).

Основные шаги:
1. Принимает на вход путь к NIfTI файлу в формате Windows.
2. Конвертирует путь файла в формат, совместимый с WSL (например, /mnt/c/...).
3. Формирует и выполняет команду `wsl mri_synthstrip ...`.
4. Сохраняет два файла: изображение мозга без черепа и маску мозга.
5. Визуализирует результат, накладывая контур маски на исходное изображение с помощью nilearn.
"""



def convert_path_to_wsl(windows_path: str) -> str:
    """Конвертирует путь Windows (C:\...) в путь WSL (/mnt/c/...)."""
    p = Path(windows_path)
    drive = p.drive.lower().replace(':', '')
    path_without_drive = str(p.relative_to(p.anchor)).replace('\\', '/')
    return f"/mnt/{drive}/{path_without_drive}"

def run_freesurfer_synthstrip(input_image: str):
    """
    Выполняет skull-stripping с помощью mri_synthstrip из FreeSurfer (через WSL).
    """
    
    # --- 1. Проверка путей на стороне Windows ---
    input_path = Path(input_image).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")
        
    output_dir = Path('./synthstrip_output').resolve()
    output_dir.mkdir(exist_ok=True)
    
    base_name = input_path.name.split('.')[0]
    out_stripped_file = output_dir / f"{base_name}_brain.nii.gz"
    out_mask_file = output_dir / f"{base_name}_brain_mask.nii.gz"
    
    # --- 2. Конвертация путей для WSL ---
    wsl_input = convert_path_to_wsl(str(input_path))
    wsl_output = convert_path_to_wsl(str(out_stripped_file))
    wsl_mask = convert_path_to_wsl(str(out_mask_file))
    
    # --- 3. Формирование и выполнение команды через WSL ---
    command = [
        "wsl",                 # Команда для запуска Linux-программы из Windows
        "mri_synthstrip",      # Правильная, официальная команда
        "-i", wsl_input,
        "-o", wsl_output,
        "-m", wsl_mask
    ]
    
    print("--- Запуск FreeSurfer SynthStrip через WSL ---")
    print(f"Команда: {' '.join(command)}")
    
    # shell=True рекомендуется для вызова `wsl` из Windows
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    
    if result.returncode != 0:
        print("!!! ОШИБКА ВЫПОЛНЕНИЯ mri_synthstrip !!!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return

    print("--- mri_synthstrip успешно завершен ---")
    print(result.stdout)

    # --- 4. Визуализация результатов (пути Windows) ---
    if out_stripped_file.exists() and out_mask_file.exists():
        print("\nПодготовка к визуализации...")
        plotting.plot_anat(
            anat_img=str(input_path), title="Исходное изображение",
            display_mode='ortho', draw_cross=False, annotate=False
        )
        # ... (остальной код визуализации как раньше) ...
        display = plotting.plot_anat(anat_img=str(input_path), title="Контур маски")
        display.add_contours(str(out_mask_file), colors='r', linewidths=2)
        plotting.show()
    else:
        print("\nОшибка: Выходные файлы не были созданы.")

# --- Точка входа в программу ---
if __name__ == '__main__':
    t1w_file = 'C:/Users/pniki/Documents/Programs/ML/Исследования/MEd/synthstrip_data_v1.5/asl_epi_101/image.nii.gz'
    run_freesurfer_synthstrip(t1w_file)