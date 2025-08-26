# -*- coding: utf-8 -*-
# train_monai_unet_3d.py
import os
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Импорты из MONAI ---
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    SpatialPadD,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    EnsureTyped,
)
from monai.metrics import DiceMetric
from monai.data import Dataset, decollate_batch
from torch.optim.lr_scheduler import ReduceLROnPlateau










"""
Скрипт для обучения 3D-модели U-Net для семантической сегментации медицинских изображений с использованием библиотек MONAI и PyTorch.

Основные функции:
- Загружает 3D-объемы и маски из одного HDF5 файла.
- Использует гибкий пайплайн трансформаций MONAI для предобработки и аугментации данных (повороты, отражения, изменение интенсивности).
- Реализует patch-based обучение с помощью `RandCropByPosNegLabeld` для эффективной работы с большими 3D-объемами.
- Проводит валидацию на полных объемах данных с помощью `sliding_window_inference`.
- Отслеживает метрику Dice и сохраняет модель с наилучшим показателем на валидационной выборке.
- Генерирует и сохраняет изображения со срезами предсказаний для визуального контроля качества обучения на каждой эпохе.
"""





# === ШАГ 1: Пользовательский трансформер для загрузки из HDF5 (без изменений) ===
# Этот класс идеально подходит для нашей задачи.
class LoadH5d(object):
    """
    Пользовательский трансформер для загрузки данных из HDF5 файла.
    ГАРАНТИРУЕТ, что на выходе данные будут иметь 3 пространственных измерения.
    Если загружен 2D-срез (H, W), он будет преобразован в 3D-объем (1, H, W).
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            meta_key = f"{key}_meta"
            h5_path, h5_internal_key = d[meta_key]
            
            with h5py.File(h5_path, 'r') as hf:
                arr = hf[h5_internal_key][:]
            
            # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ: ПРОВЕРКА И ИСПРАВЛЕНИЕ РАЗМЕРНОСТИ ---
            if arr.ndim == 2:  # Если это 2D-изображение (H, W)
                # Добавляем измерение глубины, чтобы получить (1, H, W)
                print(f"Предупреждение: ключ '{h5_internal_key}' является 2D. Преобразую в 3D (1, H, W).")
                arr = np.expand_dims(arr, axis=0)
            
            # Проверка на другие некорректные размерности
            if arr.ndim != 3:
                raise ValueError(
                    f"Ошибка загрузки: ключ '{h5_internal_key}' имеет {arr.ndim} измерений. "
                    f"Этот пайплайн ожидает только 3 пространственных измерения."
                )
            
            d[key] = arr
        return d




def main():
    # --- Настройка параметров ---
    # !!! ЗАМЕНИТЕ ЭТОТ ПУТЬ НА ПУТЬ К ВАШЕМУ ФАЙЛУ !!!
    H5_FILE_PATH = r"C:\Users\pniki\Documents\Programs\Datasets\MED_proj_datasets\dataset_grouped.h5"
    
    PATCH_SIZE = (256, 256, 256)  # Размер патча для обучения. Подберите под вашу GPU.
    BATCH_SIZE = 2             # Количество патчей в батче.
    NUM_EPOCHS = 200            # Количество эпох для обучения.
    VAL_SPLIT = 0.2            # Доля данных для валидации (20%)
    output_dir = "validation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    # --- ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (ИСПОЛЬЗУЕМ ВАШУ ЛОГИКУ) ---
    print(f"Загрузка ключей из файла: {H5_FILE_PATH}")
    
    # 1. Собираем ключи, как в вашем коде
    all_keys_tuples = []
    try:
        with h5py.File(H5_FILE_PATH, 'r') as hf:
            for group_name in hf.keys():
                for key in hf[group_name].keys():
                    all_keys_tuples.append((group_name, key))
    except FileNotFoundError:
        print(f"ОШИБКА: Файл не найден по пути '{H5_FILE_PATH}'. Пожалуйста, укажите правильный путь.")
        return

    print(f"Найдено {len(all_keys_tuples)} 3D-объемов.")

    # 2. Преобразуем список кортежей в список словарей, который ожидает наш пайплайн
    all_files_dicts = [
        {
            "image_meta": (H5_FILE_PATH, f'{group_name}/{key}/image'),
            "mask_meta": (H5_FILE_PATH, f'{group_name}/{key}/mask')
        }
        for group_name, key in all_keys_tuples
    ]

    # 3. Разделяем данные на обучающую и валидационную выборки
    train_files, val_files = train_test_split(all_files_dicts[:], test_size=VAL_SPLIT, random_state=42)
    print(f"Данные разделены: {len(train_files)} для обучения, {len(val_files)} для валидации.")

    # --- Создание пайплайнов трансформеров MONAI (без изменений) ---
    train_transforms = Compose([
        LoadH5d(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        
        # --- ДОБАВЛЯЕМ АУГМЕНТАЦИИ ---
        # Случайное отражение по каждой из трех осей
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        # Случайные повороты на 90 градусов
        RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
        # Случайное изменение интенсивности (контраста)
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        # Случайный сдвиг интенсивности (яркости)
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        
        SpatialPadD(
            keys=["image"], 
            spatial_size=PATCH_SIZE, 
            method="end",
            mode="minimum"
        ),
        SpatialPadD(
            keys=["mask"], 
            spatial_size=PATCH_SIZE, 
            method="end", 
            mode="constant",
            constant_values=0  # или другое значение фона для маски
        ),
        RandCropByPosNegLabeld(
            keys=["image", "mask"], label_key="mask", spatial_size=PATCH_SIZE,
            pos=1, neg=1, num_samples=1, image_key="image", image_threshold=0,
        ),
    ])
    
    val_transforms = Compose([
        LoadH5d(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadD(
            keys=["image"], 
            spatial_size=PATCH_SIZE, 
            method="end",
            mode="minimum"
        ),
        SpatialPadD(
            keys=["mask"], 
            spatial_size=PATCH_SIZE, 
            method="end", 
            mode="constant",
            constant_values=0  # или другое значение фона для маски
        ),
                        RandCropByPosNegLabeld(
            keys=["image", "mask"], label_key="mask", spatial_size=PATCH_SIZE,
            pos=1, neg=1, num_samples=1, image_key="image", image_threshold=0,
        ),
    ])

    # --- Создание датасетов и загрузчиков (без изменений) ---
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # --- Настройка модели и цикла обучения (без изменений) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128,256, 512), strides=(2, 2,2, 2, 2, 2),
        num_res_units=2, dropout = 0.2
    ).to(device)
    
    
    # model = UNet(
    #     spatial_dims=3, in_channels=1, out_channels=2,
    #     channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
    #     num_res_units=2, dropout = 0.2
    # ).to(device)
    model.load_state_dict(torch.load("best_metric_model.pth"))
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(NUM_EPOCHS):
        print("-" * 20)
        print(f"Эпоха {epoch + 1}/{NUM_EPOCHS}")
        
        model.train()
        epoch_loss = 0
        
        for batch_data in tqdm(train_loader, desc="Обучение"):
            inputs, labels = batch_data[0]["image"].to(device), batch_data[0]["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(train_loader)
        
        print(f"Средняя потеря за эпоху: {epoch_loss:.4f}")

        model.eval()
        with torch.no_grad():
            # <<<--- ИЗМЕНЕНИЕ: Добавляем enumerate для отслеживания индекса ---<<<
            for i, val_data in enumerate(tqdm(val_loader, desc="Валидация")):
                val_inputs, val_labels = val_data[0]["image"].to(device), val_data[0]["mask"].to(device)
                
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, roi_size=PATCH_SIZE, sw_batch_size=1,
                    predictor=model, overlap=0.5, mode="gaussian"
                )
                
                # Обновляем общую метрику для эпохи
                val_labels_onehot = torch.nn.functional.one_hot(val_labels.squeeze(1).long(), num_classes=2).permute(0, 4, 1, 2, 3)
                dice_metric(y_pred=val_outputs, y=val_labels_onehot)

                # <<<--- НОВЫЙ БЛОК: Сохраняем картинку для первого сэмпла в эпохе ---<<<
                if i == 6:
                    # Перемещаем тензоры на CPU и убираем лишние измерения
                    image_slice = val_inputs.cpu().numpy().squeeze()
                    label_slice = val_labels.cpu().numpy().squeeze()
                    # Для выхода модели нужно взять argmax, чтобы получить маску класса
                    output_slice = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().squeeze()
                    
                    # Берем центральный срез по оси Z (глубине)
                    slice_idx = image_slice.shape[0] // 2

                    # Создаем фигуру для отрисовки
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    # 1. Исходное изображение

                    axes[0].imshow(image_slice[slice_idx, :, :], cmap="jet")
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")

                    # 2. Истинная маска
                    axes[1].imshow(label_slice[slice_idx, :, :], cmap="grey")
                    axes[1].set_title("Ground Truth")
                    axes[1].axis("off")
                    
                    # 3. Предсказание модели
                    axes[2].imshow(output_slice[slice_idx, :, :], cmap="grey")
                    axes[2].set_title(f"Model Prediction - Epoch {epoch + 1}")
                    axes[2].axis("off")
                    
                    # Сохраняем файл и закрываем фигуру, чтобы освободить память
                    plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1}.png"))
                    plt.close(fig)

            # Собираем и выводим среднюю метрику за эпоху
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Dice на валидации: {metric:.4f}")
            scheduler.step(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("Сохранена новая лучшая модель!")

    print(f"Обучение завершено. Лучший Dice: {best_metric:.4f} на эпохе {best_metric_epoch}")

if __name__ == "__main__":
    
    main()