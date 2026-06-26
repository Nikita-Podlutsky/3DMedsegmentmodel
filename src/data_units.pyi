# data_units.pyi
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import DataLoader

# =============================================================================
# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (UTILITIES) ---
# Низкоуровневые функции для манипуляции с 3D-объемами
# =============================================================================

def pad_or_crop_to_shape(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Приводит 3D объем к целевому размеру путем обрезки или паддинга по центру.

    Если текущий размер больше целевого, массив обрезается по центру.
    Если меньше — массив дополняется нулями (паддинг) так, чтобы
    оригинальные данные оказались в центре.

    Args:
        data (np.ndarray): Входной 3D NumPy массив.
        target_shape (Tuple[int, int, int]): Целевой размер (D, H, W).

    Returns:
        np.ndarray: Массив, приведенный к `target_shape`.
    """
    ...

def resize_volume(
    image_data: np.ndarray,
    mask_data: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Приводит 3D изображение и маску к единому размеру с помощью ресемплинга.

    Использует `scipy.ndimage.zoom` для масштабирования. Применяет кубическую
    интерполяцию (order=3) для изображения и интерполяцию по ближайшему соседу
    (order=0) для маски, чтобы сохранить четкие границы меток.

    Args:
        image_data (np.ndarray): 3D объем изображения.
        mask_data (np.ndarray): 3D объем маски.
        target_shape (Tuple[int, int, int]): Целевой размер (D, H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Кортеж из (обработанное_изображение, обработанная_маска).
    """
    ...

def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Выполняет Z-score нормализацию для МРТ по вокселям переднего плана.

    Отсекает 0.5% самых низких и 0.5% самых высоких интенсивностей, затем
    вычисляет среднее и стандартное отклонение по оставшимся ненулевым
    вокселям и применяет Z-score.

    Args:
        volume (np.ndarray): Входной МРТ-объем.

    Returns:
        np.ndarray: Нормализованный объем.
    """
    ...

# =============================================================================
# --- МОДУЛЬ 1: АНАЛИЗ И УНИФИКАЦИЯ РАЗМЕРА ---
# =============================================================================

def analyze_dataset_dimensions(dataset_dir: str | Path) -> pd.DataFrame | None:
    """
    Сканирует датасет и выводит статистику по размерностям и вокселям.

    Анализирует все .nii.gz файлы в указанной директории, собирает информацию
    о форме (shape) и размере вокселей (voxel size), и возвращает ее в виде
    DataFrame для дальнейшего анализа. Также выводит сводную статистику в консоль.

    Args:
        dataset_dir (str | Path): Путь к корневой директории датасета.

    Returns:
        pd.DataFrame | None: DataFrame со статистикой по каждому файлу или None,
                              если данные для анализа не найдены.
    """
    ...



# =============================================================================
# --- МОДУЛЬ 2: АНАЛИЗ ДАТАСЕТА SYNTHRAD ---
# =============================================================================

def analyze_synthrad_dataset(dataset_dir: str | Path) -> pd.DataFrame | None:
    """
    Сканирует датасет SynthRad и выводит статистику по размерностям и вокселям.

    Рекурсивно ищет все .nii.gz файлы, извлекает модальность из имени файла,
    а ID пациента из имени папки. Собирает статистику по размерам и разрешению.

    Args:
        dataset_dir (str | Path): Путь к корневой директории датасета SynthRad.

    Returns:
        pd.DataFrame | None: DataFrame со статистикой по каждому файлу или None,
                              если данные не найдены.
    """
    ...

# =============================================================================
# --- МОДУЛЬ 3: УНИВЕРСАЛЬНАЯ ПРЕДОБРАБОТКА ДЛЯ ИНФЕРЕНСА ---
# =============================================================================

def universal_preprocess_for_inference(
    raw_data: np.ndarray,
    modality: str,
    golden_standard_shapes: Dict[str, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Применяет каноническую предобработку к новому скану для подачи в модель.

    Приводит входной 3D-объем к стандартному размеру и нормализует его
    в соответствии с "золотым стандартом", определенным для обучающего датасета.

    Args:
        raw_data (np.ndarray): Входной 3D-объем в виде NumPy массива.
        modality (str): Модальность скана (например, 't1', 'ct').
        golden_standard_shapes (Dict): Словарь, сопоставляющий модальность
                                       с целевым размером (shape).

    Returns:
        np.ndarray: Обработанный 3D-объем, готовый для инференса.
    """
    ...

# =============================================================================
# --- МОДУЛЬ 4: ОПТИМИЗИРОВАННЫЙ ЗАГРУЗЧИК ДАННЫХ PYTORCH ---
# =============================================================================

class PrePatchedDataset(torch.utils.data.Dataset):
    """
    Оптимизированный Dataset, который предварительно нарезает данные на патчи.

    При первой инициализации создает и кэширует индексы всех возможных патчей
    из H5 файла. Во время обучения быстро загружает только нужный патч по индексу,
    что значительно ускоряет загрузку данных.
    """
    def __init__(
        self,
        h5_path: str,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: Tuple[int, int, int] = (64, 64, 64),
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        augmentations: Optional[tio.Transform] = None,
        patch_sampling_strategy: str = "sliding_window",
        patches_per_volume: int = 16,
        min_foreground_ratio: float = 0.1,
    ) -> None: ...

    def __len__(self) -> int:
        """Возвращает общее количество патчей в датасете."""
        ...

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Загружает, обрабатывает и возвращает один патч по его индексу."""
        ...

    def get_modality_counts(self) -> Dict[str, int]:
        """Возвращает количество патчей для каждой модальности."""
        ...

    def clear_cache(self) -> None:
        """Удаляет файл с кэшем индексов патчей."""
        ...


def create_optimized_dataloaders(
    h5_path: str,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 4,
    augmentations: Optional[tio.Transform] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Создает оптимизированные DataLoader'ы для обучения и валидации.

    Использует PrePatchedDataset для создания двух загрузчиков: один для
    обучения (с аугментациями и перемешиванием) и один для валидации.

    Args:
        h5_path (str): Путь к H5 файлу с предобработанными данными.
        patch_size (Tuple, optional): Размер патчей (D, H, W).
        batch_size (int, optional): Размер батча.
        augmentations (tio.Transform, optional): TorchIO трансформации для аугментаций.

    Returns:
        Tuple[DataLoader, DataLoader]: Кортеж из двух DataLoader'ов (train, validation).
    """
    ...

# =============================================================================
# --- МОДУЛЬ 6: ПРОДВИНУТАЯ, МОДАЛЬНО-ОРИЕНТИРОВАННАЯ ПОДГОТОВКА ДАННЫХ ---
# =============================================================================

def preprocess_synthstrip_volume(
    raw_data: np.ndarray,
    source_voxel_spacing: Tuple[float, float, float],
    config: Dict[str, Any],
    modality: str,
    is_mask: bool = False
) -> np.ndarray:
    """
    Выполняет полный пайплайн обработки для одного 3D объема (изображения или маски).

    Пайплайн включает:
    1. Ресемплинг до изотропного разрешения (например, 1x1x1 мм).
    2. Паддинг или обрезка до целевого размера, специфичного для модальности.
    3. Z-score нормализация (только для изображений).

    Args:
        raw_data (np.ndarray): Входной 3D-объем.
        source_voxel_spacing (Tuple): Исходное разрешение (размер вокселя).
        config (Dict): Глобальный конфигурационный словарь.
        modality (str): Модальность скана.
        is_mask (bool, optional): Если True, применяется интерполяция ближайшего соседа.

    Returns:
        np.ndarray: Полностью обработанный 3D-объем.
    """
    ...

def create_prepared_synthstrip_h5(
    dataset_dir: str | Path, h5_path: str | Path, config: Dict[str, Any]
) -> None:
    """
    Сканирует сырые данные, обрабатывает их и сохраняет в H5 файл по модальностям.

    Создает структурированный H5 файл, где данные для каждой модальности
    (t1, t2 и т.д.) хранятся в отдельной группе и обработаны согласно
    правилам, указанным в `config`.

    Args:
        dataset_dir (str | Path): Путь к сырым данным.
        h5_path (str | Path): Путь для сохранения готового H5 файла.
        config (Dict): Конфигурационный словарь с правилами обработки.
    """
    ...

def get_prepared_synthstrip_dataset(
    dataset_dir: str | Path,
    h5_cache_path: str | Path,
    config: Dict[str, Any],
    force_create: bool = False
) -> str:
    """
    Главная функция: создает или загружает готовый H5-файл SynthStrip.

    Если H5-файл уже существует, возвращает путь к нему. В противном случае,
    запускает процесс создания с помощью `create_prepared_synthstrip_h5`.

    Args:
        dataset_dir (str | Path): Путь к сырым данным.
        h5_cache_path (str | Path): Путь для сохранения/загрузки H5 файла.
        config (Dict): Конфигурационный словарь.
        force_create (bool, optional): Принудительно пересоздать файл.

    Returns:
        str: Путь к готовому для обучения H5 файлу.
    """
    ...