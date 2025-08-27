# data_units.py
# --- Стандартные библиотеки Python ---
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Сторонние библиотеки ---
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SYNTHSTRIP_GOLDEN_STANDARD_SHAPES = {
    'dwi': (128, 128, 64),
    'epi': (64, 64, 32),    
    'flair': (432, 512, 32),
    'mra': (1024, 1024, 112),
    'pd': (256, 256, 256),
    'qt1': (256, 256, 256),
    't1': (512, 512, 256),
    't2': (256, 256, 256)
}
SYNTHSTRIP_PROCESSING_CONFIG = {
    'target_voxel_spacing': (1.0, 1.0, 1.0),
    'modalities': {
        't1': {'target_shape': (256, 256, 256)},
        't2': {'target_shape': (256, 256, 256)},
        'pd': {'target_shape': (256, 256, 256)},
        'qt1': {'target_shape': (256, 256, 256)},
        'mra': {'target_shape': (384, 384, 192)},
        'dwi': {'target_shape': (128, 128, 64)},
        'epi': {'target_shape': (96, 96, 96)},
        'flair': {'target_shape': (256, 256, 256)},
        'mprage': {'target_shape': (256, 256, 256)},
    }
}




def resize_volume(image_data, mask_data, target_shape):
    zoom_factor = [t / s for s, t in zip(image_data.shape, target_shape)]
    resampled_image = zoom(image_data, zoom_factor, order=3, mode='nearest')
    resampled_mask = zoom(mask_data, zoom_factor, order=0, mode='nearest')
    final_image = pad_or_crop_to_shape(resampled_image, target_shape)
    final_mask = pad_or_crop_to_shape(resampled_mask, target_shape)
    return final_image, final_mask


def analyze_dataset_dimensions(dataset_dir):

    print("--- Начало анализа размерностей датасета ---")
    dataset_path = Path(dataset_dir)
    subject_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and len(d.name.split('_')) >= 3]
    
    stats = []
    
    for subject_dir in tqdm(subject_dirs, desc="Анализ сканов"):
        image_path = subject_dir / "image.nii.gz"
        if not image_path.exists():
            continue
            
        try:
            subset, modality, *subject_id_parts = subject_dir.name.split('_')

            nii_header = nib.load(image_path).header
            shape = nii_header.get_data_shape()
            voxel_size = nii_header.get_zooms()
            
            stats.append({
                'modality': modality,
                'subset': subset,
                'shape': shape,
                'voxel_size_x': voxel_size[0],
                'voxel_size_y': voxel_size[1],
                'voxel_size_z': voxel_size[2],
            })
        except Exception as e:
            print(f"Ошибка при анализе {subject_dir.name}: {e}")

    if not stats:
        print("Не найдено данных для анализа.")
        return

    df = pd.DataFrame(stats)
    
    print("\n--- Общая статистика по размерностям ---")
    print(df['shape'].describe())

    print("\n--- 5 самых частых размерностей ---")
    most_common_shapes = Counter(df['shape']).most_common(5)
    for shape, count in most_common_shapes:
        print(f"Размер: {shape}, Количество: {count}")

    print("\n--- Статистика по размерностям в разрезе модальностей ---")
    
    shape_df = pd.DataFrame(df['shape'].tolist(), columns=['dim_x', 'dim_y', 'dim_z'])
    modality_shape_df = pd.concat([df['modality'], shape_df], axis=1)
    
    print(modality_shape_df.groupby('modality').agg(['mean', 'median', 'min', 'max']).round(1))
    
    print("\nАнализ завершен. Используйте эту информацию, чтобы выбрать оптимальный target_shape.")
    return df


def analyze_synthrad_dataset(dataset_dir):

    print(f"--- Начало анализа датасета SynthRad в '{dataset_dir}' ---")
    dataset_path = Path(dataset_dir)

    nii_files = list(dataset_path.glob('**/*.nii.gz'))
    
    if not nii_files:
        print("Не найдено ни одного .nii.gz файла. Проверьте путь.")
        return None

    print(f"Найдено {len(nii_files)} NIfTI файлов для анализа...")
    
    stats = []
    
    for file_path in tqdm(nii_files, desc="Анализ сканов"):
        try:
            modality = file_path.stem
            
            patient_id = file_path.parent.name
            
            header = nib.load(file_path).header
            shape = header.get_data_shape()
            voxel_size = header.get_zooms()
            
            stats.append({
                'modality': modality,
                'patient_id': patient_id,
                'shape': shape,
                'path': str(file_path),
                'voxel_size_x': voxel_size[0],
                'voxel_size_y': voxel_size[1],
                'voxel_size_z': voxel_size[2],
            })
        except Exception as e:
            print(f"Ошибка при анализе файла {file_path}: {e}")

    if not stats:
        print("Не удалось собрать статистику ни по одному файлу.")
        return None

    df = pd.DataFrame(stats)
    
    print("\n--- Общая статистика по размерностям ---")
    # Для describe нужно преобразовать кортежи в отдельные столбцы
    shape_df = pd.DataFrame(df['shape'].tolist(), columns=['dim_x', 'dim_y', 'dim_z'])
    print(shape_df.describe().round(1))

    print("\n--- 5 самых частых размерностей ---")
    most_common_shapes = Counter(df['shape']).most_common(5)
    for shape, count in most_common_shapes:
        print(f"Размер: {shape}, Количество: {count}")

    print("\n--- Статистика по размерностям в разрезе модальностей ---")
    modality_shape_df = pd.concat([df['modality'], shape_df], axis=1)
    
    # Выводим подробную статистику
    grouped_stats = modality_shape_df.groupby('modality').agg(['mean', 'median', 'min', 'max', 'count']).round(1)
    print(grouped_stats)
    
    print("\nАнализ завершен. Теперь можно принимать решение о дальнейшей обработке.")
    return df


def normalize_golden_standard(volume, modality):
    # modality_clean = modality.split('.')[0]
    
    return normalize_mri(volume)


def universal_preprocess_for_inference(raw_data, modality, golden_standard_shapes):

    modality_clean = modality.split('.')[0]
    
    try:
        target_shape = golden_standard_shapes[modality_clean]
    except KeyError:
        raise ValueError(
            f"Неизвестная модальность '{modality_clean}'! "
            f"Невозможно определить целевой размер для паддинга. "
            f"Доступные модальности: {list(golden_standard_shapes.keys())}"
        )
    
    if any(cs > ts for cs, ts in zip(raw_data.shape, target_shape)):
        print(f"ВНИМАНИЕ: Скан {raw_data.shape} больше целевого размера {target_shape}. "
              "Будет применена обрезка (cropping) по центру.")

        from scipy.ndimage import zoom

        zoom_factor = [t / s for s, t in zip(raw_data.shape, target_shape)]
        print("Применяется resize вместо обрезки. Рекомендуется реализовать обрезку.")
        processed_data = zoom(raw_data, zoom_factor, order=3, mode='nearest')
        processed_data = pad_or_crop_to_shape(processed_data, target_shape)
    else:
        processed_data = pad_or_crop_to_shape(raw_data, target_shape)

    normalized_data = normalize_golden_standard(processed_data, modality_clean)
    
    return normalized_data



class PrePatchedDataset(Dataset):

    
    def __init__(
        self, 
        h5_path: str,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: Tuple[int, int, int] = (64, 64, 64),
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        augmentations: Optional[Any] = None,
        patch_sampling_strategy: str = "sliding_window",  # "sliding_window", "random", "foreground_focused"
        patches_per_volume: int = 16,  # для random и foreground_focused
        min_foreground_ratio: float = 0.1,  # минимальный процент foreground в патче
    ):

        self.h5_path = h5_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.augmentations = augmentations
        self.patch_sampling_strategy = patch_sampling_strategy
        self.patches_per_volume = patches_per_volume
        self.min_foreground_ratio = min_foreground_ratio

        if cache_dir is None:
            cache_dir = Path(h5_path).parent / f"{Path(h5_path).stem}_patches"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        cache_name = f"patches_{patch_size}_{overlap}_{patch_sampling_strategy}_{patches_per_volume}.pkl"
        self.cache_path = self.cache_dir / cache_name

        with h5py.File(h5_path, 'r') as hf:
            self.modalities = sorted(list(hf.keys()))
            self.modality_map = {name: i for i, name in enumerate(self.modalities)}

        if force_rebuild or not self.cache_path.exists():
            print("Создание индексов патчей...")
            self.patch_indices = self._create_patch_indices()
            self._save_patch_indices()
        else:
            print("Загрузка существующих индексов патчей...")
            self.patch_indices = self._load_patch_indices()
        
        print(f"Всего патчей в датасете: {len(self.patch_indices)}")
        self._print_patch_statistics()
    
    def _create_patch_indices(self) -> List[Dict]:

        patch_indices = []
        
        with h5py.File(self.h5_path, 'r') as hf:
            for modality in tqdm(self.modalities, desc="Обработка модальностей"):
                num_volumes = len(hf[modality]['images'])
                
                for vol_idx in tqdm(range(num_volumes), desc=f"Модальность {modality}", leave=False):

                    volume_shape = hf[modality]['images'][vol_idx].shape[:3]
                    mask_data = hf[modality]['masks'][vol_idx][..., 0]
                    
                    if self.patch_sampling_strategy == "sliding_window":
                        volume_patches = self._extract_sliding_window_patches(
                            volume_shape, vol_idx, modality, mask_data
                        )
                    elif self.patch_sampling_strategy == "random":
                        volume_patches = self._extract_random_patches(
                            volume_shape, vol_idx, modality, mask_data
                        )
                    elif self.patch_sampling_strategy == "foreground_focused":
                        volume_patches = self._extract_foreground_focused_patches(
                            volume_shape, vol_idx, modality, mask_data
                        )
                    else:
                        raise ValueError(f"Неизвестная стратегия: {self.patch_sampling_strategy}")
                    
                    patch_indices.extend(volume_patches)
        
        return patch_indices
    
    def _extract_sliding_window_patches(
        self, 
        volume_shape: Tuple[int, int, int], 
        vol_idx: int, 
        modality: str,
        mask_data: np.ndarray
    ) -> List[Dict]:

        patches = []
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size
        od, oh, ow = self.overlap
        sd, sh, sw = pd - od, ph - oh, pw - ow
        
        for d in range(0, D - pd + 1, sd):
            for h in range(0, H - ph + 1, sh):
                for w in range(0, W - pw + 1, sw):

                    patch_mask = mask_data[d:d+pd, h:h+ph, w:w+pw]
                    foreground_ratio = np.sum(patch_mask > 0) / patch_mask.size
                    
                    if foreground_ratio >= self.min_foreground_ratio:
                        patches.append({
                            'modality': modality,
                            'volume_idx': vol_idx,
                            'patch_coords': (d, h, w),
                            'patch_size': self.patch_size,
                            'foreground_ratio': foreground_ratio,
                            'modality_label': self.modality_map[modality]
                        })
        
        return patches
    
    def _extract_random_patches(
        self, 
        volume_shape: Tuple[int, int, int], 
        vol_idx: int, 
        modality: str,
        mask_data: np.ndarray
    ) -> List[Dict]:
        
        patches = []
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size
        
        attempts = 0
        max_attempts = self.patches_per_volume * 10
        
        while len(patches) < self.patches_per_volume and attempts < max_attempts:

            d = np.random.randint(0, max(1, D - pd + 1))
            h = np.random.randint(0, max(1, H - ph + 1))
            w = np.random.randint(0, max(1, W - pw + 1))

            patch_mask = mask_data[d:d+pd, h:h+ph, w:w+pw]
            foreground_ratio = np.sum(patch_mask > 0) / patch_mask.size
            
            if foreground_ratio >= self.min_foreground_ratio:
                patches.append({
                    'modality': modality,
                    'volume_idx': vol_idx,
                    'patch_coords': (d, h, w),
                    'patch_size': self.patch_size,
                    'foreground_ratio': foreground_ratio,
                    'modality_label': self.modality_map[modality]
                })
            
            attempts += 1
        
        return patches
    
    def _extract_foreground_focused_patches(
        self, 
        volume_shape: Tuple[int, int, int], 
        vol_idx: int, 
        modality: str,
        mask_data: np.ndarray
    ) -> List[Dict]:

        patches = []
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size

        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(mask_data > 0)
        
        if num_features == 0:
            return self._extract_random_patches(volume_shape, vol_idx, modality, mask_data)
        
        centers_of_mass = ndimage.center_of_mass(mask_data, labeled_mask, range(1, num_features + 1))
        if not isinstance(centers_of_mass, list):
            centers_of_mass = [centers_of_mass]

        for _ in range(self.patches_per_volume):
            if len(centers_of_mass) > 0:

                center = centers_of_mass[np.random.randint(len(centers_of_mass))]
                cd, ch, cw = [int(c) for c in center]
                
                jitter = 32
                d = np.clip(cd - pd//2 + np.random.randint(-jitter, jitter), 0, D - pd)
                h = np.clip(ch - ph//2 + np.random.randint(-jitter, jitter), 0, H - ph)
                w = np.clip(cw - pw//2 + np.random.randint(-jitter, jitter), 0, W - pw)
                
                patch_mask = mask_data[d:d+pd, h:h+ph, w:w+pw]
                foreground_ratio = np.sum(patch_mask > 0) / patch_mask.size
                
                patches.append({
                    'modality': modality,
                    'volume_idx': vol_idx,
                    'patch_coords': (d, h, w),
                    'patch_size': self.patch_size,
                    'foreground_ratio': foreground_ratio,
                    'modality_label': self.modality_map[modality]
                })
        
        return patches
    
    def _save_patch_indices(self):
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'patch_indices': self.patch_indices,
                'modalities': self.modalities,
                'modality_map': self.modality_map,
                'patch_size': self.patch_size,
                'overlap': self.overlap,
                'strategy': self.patch_sampling_strategy
            }, f)
        print(f"Индексы патчей сохранены в {self.cache_path}")
    
    def _load_patch_indices(self) -> List[Dict]:

        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Проверяем совместимость
        if (data['patch_size'] != self.patch_size or 
            data['overlap'] != self.overlap or
            data['strategy'] != self.patch_sampling_strategy):
            print("Параметры изменились, пересоздаем индексы...")
            return self._create_patch_indices()
        
        return data['patch_indices']
    
    def _print_patch_statistics(self):

        modality_counts = defaultdict(int)
        foreground_ratios = []
        
        for patch_info in self.patch_indices:
            modality_counts[patch_info['modality']] += 1
            foreground_ratios.append(patch_info['foreground_ratio'])

        print("\n--- Статистика патчей ---")
        for modality, count in modality_counts.items():
            print(f"{modality}: {count} патчей")
        
        print(f"Средний % foreground: {np.mean(foreground_ratios):.2%}")
        print(f"Медианный % foreground: {np.median(foreground_ratios):.2%}")
        print("-------------------------\n")
    
    def __len__(self) -> int:
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        patch_info = self.patch_indices[idx]
        
        # Извлекаем информацию о патче
        modality = patch_info['modality']
        volume_idx = patch_info['volume_idx']
        d, h, w = patch_info['patch_coords']
        pd, ph, pw = patch_info['patch_size']
        

        with h5py.File(self.h5_path, 'r') as hf:

            image_patch = hf[modality]['images'][volume_idx][d:d+pd, h:h+ph, w:w+pw, :]
            mask_patch = hf[modality]['masks'][volume_idx][d:d+pd, h:h+ph, w:w+pw, :]

        image_tensor = torch.from_numpy(image_patch.transpose(3, 2, 1, 0)).float()  # (C, W, H, D)
        mask_tensor = torch.from_numpy(mask_patch.transpose(3, 2, 1, 0)).float()    # (C, W, H, D)

        if self.augmentations:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor),
                mask=tio.LabelMap(tensor=mask_tensor)
            )
            transformed_subject = self.augmentations(subject)
            image_tensor = transformed_subject.image.data
            mask_tensor = transformed_subject.mask.data
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'modality_label': torch.tensor(patch_info['modality_label'], dtype=torch.long),
            'foreground_ratio': torch.tensor(patch_info['foreground_ratio'], dtype=torch.float),
            'modality_name': modality,
            'volume_idx': volume_idx,
            'patch_coords': patch_info['patch_coords']
        }
    
    def get_modality_counts(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for patch_info in self.patch_indices:
            counts[patch_info['modality']] += 1
        return dict(counts)
    
    def get_patches_by_modality(self, modality: str) -> List[int]:
        return [i for i, patch_info in enumerate(self.patch_indices) 
                if patch_info['modality'] == modality]
    
    def clear_cache(self):
        if self.cache_path.exists():
            self.cache_path.unlink()
            print("Кэш патчей очищен")


def create_optimized_dataloaders(
    h5_path: str,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 4,
    augmentations = None
) -> Tuple[DataLoader, DataLoader]:


    train_dataset = PrePatchedDataset(
        h5_path=h5_path,
        patch_size=patch_size,
        patch_sampling_strategy="sliding_window",
        augmentations=augmentations,
        min_foreground_ratio=0.00,
        overlap=(32, 32, 32)
    )[:10]
    
    val_dataset = PrePatchedDataset(
        h5_path=h5_path,
        patch_size=patch_size,
        patch_sampling_strategy="sliding_window",
        overlap=(32, 32, 32),
        augmentations=None,
        min_foreground_ratio=0.0
    )[:10]

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def pad_or_crop_to_shape(data, target_shape):

    current_shape = data.shape
    result = np.zeros(target_shape, dtype=data.dtype)
    
    slices_in = tuple(
        slice((cs - ts) // 2, (cs - ts) // 2 + ts) if cs > ts else slice(None)
        for cs, ts in zip(current_shape, target_shape)
    )
    slices_out = tuple(
        slice((ts - cs) // 2, (ts - cs) // 2 + cs) if cs < ts else slice(None)
        for cs, ts in zip(current_shape, target_shape)
    )
    
    result[slices_out] = data[slices_in]
    return result

def normalize_mri(volume):

    p0_5 = np.percentile(volume, 0.5)
    p99_5 = np.percentile(volume, 99.5)
    clipped = np.clip(volume, p0_5, p99_5)
    
    foreground_voxels = clipped[clipped > 1e-4]
    if len(foreground_voxels) == 0:
        return clipped
        
    mean = np.mean(foreground_voxels)
    std = np.std(foreground_voxels)
    
    return (clipped - mean) / std if std > 0 else clipped - mean

def preprocess_synthstrip_volume(raw_data, source_voxel_spacing, config, modality, is_mask=False):

    target_voxel_spacing = config['target_voxel_spacing']
    zoom_factor = [s / t for s, t in zip(source_voxel_spacing, target_voxel_spacing)]

    interp_order = 0 if is_mask else 3
    isotropic_volume = zoom(raw_data, zoom_factor, order=interp_order, mode='nearest')

    target_shape = config['modalities'][modality]['target_shape']
    processed_volume = pad_or_crop_to_shape(isotropic_volume, target_shape)
    
    if not is_mask:
        processed_volume = normalize_mri(processed_volume)
        
    return processed_volume.astype(np.float32 if not is_mask else np.uint8)



def create_prepared_synthstrip_h5(dataset_dir, h5_path, config):

    dataset_path = Path(dataset_dir)
    subject_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and len(d.name.split('_')) >= 3]

    print("Этап 1: Предварительное сканирование для подсчета сэмплов...")
    modality_counts = defaultdict(int)
    modality_paths = defaultdict(list)
    for subject_dir in tqdm(subject_dirs, desc="Сканирование"):
        try:
            modality = subject_dir.name.split('_')[1].lower()
            if modality not in config['modalities']:
                if 'mprage' in modality and 'mprage' in config['modalities']: modality = 'mprage'
                else: continue

            if not (subject_dir / "image.nii.gz").exists() or not (subject_dir / "mask.nii.gz").exists():
                continue

            modality_counts[modality] += 1
            modality_paths[modality].append(subject_dir)
        except Exception:
            continue

    print("Этап 2: Инициализация H5 файла...")
    with h5py.File(h5_path, 'w') as hf:
        hf.attrs['target_voxel_spacing'] = config['target_voxel_spacing']
        
        for modality, count in modality_counts.items():
            print(f"  - Создание группы '{modality}' на {count} сэмплов...")
            modality_group = hf.create_group(modality)
            target_shape = config['modalities'][modality]['target_shape']
            hf.attrs[f'shape_{modality}'] = target_shape

            img_shape = (count, *target_shape, 1)
            mask_shape = (count, *target_shape, 1)
            
            modality_group.create_dataset('images', shape=img_shape, dtype=np.float32, chunks=True)
            modality_group.create_dataset('masks', shape=mask_shape, dtype=np.uint8, chunks=True)
            modality_group.create_dataset('sources', shape=(count,), dtype=h5py.string_dtype(encoding='utf-8'))

    print("Этап 3: Обработка и запись данных в H5 файл...")
    with h5py.File(h5_path, 'a') as hf:
        modality_indices = defaultdict(int)
        
        for modality, paths in modality_paths.items():
            for subject_dir in tqdm(paths, desc=f"Запись модальности '{modality}'"):
                try:
                    image_path = subject_dir / "image.nii.gz"
                    mask_path = subject_dir / "mask.nii.gz"
                    
                    image_nii = nib.load(image_path)
                    mask_nii = nib.load(mask_path)
                    
                    image_data = image_nii.get_fdata().astype(np.float32)
                    mask_data = mask_nii.get_fdata().astype(np.uint8)
                    voxel_spacing = image_nii.header.get_zooms()

                    if image_data.ndim == 4: image_data = image_data[..., 0]
                    if mask_data.ndim == 4: mask_data = mask_data[..., 0]
                    if image_data.ndim != 3 or mask_data.ndim != 3: continue

                    processed_image = preprocess_synthstrip_volume(image_data, voxel_spacing, config, modality, is_mask=False)
                    processed_mask = preprocess_synthstrip_volume(mask_data, voxel_spacing, config, modality, is_mask=True)
    
                    current_idx = modality_indices[modality]
                    hf[modality]['images'][current_idx, ..., 0] = processed_image
                    hf[modality]['masks'][current_idx, ..., 0] = processed_mask
                    hf[modality]['sources'][current_idx] = str(subject_dir)
                    
                    modality_indices[modality] += 1
                except Exception as e:
                    print(f"\nОшибка при обработке {subject_dir.name} на этапе записи: {e}")

    print("\n--- Подготовка SynthStrip завершена! ---")


def get_prepared_synthstrip_dataset(dataset_dir, h5_cache_path, config, force_create=False):

    if os.path.exists(h5_cache_path) and not force_create:
        print(f"Найден готовый файл. Загрузка из {h5_cache_path}...")

        return h5_cache_path
    else:
        print("Готовый файл не найден или принудительное создание. Начинаем подготовку...")
        create_prepared_synthstrip_h5(dataset_dir, h5_cache_path, config)
        return h5_cache_path




class FullImageDataset(Dataset):
    """
    Dataset, который загружает и возвращает полные 3D-изображения из H5 файла.
    Предназначен для продвинутого обучения, где требуется глобальный контекст.
    """
    def __init__(self, h5_path: str, augmentations: Optional[Any] = None):
        self.h5_path = h5_path
        self.augmentations = augmentations
        
        self.volumes_info = []
        with h5py.File(h5_path, 'r') as hf:
            self.modalities = sorted(list(hf.keys()))
            self.modality_map = {name: i for i, name in enumerate(self.modalities)}
            
            print("Сканирование полных объемов для FullImageDataset...")
            for modality in self.modalities:
                num_volumes = len(hf[modality]['images'])
                for vol_idx in range(num_volumes):
                    self.volumes_info.append({
                        'modality': modality,
                        'volume_idx': vol_idx,
                        'modality_label': self.modality_map[modality]
                    })
        print(f"Найдено {len(self.volumes_info)} полных 3D-объемов.")

    def __len__(self) -> int:
        return 2
        return len(self.volumes_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.volumes_info[idx]
        modality = info['modality']
        volume_idx = info['volume_idx']

        with h5py.File(self.h5_path, 'r') as hf:

            image_volume = hf[modality]['images'][volume_idx][...]
            mask_volume = hf[modality]['masks'][volume_idx][...]


        image_tensor = torch.from_numpy(image_volume.transpose(3, 0, 1, 2)).float()
        mask_tensor = torch.from_numpy(mask_volume.transpose(3, 0, 1, 2)).float()

        if self.augmentations:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor),
                mask=tio.LabelMap(tensor=mask_tensor)
            )
            transformed = self.augmentations(subject)
            image_tensor = transformed.image.data
            mask_tensor = transformed.mask.data

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'modality_label': torch.tensor(info['modality_label'], dtype=torch.long),
            'modality_name': modality
        }