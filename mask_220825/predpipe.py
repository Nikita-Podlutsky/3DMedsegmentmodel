# File: predpipe.py

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    from models import CoarseUNet_Medium, MultiTask_FineUNet_MoE
    from inference import UnifiedPatchedModel
except ImportError as e:
    print(f"Ошибка импорта: {e}\nУбедитесь, что файлы models.py и inference.py находятся в той же директории.")
    exit()

# --- Вспомогательная функция ---

def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """Выполняет Z-score нормализацию для МРТ по вокселям переднего плана."""
    p0_5 = np.percentile(volume, 0.5)
    p99_5 = np.percentile(volume, 99.5)
    clipped = np.clip(volume, p0_5, p99_5)
    
    foreground_voxels = clipped[clipped > 1e-4]
    if len(foreground_voxels) == 0:
        return clipped
        
    mean = np.mean(foreground_voxels)
    std = np.std(foreground_voxels)
    
    return (clipped - mean) / std if std > 0 else clipped - mean

# --- Основной класс для продакшена ---

class BrainMasker:
    """
    Готовый к использованию класс для сегментации мозга (skull stripping).

    Инициализируется один раз путем загрузки обученного чекпоинта.
    Предоставляет простой метод .predict() для обработки новых NIfTI файлов.
    """
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Инициализирует и загружает все необходимые модели.

        Args:
            checkpoint_path (str): Путь к файлу чекпоинта (.pth).
            device (str, optional): Устройство для вычислений ('cuda', 'cpu'). 
                                    Если None, определяется автоматически.
        """
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Используемое устройство: {self.device}")

        print(f"Загрузка чекпоинта из {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.train_config = self.checkpoint['config']
        
        # Инициализация моделей с параметрами из чекпоинта
        coarse_model = CoarseUNet_Medium(base_filters=16).to(self.device)
        fine_in_channels = 2 + (3 if self.train_config.get('use_coord_maps', False) else 0)
        fine_model = MultiTask_FineUNet_MoE(in_channels=fine_in_channels, num_classes=8).to(self.device)

        # Инициализация главной inference-модели
        self.inference_model = UnifiedPatchedModel(
            coarse_model=coarse_model,
            fine_model=fine_model,
            coarse_size=self.train_config['coarse_input_size'],
            patch_size=self.train_config['fine_patch_size'],
            patch_overlap=self.train_config['fine_patch_overlap'],
            use_coord_maps=self.train_config.get('use_coord_maps', False)
        )
        
        self.inference_model.coarse_model.load_state_dict(self.checkpoint['coarse_model_state_dict'])
        self.inference_model.fine_model.load_state_dict(self.checkpoint['fine_model_state_dict'])
        self.inference_model.eval()
        print("Модели успешно загружены и готовы к работе.")

    def _preprocess(self, nifti_image: nib.Nifti1Image, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[torch.Tensor, Dict]:
        """Предобработка сырого NIfTI файла."""
        original_affine = nifti_image.affine
        original_shape = nifti_image.shape
        
        # 1. Получение данных и спейсинга
        raw_data = nifti_image.get_fdata().astype(np.float32)
        source_spacing = nifti_image.header.get_zooms()[:3]
        
        # 2. Ресемплинг до изотропного разрешения
        zoom_factor = [s / t for s, t in zip(source_spacing, target_spacing)]
        isotropic_data = zoom(raw_data, zoom_factor, order=3, mode='nearest')
        
        # 3. Нормализация
        normalized_data = normalize_mri(isotropic_data)
        
        # 4. Преобразование в тензор
        tensor = torch.from_numpy(normalized_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Сохраняем метаданные для пост-обработки
        metadata = {
            'original_shape': original_shape,
            'isotropic_shape': isotropic_data.shape
        }
        
        return tensor, metadata

    def _postprocess(self, seg_logits: torch.Tensor, metadata: Dict, threshold: float) -> np.ndarray:
        """Пост-обработка предсказания модели."""
        # 1. Применение порога
        probs = torch.sigmoid(seg_logits)
        binary_mask_iso = (probs > threshold)[0, 0].cpu().numpy()

        # 2. Ресемплинг обратно к оригинальному разрешению
        zoom_factor = [t / s for s, t in zip(metadata['isotropic_shape'], metadata['original_shape'])]
        final_mask = zoom(binary_mask_iso, zoom_factor, order=0, mode='nearest') # order=0 для масок!
        
        return final_mask.astype(np.uint8)

    def predict(
        self, 
        input_nifti_path: str, 
        output_nifti_path: Optional[str] = None, 
        threshold: float = 0.5,
        auto_invert: bool = True 
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Выполняет полный пайплайн сегментации для одного NIfTI файла.

        Args:
            input_nifti_path (str): Путь к входному .nii или .nii.gz файлу.
            output_nifti_path (str, optional): Путь для сохранения маски. 
                                               Если None, файл не сохраняется.
            threshold (float, optional): Порог бинаризации (от 0 до 1).

        Returns:
            Tuple[np.ndarray, Optional[str]]: 
                - Финальная бинарная маска в виде NumPy массива.
                - Путь к сохраненному файлу, если он был сохранен.
        """
        print(f"\n--- Начало обработки файла: {input_nifti_path} ---")
        
        # 1. Загрузка
        nifti_image = nib.load(input_nifti_path)
        
        # 2. Предобработка
        print("Предобработка изображения...")
        processed_tensor, metadata = self._preprocess(nifti_image)
        
        # 3. Предсказание
        print("Выполнение предсказания моделью...")
        with torch.no_grad():
            seg_logits, _ = self.inference_model(processed_tensor)
        
        # 4. Пост-обработка
        print("Пост-обработка результата...")
        final_mask_np = self._postprocess(seg_logits, metadata, threshold)
        
        if auto_invert:
            # Считаем, какую долю объема занимает предсказанная маска
            mask_volume_ratio = np.sum(final_mask_np) / final_mask_np.size
            print(f"Доля предсказанной маски в объеме: {mask_volume_ratio:.2%}")
            
            # Если маска занимает больше половины объема, это почти наверняка фон
            if mask_volume_ratio > 0.5:
                print("Объем маски > 50%. Применяется автоматическая инверсия.")
                final_mask_np = 1 - final_mask_np
        # 5. Сохранение (опционально)
        saved_path = None
        if output_nifti_path:
            # Создаем новый NIfTI объект, сохраняя аффинное преобразование и заголовок оригинала
            output_nifti = nib.Nifti1Image(final_mask_np, nifti_image.affine, nifti_image.header)
            nib.save(output_nifti, output_nifti_path)
            saved_path = output_nifti_path
            print(f"Маска успешно сохранена в: {saved_path}")
        
        print("--- Обработка завершена ---")
        return final_mask_np, saved_path



if __name__ == '__main__':
    
    CHECKPOINT_FILE = r"C:\Users\pniki\Documents\Programs\ML\Исследования\MEd\mask_220825\checkpoints\checkpoint_epoch_3.pth"
    INPUT_NIFTI_FILE = r"C:\Users\pniki\Documents\Programs\Datasets\SynthRad\23\1BA082\mr.nii.gz"
    OUTPUT_MASK_FILE = r"C:\Users\pniki\Documents\Programs\ML\Исследования\MEd\mask_220825\ct.nii.gz"
    
    # Рекомендуемый порог, который вы нашли с помощью скрипта анализа
    OPTIMAL_THRESHOLD = 0.577

    # --- ИСПОЛЬЗОВАНИЕ ---

    try:
        masker = BrainMasker(checkpoint_path=CHECKPOINT_FILE)

        brain_mask, saved_file = masker.predict(
            input_nifti_path=INPUT_NIFTI_FILE,
            output_nifti_path=OUTPUT_MASK_FILE,
            threshold=OPTIMAL_THRESHOLD
        )

        print(f"\nРазмер полученной маски: {brain_mask.shape}")
        

    except FileNotFoundError:
        print("\nОШИБКА: Один из указанных файлов не найден.")
        print("Пожалуйста, убедитесь, что пути в `CHECKPOINT_FILE`, `INPUT_NIFTI_FILE` и `OUTPUT_MASK_FILE` верны.")
    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {e}")