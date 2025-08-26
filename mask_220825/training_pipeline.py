import os
import h5py
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import torch.nn.functional as F
import torchio as tio

# --- Импортируем наши кастомные модули ---
# Убедитесь, что эти файлы находятся в той же директории
from models import CoarseUNet_Medium
from models import PatchedFineModel,MultiTask_FineUNet_MoE
from augmentations import get_augmentations_transform

# --- КОНФИГУРАЦИЯ ТРЕНИРОВКИ ---
# ==============================================================================
# Путь к вашему готовому H5 файлу
H5_DATA_PATH = r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_prepared_golden.h5" 
# Для внутреннего патчинга BATCH_SIZE всегда должен быть 1!
BATCH_SIZE = 1
# Размер патчей, которые видит "внутренняя" Fine-модель
PATCH_SIZE = (128, 128, 128) 
# Перекрытие патчей для более гладкого результата (50% от размера патча)
PATCH_OVERLAP = (64, 64, 64) 
# Унифицированный размер для Coarse-модели. Должен быть кратен степени 2.
COARSE_UNIFIED_SIZE = (64, 64, 64) 
NUM_EPOCHS = 100           # Количество эпох для тренировки
LEARNING_RATE = 1e-4       # Скорость обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_ALPHA = 0.2           # Вес для функции потерь классификатора
NUM_WORKERS = 0            # Для Windows рекомендуется 0
# ==============================================================================
ACCUMULATION_STEPS = 16 

# --- ЗАГРУЗЧИК ДАННЫХ ---

class ModalityAwareH5Dataset(Dataset):
    """
    Dataset, который работает с модально-специфичным H5 файлом,
    обрабатывает многопоточность и применяет аугментации.
    Загружает ЦЕЛЫЕ 3D-объемы.
    """
    def __init__(self, h5_path, augmentations=None):
        self.h5_path = h5_path
        self.h5_file = None
        self.augmentations = augmentations
        
        with h5py.File(h5_path, 'r') as hf:
            self.modalities = sorted(list(hf.keys()))
            self.modality_map = {name: i for i, name in enumerate(self.modalities)}
            self.samples = []
            self.modality_indices = defaultdict(list)
            total_idx = 0
            for modality in self.modalities:
                num_samples = len(hf[modality]['images'])
                for i in range(num_samples):
                    self.samples.append((modality, i))
                    self.modality_indices[modality].append(total_idx)
                    total_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        modality_name, index_in_modality = self.samples[idx]
        modality_label = self.modality_map[modality_name]
        
        image = self.h5_file[modality_name]['images'][index_in_modality]
        mask = self.h5_file[modality_name]['masks'][index_in_modality]
        
        image = torch.from_numpy(image.transpose(3, 2, 0, 1))
        mask = torch.from_numpy(mask.transpose(3, 2, 0, 1))
        
        if self.augmentations:
            subject = tio.Subject(image=tio.ScalarImage(tensor=image), mask=tio.LabelMap(tensor=mask))
            transformed_subject = self.augmentations(subject)
            image, mask = transformed_subject.image.data, transformed_subject.mask.data
        
        return {
            'image': image.float(), 
            'mask': mask.float(), 
            'modality_label': torch.tensor(modality_label, dtype=torch.long)
        }

class ModalityBatchSampler(Sampler):
    """Сэмплер, который формирует батчи из данных одной модальности."""
    def __init__(self, modality_indices, batch_size, shuffle=True):
        self.modality_indices = modality_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.batches = []
        for modality in self.modality_indices:
            indices = self.modality_indices[modality]
            for i in range(len(indices) // batch_size):
                self.batches.append(indices[i * batch_size : (i + 1) * batch_size])

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        yield from self.batches

    def __len__(self):
        return len(self.batches)


# --- ФУНКЦИЯ ПОТЕРЬ ДЛЯ СЕГМЕНТАЦИИ ---

class DiceBCELoss(nn.Module):
    """Комбинированная Dice + BCE функция потерь."""
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss




# --- ГЛАВНАЯ ТРЕНИРОВОЧНАЯ ФУНКЦИЯ ---
def train_final_optimized():
    print(f"Используется устройство: {DEVICE}")
    
    # 1. Настройка данных
    augmentations = get_augmentations_transform()
    dataset = ModalityAwareH5Dataset(H5_DATA_PATH, augmentations=augmentations)
    sampler = ModalityBatchSampler(dataset.modality_indices, batch_size=BATCH_SIZE)
    data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=NUM_WORKERS)

    # 2. Инициализация моделей (теперь fine_model - это просто MultiTask_Fine_UNet_MoE)
    num_modalities = len(dataset.modalities)
    coarse_model = CoarseUNet_Medium().to(DEVICE)
    fine_model = MultiTask_FineUNet_MoE(num_classes=num_modalities).to(DEVICE)

    # 3. Настройка оптимизаторов
    optimizer_coarse = torch.optim.Adam(coarse_model.parameters(), lr=LEARNING_RATE)
    optimizer_fine = torch.optim.Adam(fine_model.parameters(), lr=LEARNING_RATE)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = DiceBCELoss()

    # 4. Тренировочный цикл
    for epoch in range(NUM_EPOCHS):
        coarse_model.train()
        fine_model.train()
        
        progress_bar = tqdm(data_loader, desc=f"Эпоха {epoch+1}", unit="scan")
        
        for batch in progress_bar:
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            modality_labels = batch['modality_label'].to(DEVICE)

            # --- Шаг 1: Coarse-модель (остается без изменений) ---
            coarse_images = F.interpolate(images, size=COARSE_UNIFIED_SIZE, mode='trilinear', align_corners=False)
            coarse_masks = F.interpolate(masks, size=COARSE_UNIFIED_SIZE, mode='nearest')
            optimizer_coarse.zero_grad()
            coarse_preds = coarse_model(coarse_images)
            loss_coarse = criterion_seg(coarse_preds, coarse_masks)
            loss_coarse.backward()
            optimizer_coarse.step()

            # --- Шаг 2: Fine-модель с ручным патчингом и аккумуляцией градиентов ---
            with torch.no_grad():
                upsampled_coarse_preds = F.interpolate(coarse_preds, size=images.shape[2:], mode='trilinear', align_corners=False)
            
            fine_input_full = torch.cat([images, upsampled_coarse_preds], dim=1)
            
            # ---> НАЧАЛО ИЗМЕНЕНИЙ
            
            pd, ph, pw = PATCH_SIZE
            sd, sh, sw = [s - o for s, o in zip(PATCH_SIZE, PATCH_OVERLAP)]
            *_, D, H, W = fine_input_full.shape
            
            # Сбрасываем градиенты ПЕРЕД обработкой нового большого скана
            optimizer_fine.zero_grad()
            
            patch_count = 0
            total_loss_fine_accum = 0.0

            # Цикл по патчам теперь здесь, в тренировочном цикле
            for d in range(0, D, sd):
                for h in range(0, H, sh):
                    for w in range(0, W, sw):
                        # Определяем границы, чтобы не выйти за пределы тензора
                        d_end, h_end, w_end = min(d+pd, D), min(h+ph, H), min(w+pw, W)
                        d_start, h_start, w_start = d_end-pd, h_end-ph, w_end-pw

                        # Вырезаем патчи
                        input_patch = fine_input_full[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        mask_patch = masks[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        
                        # Пропускаем ОДИН патч через модель
                        seg_logits_patch, cls_logits_patch = fine_model(input_patch)
                        
                        # Считаем потери для этого патча
                        loss_seg_patch = criterion_seg(seg_logits_patch, mask_patch)
                        loss_cls_patch = criterion_cls(cls_logits_patch, modality_labels)
                        
                        total_loss_patch = loss_seg_patch + LOSS_ALPHA * loss_cls_patch
                        
                        # Нормализуем loss, чтобы сумма была корректной
                        total_loss_patch = total_loss_patch / ACCUMULATION_STEPS
                        
                        # Вычисляем градиенты для этого патча и ДОБАВЛЯЕМ их к существующим
                        total_loss_patch.backward()
                        
                        total_loss_fine_accum += total_loss_patch.item()
                        patch_count += 1
                        
                        # Если мы накопили достаточно градиентов - делаем шаг оптимизатора
                        if patch_count % ACCUMULATION_STEPS == 0:
                            optimizer_fine.step()
                            # И СРАЗУ ЖЕ сбрасываем градиенты
                            optimizer_fine.zero_grad()

            # После обработки всех патчей может остаться "хвост" градиентов
            # Делаем финальный шаг, если нужно
            if patch_count % ACCUMULATION_STEPS != 0:
                optimizer_fine.step()
                optimizer_fine.zero_grad()

            # Обновляем информацию в прогресс-баре
            progress_bar.set_postfix(
                loss_coarse=f"{loss_coarse.item():.4f}", 
                avg_loss_fine=f"{total_loss_fine_accum / patch_count * ACCUMULATION_STEPS:.4f}"
            )
        # Здесь можно добавить логику сохранения чекпоинтов модели, например, каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            torch.save(coarse_model.state_dict(), f"coarse_model_epoch_{epoch+1}.pth")
            torch.save(fine_model.state_dict(), f"fine_model_epoch_{epoch+1}.pth")


if __name__ == '__main__':
    train_final_optimized()