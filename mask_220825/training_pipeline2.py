# training_pipeline2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from pathlib import Path

from models import CoarseUNet_Medium, MultiTask_FineUNet_MoE
from augmentations import get_augmentations_transform
from data_units import create_optimized_dataloaders
import h5py

class SimplifiedTrainer:
    
    def __init__(
        self, 
        h5_path: str,
        patch_size: tuple = (64, 64, 64),
        coarse_size: tuple = (32, 32, 32),
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        use_wandb: bool = False
    ):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.coarse_size = coarse_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Инициализация логирования
        if use_wandb:
            wandb.init(project="brain-segmentation", config={
                "patch_size": patch_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            })
        
        # Подготовка данных
        self._setup_data()
        
        # Подготовка моделей
        self._setup_models()
        
        # Подготовка оптимизаторов и функций потерь
        self._setup_training()
    
    def _setup_data(self):
        """Настройка данных и DataLoader'ов"""
        print("Настройка данных...")
        
        # Получаем количество модальностей
        
        with h5py.File(self.h5_path, 'r') as hf:
            self.num_modalities = len(list(hf.keys()))
        
        # Создаем аугментации
        augmentations = get_augmentations_transform()
        
        # Создаем оптимизированные DataLoader'ы
        self.train_loader, self.val_loader = create_optimized_dataloaders(
            h5_path=self.h5_path,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            augmentations=augmentations
        )
        
        print(f"Обучающих батчей: {len(self.train_loader)}")
        print(f"Валидационных батчей: {len(self.val_loader)}")
    
    def _setup_models(self):
        """Настройка моделей"""
        print("Инициализация моделей...")
        
        # Coarse модель - работает с уменьшенным разрешением
        self.coarse_model = CoarseUNet_Medium(
            in_channels=1, 
            out_channels=1, 
            base_filters=16
        ).to(self.device)
        
        # Fine модель - работает с полным разрешением патчей
        self.fine_model = MultiTask_FineUNet_MoE(
            in_channels=2,  # изображение + coarse маска
            out_channels_seg=1,
            num_classes=self.num_modalities,
            base_filters=16,
            num_experts=4
        ).to(self.device)
        
        # Подсчет параметров
        coarse_params = sum(p.numel() for p in self.coarse_model.parameters())
        fine_params = sum(p.numel() for p in self.fine_model.parameters())
        
        print(f"Coarse модель: {coarse_params:,} параметров")
        print(f"Fine модель: {fine_params:,} параметров")
        print(f"Всего: {coarse_params + fine_params:,} параметров")
    
    def _setup_training(self):
        """Настройка компонентов для обучения"""
        # Оптимизаторы
        self.optimizer_coarse = torch.optim.AdamW(
            self.coarse_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        self.optimizer_fine = torch.optim.AdamW(
            self.fine_model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Шедулеры
        self.scheduler_coarse = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_coarse, mode='min', patience=5, factor=0.5
        )
        
        self.scheduler_fine = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fine, mode='min', patience=5, factor=0.5
        )
        
        # Функции потерь
        self.criterion_seg = DiceBCELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        
        # Веса для multi-task learning
        self.seg_weight = 1.0
        self.cls_weight = 0.2
    
    def train_epoch(self) -> dict:
        """Обучение на одной эпохе"""
        self.coarse_model.train()
        self.fine_model.train()
        
        total_loss_coarse = 0.0
        total_loss_fine = 0.0
        total_loss_seg = 0.0
        total_loss_cls = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Обучение")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)  # (B, C, D, H, W)
            masks = batch['mask'].to(self.device)    # (B, C, D, H, W)
            modality_labels = batch['modality_label'].to(self.device)  # (B,)
            
            batch_size = images.size(0)
            
            # ===============================
            # ШАГ 1: COARSE МОДЕЛЬ
            # ===============================
            
            # Уменьшаем разрешение для coarse модели
            coarse_images = F.interpolate(
                images, size=self.coarse_size, 
                mode='trilinear', align_corners=False
            )
            coarse_masks = F.interpolate(
                masks, size=self.coarse_size, 
                mode='nearest'
            )
            
            # Обучаем coarse модель
            self.optimizer_coarse.zero_grad()
            coarse_preds = self.coarse_model(coarse_images)
            loss_coarse = self.criterion_seg(coarse_preds, coarse_masks)
            loss_coarse.backward()
            self.optimizer_coarse.step()
            
            # ===============================
            # ШАГ 2: FINE МОДЕЛЬ
            # ===============================
            
            # Апсемплинг предсказаний coarse модели до размера патча
            with torch.no_grad():
                upsampled_coarse = F.interpolate(
                    coarse_preds, size=images.shape[2:], 
                    mode='trilinear', align_corners=False
                )
            
            # Объединяем оригинальное изображение с coarse предсказанием
            fine_input = torch.cat([images, upsampled_coarse], dim=1)  # (B, 2, D, H, W)
            
            # Обучаем fine модель (мультизадачность)
            self.optimizer_fine.zero_grad()
            seg_preds, cls_preds = self.fine_model(fine_input)
            
            loss_seg = self.criterion_seg(seg_preds, masks)
            loss_cls = self.criterion_cls(cls_preds, modality_labels)
            
            total_loss_fine = (
                self.seg_weight * loss_seg + 
                self.cls_weight * loss_cls
            )
            
            total_loss_fine.backward()
            self.optimizer_fine.step()
            
            # ===============================
            # НАКОПЛЕНИЕ СТАТИСТИКИ
            # ===============================
            
            total_loss_coarse += loss_coarse.item()
            total_loss_fine += total_loss_fine.item()
            total_loss_seg += loss_seg.item()
            total_loss_cls += loss_cls.item()
            num_batches += 1
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({
                'L_coarse': f"{loss_coarse.item():.4f}",
                'L_seg': f"{loss_seg.item():.4f}",
                'L_cls': f"{loss_cls.item():.4f}"
            })
        
        # Возвращаем средние значения потерь
        return {
            'loss_coarse': total_loss_coarse / num_batches,
            'loss_fine': total_loss_fine / num_batches,
            'loss_seg': total_loss_seg / num_batches,
            'loss_cls': total_loss_cls / num_batches
        }
    
    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Валидация на одной эпохе"""
        self.coarse_model.eval()
        self.fine_model.eval()
        
        total_loss_coarse = 0.0
        total_loss_fine = 0.0
        total_loss_seg = 0.0
        total_loss_cls = 0.0
        num_batches = 0
        
        # Метрики для сегментации
        total_dice = 0.0
        total_iou = 0.0
        correct_cls = 0
        total_samples = 0
        
        progress_bar = tqdm(self.val_loader, desc="Валидация")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            modality_labels = batch['modality_label'].to(self.device)
            
            batch_size = images.size(0)
            
            # Coarse предсказания
            coarse_images = F.interpolate(
                images, size=self.coarse_size, 
                mode='trilinear', align_corners=False
            )
            coarse_masks = F.interpolate(
                masks, size=self.coarse_size, 
                mode='nearest'
            )
            coarse_preds = self.coarse_model(coarse_images)
            loss_coarse = self.criterion_seg(coarse_preds, coarse_masks)
            
            # Fine предсказания
            upsampled_coarse = F.interpolate(
                coarse_preds, size=images.shape[2:], 
                mode='trilinear', align_corners=False
            )
            fine_input = torch.cat([images, upsampled_coarse], dim=1)
            seg_preds, cls_preds = self.fine_model(fine_input)
            
            loss_seg = self.criterion_seg(seg_preds, masks)
            loss_cls = self.criterion_cls(cls_preds, modality_labels)
            total_loss_fine_batch = (
                self.seg_weight * loss_seg + 
                self.cls_weight * loss_cls
            )
            
            # Накопление потерь
            total_loss_coarse += loss_coarse.item()
            total_loss_fine += total_loss_fine_batch.item()
            total_loss_seg += loss_seg.item()
            total_loss_cls += loss_cls.item()
            
            # Вычисление метрик
            dice_score = self._compute_dice(seg_preds, masks)
            iou_score = self._compute_iou(seg_preds, masks)
            
            total_dice += dice_score
            total_iou += iou_score
            
            # Точность классификации
            _, predicted_cls = torch.max(cls_preds, 1)
            correct_cls += (predicted_cls == modality_labels).sum().item()
            total_samples += batch_size
            
            num_batches += 1
            
            progress_bar.set_postfix({
                'Dice': f"{dice_score:.3f}",
                'IoU': f"{iou_score:.3f}",
                'Cls_Acc': f"{correct_cls/total_samples:.3f}"
            })
        
        return {
            'loss_coarse': total_loss_coarse / num_batches,
            'loss_fine': total_loss_fine / num_batches,
            'loss_seg': total_loss_seg / num_batches,
            'loss_cls': total_loss_cls / num_batches,
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches,
            'cls_accuracy': correct_cls / total_samples
        }
    
    def _compute_dice(self, preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
        """Вычисляет Dice коэффициент"""
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()
    
    def _compute_iou(self, preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
        """Вычисляет IoU (Intersection over Union)"""
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """Сохраняет чекпоинт модели"""
        checkpoint = {
            'epoch': epoch,
            'coarse_model_state_dict': self.coarse_model.state_dict(),
            'fine_model_state_dict': self.fine_model.state_dict(),
            'optimizer_coarse_state_dict': self.optimizer_coarse.state_dict(),
            'optimizer_fine_state_dict': self.optimizer_fine.state_dict(),
            'scheduler_coarse_state_dict': self.scheduler_coarse.state_dict(),
            'scheduler_fine_state_dict': self.scheduler_fine.state_dict(),
            'val_metrics': val_metrics,
            'config': {
                'patch_size': self.patch_size,
                'coarse_size': self.coarse_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }
        }
        
        # Сохраняем обычный чекпоинт
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучший чекпоинт
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Сохранен лучший чекпоинт: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Загружает чекпоинт модели"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
        self.fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
        self.optimizer_coarse.load_state_dict(checkpoint['optimizer_coarse_state_dict'])
        self.optimizer_fine.load_state_dict(checkpoint['optimizer_fine_state_dict'])
        self.scheduler_coarse.load_state_dict(checkpoint['scheduler_coarse_state_dict'])
        self.scheduler_fine.load_state_dict(checkpoint['scheduler_fine_state_dict'])
        
        print(f"📂 Загружен чекпоинт с эпохи {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, resume_from: str = None):
        """Главный цикл обучения"""
        start_epoch = 0
        best_dice = 0.0
        
        # Восстановление обучения, если нужно
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"🚀 Начинаем обучение с эпохи {start_epoch + 1}")
        print(f"💾 Чекпоинты будут сохраняться в: {self.save_dir}")
        
        for epoch in range(start_epoch, self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Обучение
            train_metrics = self.train_epoch()
            
            # Валидация
            val_metrics = self.validate_epoch()
            
            # Обновляем шедулеры
            self.scheduler_coarse.step(val_metrics['loss_coarse'])
            self.scheduler_fine.step(val_metrics['loss_fine'])
            
            # Логирование
            print(f"\n📊 Результаты эпохи {epoch + 1}:")
            print(f"  Train - Coarse: {train_metrics['loss_coarse']:.4f}, "
                  f"Seg: {train_metrics['loss_seg']:.4f}, "
                  f"Cls: {train_metrics['loss_cls']:.4f}")
            print(f"  Val   - Dice: {val_metrics['dice']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, "
                  f"Cls Acc: {val_metrics['cls_accuracy']:.4f}")
            
            # Wandb логирование
            if wandb.run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss_coarse': train_metrics['loss_coarse'],
                    'train/loss_seg': train_metrics['loss_seg'],
                    'train/loss_cls': train_metrics['loss_cls'],
                    'val/loss_coarse': val_metrics['loss_coarse'],
                    'val/loss_seg': val_metrics['loss_seg'],
                    'val/loss_cls': val_metrics['loss_cls'],
                    'val/dice': val_metrics['dice'],
                    'val/iou': val_metrics['iou'],
                    'val/cls_accuracy': val_metrics['cls_accuracy'],
                    'lr_coarse': self.optimizer_coarse.param_groups[0]['lr'],
                    'lr_fine': self.optimizer_fine.param_groups[0]['lr'],
                })
            
            # Сохранение чекпоинтов
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
            
            # Сохраняем каждые 10 эпох или если это лучший результат
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_metrics, is_best)
        
        print(f"\n🎉 Обучение завершено!")
        print(f"🏆 Лучший Dice score: {best_dice:.4f}")


class DiceBCELoss(nn.Module):
    """Комбинированная Dice + BCE функция потерь"""
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
        # Sigmoid активация
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Flatten для вычислений
        inputs_flat = inputs_sigmoid.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Dice loss
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
            inputs_flat.sum() + targets_flat.sum() + smooth
        )
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs_sigmoid, targets, reduction='mean')
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# --- ИСПОЛЬЗОВАНИЕ ---

def main():
    """Главная функция для запуска обучения"""
    
    # Конфигурация
    config = {
        'h5_path': r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_prepared_golden.h5",
        'patch_size': (32, 32, 32),
        'coarse_size': (16, 16, 16),
        'batch_size': 2,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'save_dir': "./checkpoints",
        'use_wandb': False
    }
    
    # Создаем тренер
    trainer = SimplifiedTrainer(**config)
    
    # Запускаем обучение
    trainer.train()
    
    # Или восстанавливаем обучение с чекпоинта
    # trainer.train(resume_from="./checkpoints/checkpoint_epoch_3.pth")


if __name__ == "__main__":
    main()