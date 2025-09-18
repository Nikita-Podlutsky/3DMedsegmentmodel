import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any, Optional
import wandb

try:
    from models import CoarseUNet_Medium, MultiTask_FineUNet_MoE, MoETrainingManager
    from data_units import FullImageDataset
    from inference import UnifiedPatchedModel
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    exit()




class DiceBCELoss(nn.Module):
    
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):

        inputs_sigmoid = torch.sigmoid(inputs)
        

        inputs_flat = inputs_sigmoid.reshape(-1)
        targets_flat = targets.reshape(-1)
        

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
            inputs_flat.sum() + targets_flat.sum() + smooth
        )

        bce_loss = F.binary_cross_entropy(inputs_sigmoid, targets, reduction='mean')
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss



def create_coordinate_maps(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    Создает 3-канальный тензор, кодирующий координаты для каждого вокселя.
    Координаты нормализованы в диапазоне [-1, 1].
    """
    d, h, w = shape
    d_coords = torch.linspace(-1, 1, d, device=device)
    h_coords = torch.linspace(-1, 1, h, device=device)
    w_coords = torch.linspace(-1, 1, w, device=device)
    
    d_map = d_coords.view(d, 1, 1).expand(d, h, w)
    h_map = h_coords.view(1, h, 1).expand(d, h, w)
    w_map = w_coords.view(1, 1, w).expand(d, h, w)
    
    coord_maps = torch.stack([d_map, h_map, w_map], dim=0).unsqueeze(0)
    return coord_maps

def sample_random_patch(
    image_vol: torch.Tensor,
    mask_vol: torch.Tensor,
    coarse_map_vol: torch.Tensor,
    coord_maps_vol: Optional[torch.Tensor],
    patch_size: Tuple[int, int, int]
) -> Tuple[torch.Tensor, ...]:
    """Вспомогательная функция для вырезания случайного патча из полных объемов."""
    *_, D, H, W = image_vol.shape
    pd, ph, pw = patch_size

    if D < pd or H < ph or W < pw:
        raise ValueError(f"Размер патча {patch_size} больше размера изображения {(D,H,W)}")

    d = torch.randint(0, D - pd + 1, (1,)).item()
    h = torch.randint(0, H - ph + 1, (1,)).item()
    w = torch.randint(0, W - pw + 1, (1,)).item()
    
    image_patch = image_vol[:, :, d:d+pd, h:h+ph, w:w+pw]
    mask_patch = mask_vol[:, :, d:d+pd, h:h+ph, w:w+pw]
    coarse_map_patch = coarse_map_vol[:, :, d:d+pd, h:h+ph, w:w+pw]
    
    coord_patch = None
    if coord_maps_vol is not None:
        coord_patch = coord_maps_vol[:, :, d:d+pd, h:h+ph, w:w+pw]
    
    return image_patch, mask_patch, coarse_map_patch, coord_patch


class AdvancedTrainer:
    """
    Реализует продвинутый пайплайн обучения с глобальным контекстом
    и корректной валидацией.
    """
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        self.config = config
        self.device = torch.device(config['device'])

        print("Настройка данных...")
        dataset = FullImageDataset(config['h5_path'])
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(config['validation_split'] * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=train_sampler, num_workers=config.get('num_workers', 2))
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=val_sampler, num_workers=config.get('num_workers', 2))
        print(f"Данные разделены: {len(train_indices)} train, {len(val_indices)} validation.")

        if config["use_wandb"]:
            wandb.init(project="brain-segmentation", config={
                "patch_size": config["fine_patch_size"],
                "batch_size": 1,
                "learning_rate": config["learning_rate"],
                "num_epochs": config["num_epochs"]
            })

        print("Инициализация моделей...")
        self.coarse_model = CoarseUNet_Medium(base_filters=16).to(self.device)
        
        fine_in_channels = 2
        if config['use_coord_maps']:
            fine_in_channels += 3
            print("Обучение с явным позиционным кодированием (координатные карты).")
            
        self.fine_model = MultiTask_FineUNet_MoE(
            in_channels=fine_in_channels,
            num_classes=len(dataset.modalities)
        ).to(self.device)
        self.moe_manager = MoETrainingManager(self.fine_model, load_balance_weight=0.01)

        
        

        if config['use_coord_maps']:
            self.coord_maps_cache = {}
        
        print("Настройка компонентов обучения...")
        params = list(self.coarse_model.parameters()) + list(self.fine_model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5, verbose=True
        )
        self.criterion_seg = DiceBCELoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        
        self.inference_model = UnifiedPatchedModel(
            coarse_model=self.coarse_model,
            fine_model=self.fine_model,
            coarse_size=config['coarse_input_size'],
            patch_size=config['fine_patch_size'],
            patch_overlap=config['fine_patch_overlap'],
            use_coord_maps=config['use_coord_maps']
        )
        self.best_val_dice = 0.0
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        if config["model_checkpoint_path"]:
            
            save_point = torch.load(config["model_checkpoint_path"])
            self.coarse_model.load_state_dict(save_point["coarse_model_state_dict"])
            self.fine_model.load_state_dict(save_point["fine_model_state_dict"])
            if config["load_optim"]:
                self.optimizer.load_state_dict(save_point["optimizer_state_dict"])
                self.scheduler.load_state_dict(save_point["scheduler_state_dict"])

        print("AdvancedTrainer готов к работе.")

    def train_epoch(self):
        self.moe_manager.reset_stats_if_needed(self.current_epoch)
        self.coarse_model.train()
        self.fine_model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Обучение Эпоха {self.current_epoch}")
        
        for batch in progress_bar:
            full_image = batch['image'].to(self.device)
            full_mask = batch['mask'].to(self.device)
            modality_label = batch['modality_label'].to(self.device)
            
            coord_maps_full = None
            if self.config['use_coord_maps']:
                shape = full_image.shape[2:]
                if shape not in self.coord_maps_cache:
                    self.coord_maps_cache[shape] = create_coordinate_maps(shape, self.device)
                coord_maps_full = self.coord_maps_cache[shape]

            self.optimizer.zero_grad()
            
            image_downsampled = F.interpolate(full_image, size=self.config['coarse_input_size'], mode='trilinear', align_corners=False)
            mask_downsampled = F.interpolate(full_mask, size=self.config['coarse_input_size'], mode='nearest')
            coarse_logits_full = self.coarse_model(image_downsampled)
            loss_coarse = self.criterion_seg(coarse_logits_full, mask_downsampled)
            coarse_map_upsampled = F.interpolate(
                coarse_logits_full.detach(), size=full_image.shape[2:], mode='trilinear', align_corners=False
            )

            total_loss_fine_seg, total_loss_fine_cls = 0.0, 0.0
            for _ in range(self.config['patches_per_volume']):
                img_patch, mask_patch, coarse_map_patch, coord_patch = sample_random_patch(
                    full_image, full_mask, coarse_map_upsampled, coord_maps_full, self.config['fine_patch_size']
                )
                fine_input_list = [img_patch, coarse_map_patch]
                if self.config['use_coord_maps']:
                    fine_input_list.append(coord_patch)
                fine_input = torch.cat(fine_input_list, dim=1)
                
                seg_preds_patch, cls_preds_patch = self.fine_model(fine_input)
                total_loss_fine_seg += self.criterion_seg(seg_preds_patch, mask_patch)
                total_loss_fine_cls += self.criterion_cls(cls_preds_patch, modality_label)

            avg_loss_fine_seg = total_loss_fine_seg / self.config['patches_per_volume']
            avg_loss_fine_cls = total_loss_fine_cls / self.config['patches_per_volume']
            
            total_loss = loss_coarse + avg_loss_fine_seg + 0.2 * avg_loss_fine_cls
            total_loss, lb_loss = self.moe_manager.compute_total_loss(total_loss)
            total_loss.backward()
            self.optimizer.step()
            if self.config.get('use_wandb'):
                wandb.log({
                    'train_loss_total': total_loss.item(),
                    'train_loss_coarse': loss_coarse.item(),
                    'load_balance_loss': lb_loss.item() if lb_loss else 0,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            progress_bar.set_postfix({
                'L_total': f"{total_loss.item():.4f}",
                'L_coarse': f"{loss_coarse.item():.4f}",
                'L_fine_seg': f"{avg_loss_fine_seg.item():.4f}"
            })

    def _compute_dice(self, preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        return (2. * intersection + smooth) / (union + smooth)

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        self.coarse_model.eval()
        self.fine_model.eval()
        total_val_dice, total_val_loss_seg, total_val_loss_cls = 0.0, 0.0, 0.0
        progress_bar = tqdm(self.val_loader, desc=f"Валидация Эпоха {self.current_epoch}")
        
        for batch in progress_bar:
            full_image = batch['image'].to(self.device)
            full_mask = batch['mask'].to(self.device)
            modality_label = batch['modality_label'].to(self.device)

            seg_logits, cls_logits = self.inference_model(full_image)
            total_val_loss_seg += self.criterion_seg(seg_logits, full_mask).item()
            total_val_loss_cls += self.criterion_cls(cls_logits, modality_label).item()
            total_val_dice += self._compute_dice(seg_logits, full_mask).item()
        
        num_batches = len(self.val_loader)
        return {
            'dice': total_val_dice / num_batches,
            'loss_seg': total_val_loss_seg / num_batches,
            'loss_cls': total_val_loss_cls / num_batches
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'coarse_model_state_dict': self.coarse_model.state_dict(),
            'fine_model_state_dict': self.fine_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        filename = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(state, filename)
        if is_best:
            best_filename = self.save_dir / "best_model.pth"
            torch.save(state, best_filename)
            print(f"🏆 Обновлен лучший чекпоинт: {best_filename} (Dice: {self.best_val_dice:.4f})")

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            print(f"\n{'='*25} Эпоха {epoch}/{num_epochs} {'='*25}")
            
            self.train_epoch()
            val_metrics = self.validate_epoch()
            current_dice = val_metrics['dice']
            
            print(f"\n📊 Результаты валидации: "
                  f"Dice = {current_dice:.4f}, "
                  f"Seg Loss = {val_metrics['loss_seg']:.4f}, "
                  f"Cls Loss = {val_metrics['loss_cls']:.4f}")
            
            self.scheduler.step(current_dice)
            
            is_best = current_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = current_dice
            
            if (epoch % self.config.get('save_freq', 5) == 0) or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

        print(f"\nОбучение завершено! Лучший Dice score на валидации: {self.best_val_dice:.4f}")



def main_advanced_training():
    """Главная функция для конфигурации и запуска AdvancedTrainer."""
    
    
    config = {
        'h5_path': r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_prepared_golden.h5",
        'save_dir': "./mask_220825/checkpoints",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        
        'coarse_input_size': (128, 128, 128),
        'fine_patch_size': (64, 64, 64),
        'fine_patch_overlap': (32, 32, 32),
        'use_coord_maps': True,
        'patches_per_volume': 8,
        
        'num_epochs': 150,
        'learning_rate': 1e-2,
        'validation_split': 0.5,

        'num_workers': 4,
        'save_freq': 5,
        
        'use_wandb': False,
        'wandb_project': 'med_segmentation',
        # Пути к сэйвам

        # "model_checkpoint_path":r"C:\Users\pniki\Documents\Programs\ML\Исследования\MEd\mask_220825\checkpoints\best_model.pth",
        "model_checkpoint_path":"",
        "load_optim":True
    }

    print("--- Запуск продвинутого обучения ---")
    print("Конфигурация:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("---------------------------------")
    
    trainer = AdvancedTrainer(config=config)
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main_advanced_training()