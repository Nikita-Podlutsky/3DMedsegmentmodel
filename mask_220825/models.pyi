# File: models.pyi
from typing import Tuple
import torch
import torch.nn as nn

# =============================================================================
# --- БАЗОВЫЕ СТРОИТЕЛЬНЫЕ БЛОКИ ---
# =============================================================================

class ResBlock3D(nn.Module):
    """
    Базовый строительный блок ResNet в 3D.

    Состоит из двух сверточных слоев, батч-нормализации и skip connection.
    Позволяет строить глубокие сети, эффективно справляясь с проблемой
    затухания градиентов.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        Args:
            in_channels (int): Количество входных каналов.
            out_channels (int): Количество выходных каналов.
            stride (int, optional): Шаг (stride) для первой свертки. Defaults to 1.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет ResNet блок к входному тензору."""
        ...

class MoELayer(nn.Module):
    """
    Слой "Смесь Экспертов" (Mixture of Experts).

    Содержит N "экспертов" (в данном случае, ResBlock3D) и "диспетчера"
    (gating network). Диспетчер динамически определяет веса для каждого
    эксперта на основе входных данных. Итоговый выход является взвешенной
    суммой выходов всех экспертов.
    """
    def __init__(self, in_channels: int, out_channels: int, num_experts: int = 4) -> None:
        """
        Args:
            in_channels (int): Количество входных каналов.
            out_channels (int): Количество выходных каналов.
            num_experts (int, optional): Количество параллельных экспертов. Defaults to 4.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет взвешенный выход экспертов."""
        ...

# =============================================================================
# --- АРХИТЕКТУРЫ МОДЕЛЕЙ ---
# =============================================================================

class Classifier(nn.Module):
    """Простая 3D CNN для определения модальности входного изображения."""
    def __init__(self, in_channels: int = 1, num_classes: int = 8) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты классификации (B, num_classes)."""
        ...

class CoarseUNet_Medium(nn.Module):
    """
    Легковесная 3D U-Net для "грубой" (Coarse) сегментации.

    Используется на первом этапе Coarse-to-Fine пайплайна. Быстро создает
    приблизительную маску, которая затем используется как подсказка для
    более мощной Fine-модели.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 8) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты грубой сегментационной маски."""
        ...

class MultiTask_FineUNet_MoE(nn.Module):
    """
    Многозадачная 3D U-Net для "точной" (Fine) сегментации и классификации.

    Является ядром системы. На вход принимает конкатенацию исходного изображения
    и грубой маски. Модель имеет общий кодировщик и две "головы": одну для
    финальной сегментации и вторую для классификации модальности.
    Построена на основе MoE-слоев.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels_seg: int = 1,
        num_classes: int = 8,
        base_filters: int = 16,
        num_experts: int = 4
    ) -> None:
        """
        Args:
            in_channels (int, optional): Входные каналы (изображение + грубая маска). Defaults to 2.
            out_channels_seg (int, optional): Выходные каналы для сегментации. Defaults to 1.
            num_classes (int, optional): Количество классов для классификации модальностей. Defaults to 8.
            base_filters (int, optional): Начальное количество фильтров. Defaults to 16.
            num_experts (int, optional): Количество экспертов в каждом MoE-слое. Defaults to 4.
        """
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Выполняет прямое распространение и возвращает выходы обеих задач.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Кортеж из (seg_logits, cls_logits).
                - seg_logits: Логиты для точной сегментационной маски.
                - cls_logits: Логиты для классификации модальности.
        """
        ...

# =============================================================================
# --- МОДЕЛЬ-ОБЕРТКА ДЛЯ ИНФЕРЕНСА ---
# =============================================================================

class PatchedFineModel(nn.Module):
    """
    Модель-обертка, реализующая логику "скользящего окна" (sliding window) для инференса.

    Эта модель предназначена для применения `MultiTask_FineUNet_MoE` (обученной на
    патчах) к полноразмерным 3D-изображениям. Она автоматически нарезает входной
    объем на патчи с перекрытием, получает предсказания для каждого патча
    и "сшивает" их обратно в единую маску, усредняя результаты в местах перекрытий.
    """
    def __init__(
        self,
        num_classes: int,
        patch_size: Tuple[int, int, int],
        patch_overlap: Tuple[int, int, int]
    ) -> None:
        """
        Args:
            num_classes (int): Количество классов для внутреннего классификатора.
            patch_size (Tuple[int, int, int]): Размер патча (D, H, W).
            patch_overlap (Tuple[int, int, int]): Перекрытие между патчами (D, H, W).
        """
        ...

    def forward(
        self, x_full: torch.Tensor, x_coarse_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Применяет модель к полноразмерным 3D-тензорам.

        Args:
            x_full (torch.Tensor): Полноразмерное входное изображение.
            x_coarse_full (torch.Tensor): Полноразмерная грубая маска от Coarse-модели.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Кортеж из (final_seg_logits, final_cls_logits).
                - final_seg_logits: Финальная "сшитая" маска для всего объема.
                - final_cls_logits: Усредненные логиты классификации для всего объема.
        """
        ...