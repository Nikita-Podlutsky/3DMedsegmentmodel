import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """
    Классический остаточный блок (Residual Block) для 3D сверточных сетей.
    Обеспечивает стабильное прохождение градиента по глубоким веткам.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class MoEBlock3D(nn.Module):
    """
    Блок смеси экспертов (Mixture of Experts) для 3D данных.
    Использует внешние логиты классов для динамического взвешивания выходов экспертов.
    """
    def __init__(self, in_channels, out_channels, num_experts=4, num_classes=8):
        super().__init__()
        self.num_experts = num_experts
        
        # Набор независимых экспертов на базе 3D остаточных блоков
        self.experts = nn.ModuleList(
            [ResidualBlock3D(in_channels, out_channels) for _ in range(num_experts)]
        )
        
        # Адаптер, преобразующий логиты классификатора в веса активации экспертов
        self.route_adapter = nn.Sequential(
            nn.Linear(num_classes, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x, class_logits):
        # class_logits передаются из глобального классификатора CoarseUNet
        weights = self.route_adapter(class_logits)  # (Batch, num_experts)

        # Прогон через всех экспертов
        expert_outputs = [expert(x) for expert in self.experts]
        
        # Сборка и изменение размерности для поэлементного взвешивания
        stacked_outputs = torch.stack(expert_outputs, dim=0)  # (num_experts, Batch, C, D, H, W)
        stacked_outputs = stacked_outputs.permute(1, 0, 2, 3, 4, 5)  # (Batch, num_experts, C, D, H, W)

        # Расширяем веса для совместимости с формой тензора выходов
        weights = weights.view(-1, self.num_experts, 1, 1, 1, 1)
        mixed_output = (stacked_outputs * weights).sum(dim=1)
        
        return mixed_output


class CoarseUNet(nn.Module):
    """
    Грубая (Coarse) модель. Принимает сжатый по пространству 3D снимок целиком.
    Выполняет две задачи одновременно:
      1. Строит грубую карту сегментации.
      2. Глобально классифицирует модальность всего снимка в бутылочном горлышке (bottleneck).
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=8, num_classes=8):
        super().__init__()

        def conv_block(ic, oc):
            return nn.Sequential(
                nn.Conv3d(ic, oc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(oc),
                nn.ReLU(inplace=True),
                nn.Conv3d(oc, oc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(oc),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(base_filters, base_filters * 2)

        # Глобальный классификатор модальности
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_filters * 2, num_classes)
        )

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_filters * 2, base_filters)

        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        enc1_out = self.enc1(x)
        bottleneck_in = self.pool1(enc1_out)
        
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        # Предсказываем модальность один раз на глобальном уровне
        cls_logits = self.classification_head(bottleneck_out)
        
        dec1_in = self.up1(bottleneck_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        coarse_seg_logits = self.final_conv(dec1_out)
        
        return coarse_seg_logits, cls_logits


class FineUNet(nn.Module):
    """
    Точная (Fine) модель. Работает на уровне локальных 3D-патчей.
    Принимает склеенные каналы снимка и грубой маски, а также логиты из CoarseUNet 
    для управления выбором экспертов в MoE-блоках.
    """
    def __init__(self, in_channels=2, out_channels_seg=1, num_classes=8, base_filters=16, num_experts=4):
        super().__init__()

        self.enc1 = MoEBlock3D(in_channels, base_filters, num_experts, num_classes)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = MoEBlock3D(base_filters, base_filters * 2, num_experts, num_classes)
        self.pool2 = nn.MaxPool3d(2)
        
        self.bottleneck = MoEBlock3D(base_filters * 2, base_filters * 4, num_experts, num_classes)

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = MoEBlock3D(base_filters * 4, base_filters * 2, num_experts, num_classes) 
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = MoEBlock3D(base_filters * 2, base_filters, num_experts, num_classes)
        
        self.final_conv = nn.Conv3d(base_filters, out_channels_seg, kernel_size=1)

    def forward(self, x, cls_logits):
        enc1_out = self.enc1(x, cls_logits)
        enc2_out = self.enc2(self.pool1(enc1_out), cls_logits)
        bottleneck_out = self.bottleneck(self.pool2(enc2_out), cls_logits)

        dec2_in = self.up2(bottleneck_out)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in, cls_logits)
        
        dec1_in = self.up1(dec2_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in, cls_logits)
        
        seg_logits = self.final_conv(dec1_out)
        
        return seg_logits


class PatchedFineModel(nn.Module):
    """
    Обертка над FineUNet для инференса методом скользящего окна (патчинга).
    """
    def __init__(self, num_classes, patch_size, patch_overlap, base_filters=16, num_experts=4):
        super().__init__()
        self.fine_model = FineUNet(
            in_channels=2,
            out_channels_seg=1,
            num_classes=num_classes,
            base_filters=base_filters,
            num_experts=num_experts
        )

        assert all(s > 0 for s in patch_size), "Размеры патча должны быть > 0"
        assert all(0 <= o < s for o, s in zip(patch_overlap, patch_size)), "Перекрытие должно быть меньше размера патча"
        
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.stride = [s - o for s, o in zip(patch_size, patch_overlap)]

    @torch.no_grad()
    def forward(self, x_full, x_coarse_full, cls_logits):
        fine_input_full = torch.cat([x_full, x_coarse_full], dim=1)

        *_, D, H, W = fine_input_full.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        pad_d = (sd - (D - pd) % sd) % sd
        pad_h = (sh - (H - ph) % sh) % sh
        pad_w = (sw - (W - pw) % sw) % sw

        padded_input = F.pad(fine_input_full, (0, pad_w, 0, pad_h, 0, pad_d))
        *_, D_pad, H_pad, W_pad = padded_input.shape

        padded_canvas = torch.zeros((x_full.shape[0], 1, D_pad, H_pad, W_pad), device=x_full.device)
        padded_norm_map = torch.zeros_like(padded_canvas)
        
        for d in range(0, D_pad - pd + 1, sd):
            for h in range(0, H_pad - ph + 1, sh):
                for w in range(0, W_pad - pw + 1, sw):
                    patch = padded_input[:, :, d:d+pd, h:h+ph, w:w+pw]
                    
                    # Прогоняем каждый патч, используя глобальные cls_logits
                    seg_logits_patch = self.fine_model(patch, cls_logits)
                    
                    padded_canvas[:, :, d:d+pd, h:h+ph, w:w+pw] += seg_logits_patch
                    padded_norm_map[:, :, d:d+pd, h:h+ph, w:w+pw] += 1
        
        final_seg_logits = (padded_canvas / padded_norm_map)[:, :, :D, :H, :W]
        
        return final_seg_logits