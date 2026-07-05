import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Основной класс ---

class UnifiedPatchedModel(nn.Module):
    """
    Инференс-модель, которая объединяет CoarseUNet и FineUNet.
    Выполняет однократную глобальную классификацию на сжатом скане, 
    строит грубую маску, нарезает данные на патчи, сегментирует их в FineUNet 
    с учетом предсказанных логитов класса и склеивает обратно.
    """
    def __init__(self, coarse_model, fine_model, coarse_size, patch_size, patch_overlap, use_coord_maps=False):
        super().__init__()
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coarse_size = coarse_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.stride = [s - o for s, o in zip(patch_size, patch_overlap)]
        self.use_coord_maps = use_coord_maps

    @torch.no_grad()
    def forward(self, x_full):
        # 1. Сжатие и глобальный прогон через CoarseUNet
        x_downsampled = F.interpolate(x_full, size=self.coarse_size, mode='trilinear', align_corners=False)
        coarse_logits_full, cls_logits = self.coarse_model(x_downsampled)
        
        # 2. Апсемплинг грубой маски до оригинального разрешения
        coarse_map_upsampled = F.interpolate(
            coarse_logits_full, size=x_full.shape[2:], mode='trilinear', align_corners=False
        )
        
        # 3. Подготовка входа для Fine-модели
        fine_input_list = [x_full, coarse_map_upsampled]
        if self.use_coord_maps:
            # Динамически импортируем функцию, чтобы не было циклического импорта
            from training_pipeline import create_coordinate_maps
            coord_maps = create_coordinate_maps(x_full.shape[2:], x_full.device)
            fine_input_list.append(coord_maps)
            
        fine_input_full = torch.cat(fine_input_list, dim=1)
        
        # 4. Паддинг для скользящего окна
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
        
        # 5. Сборка патчей скользящим окном с передачей cls_logits
        for d in range(0, D_pad - pd + 1, sd):
            for h in range(0, H_pad - ph + 1, sh):
                for w in range(0, W_pad - pw + 1, sw):
                    patch = padded_input[:, :, d:d+pd, h:h+ph, w:w+pw]
                    
                    # Прогоняем локальный патч через FineUNet, передавая глобальные логиты класса
                    seg_logits_patch = self.fine_model(patch, cls_logits)
                    
                    padded_canvas[:, :, d:d+pd, h:h+ph, w:w+pw] += seg_logits_patch
                    padded_norm_map[:, :, d:d+pd, h:h+ph, w:w+pw] += 1
                    
        # 6. Усреднение зон перекрытий и обрезка паддинга
        final_seg_logits = (padded_canvas / padded_norm_map)[:, :, :D, :H, :W]
        
        # Возвращаем склеенную точную сегментацию и глобальные логиты модальности
        return final_seg_logits, cls_logits