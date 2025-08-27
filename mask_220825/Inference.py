import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


try:
    from models import CoarseUNet_Medium, MultiTask_FineUNet_MoE
except ImportError as e:
    print(f"Ошибка импорта: {e}. Убедитесь, что файл models.py находится рядом.")
    exit()



def create_coordinate_maps(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    d, h, w = shape
    d_coords = torch.linspace(-1, 1, d, device=device)
    h_coords = torch.linspace(-1, 1, h, device=device)
    w_coords = torch.linspace(-1, 1, w, device=device)
    
    d_map = d_coords.view(d, 1, 1).expand(d, h, w)
    h_map = h_coords.view(1, h, 1).expand(d, h, w)
    w_map = w_coords.view(1, 1, w).expand(d, h, w)
    
    coord_maps = torch.stack([d_map, h_map, w_map], dim=0).unsqueeze(0)
    return coord_maps

def pad_or_crop_to_shape(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
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

# --- Основной класс ---

class UnifiedPatchedModel(nn.Module):
    def __init__(
        self,
        coarse_model: CoarseUNet_Medium,
        fine_model: MultiTask_FineUNet_MoE,
        coarse_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        patch_overlap: Tuple[int, int, int],
        use_coord_maps: bool = False
    ) -> None:
        super().__init__()
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coarse_size = coarse_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.use_coord_maps = use_coord_maps
        
        self.stride = [s - o for s, o in zip(self.patch_size, self.patch_overlap)]
        if any(st <= 0 for st in self.stride):
            raise ValueError(f"Stride должен быть положительным! Убедитесь, что overlap меньше patch_size.")

        self.coarse_model.eval()
        self.fine_model.eval()

    @torch.no_grad()
    def forward(self, x_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_full.device
        *_, D, H, W = x_full.shape
        pd, ph, pw = self.patch_size
        
        # --- ЭТАП 1: Coarse модель ---
        x_full_downsampled = F.interpolate(x_full, size=self.coarse_size, mode='trilinear', align_corners=False)
        full_coarse_logits = self.coarse_model(x_full_downsampled)
        full_coarse_map_upsampled = F.interpolate(
            full_coarse_logits, size=x_full.shape[2:], mode='trilinear', align_corners=False
        )
        

        coord_maps_full = None
        if self.use_coord_maps:
            coord_maps_full = create_coordinate_maps((D, H, W), device)

        # --- ЭТАП 2: Fine модель (Адаптивная логика) ---
        is_large_enough_for_sliding = (D > pd) or (H > ph) or (W > pw)
        
        if is_large_enough_for_sliding:

            sd, sh, sw = self.stride
            pad_d = (sd - (D - pd) % sd) % sd if D > pd else 0
            pad_h = (sh - (H - ph) % sh) % sh if H > ph else 0
            pad_w = (sw - (W - pw) % sw) % sw if W > pw else 0
            
            padded_x = F.pad(x_full, (0, pad_w, 0, pad_h, 0, pad_d))
            padded_coarse_map = F.pad(full_coarse_map_upsampled, (0, pad_w, 0, pad_h, 0, pad_d))
            padded_coord_maps = F.pad(coord_maps_full, (0, pad_w, 0, pad_h, 0, pad_d)) if self.use_coord_maps else None

            *_, D_pad, H_pad, W_pad = padded_x.shape
            padded_canvas = torch.zeros((x_full.shape[0], 1, D_pad, H_pad, W_pad), device=device)
            padded_norm_map = torch.zeros_like(padded_canvas)
            list_cls_logits = []

            for d in range(0, D_pad - pd + 1, sd):
                for h in range(0, H_pad - ph + 1, sh):
                    for w in range(0, W_pad - pw + 1, sw):
                        patch_x = padded_x[:, :, d:d+pd, h:h+ph, w:w+pw]
                        patch_coarse = padded_coarse_map[:, :, d:d+pd, h:h+ph, w:w+pw]

                        fine_input_list = [patch_x, patch_coarse]
                        if self.use_coord_maps:
                            patch_coord = padded_coord_maps[:, :, d:d+pd, h:h+ph, w:w+pw]
                            fine_input_list.append(patch_coord)
                        fine_input_patch = torch.cat(fine_input_list, dim=1)

                        
                        seg_logits_patch, cls_logits_patch = self.fine_model(fine_input_patch)
                        
                        padded_canvas[:, :, d:d+pd, h:h+ph, w:w+pw] += seg_logits_patch
                        padded_norm_map[:, :, d:d+pd, h:h+ph, w:w+pw] += 1
                        list_cls_logits.append(cls_logits_patch)
            

            if not list_cls_logits:
                raise RuntimeError("Критическая ошибка: цикл инференса не выполнился.")
            padded_norm_map[padded_norm_map == 0] = 1
            final_seg_logits = (padded_canvas / padded_norm_map)[:, :, :D, :H, :W]
            final_cls_logits = torch.stack(list_cls_logits).mean(dim=0)

        else:
            # --- СЛУЧАЙ 2: Изображение маленькое ---

            padded_x = torch.zeros((x_full.shape[0], 1, pd, ph, pw), device=device)
            padded_coarse_map = torch.zeros_like(padded_x)
            padded_x[0, 0] = torch.from_numpy(pad_or_crop_to_shape(x_full.cpu().numpy()[0, 0], self.patch_size)).to(device)
            padded_coarse_map[0, 0] = torch.from_numpy(pad_or_crop_to_shape(full_coarse_map_upsampled.cpu().numpy()[0, 0], self.patch_size)).to(device)


            fine_input_list = [padded_x, padded_coarse_map]
            if self.use_coord_maps:

                padded_coord_maps = create_coordinate_maps(self.patch_size, device)
                fine_input_list.append(padded_coord_maps)
            fine_input = torch.cat(fine_input_list, dim=1)
            # ----------------------------------------------

            seg_logits_padded, final_cls_logits = self.fine_model(fine_input)

            seg_logits_cropped_np = pad_or_crop_to_shape(seg_logits_padded.cpu().numpy()[0, 0], (D, H, W))
            final_seg_logits = torch.from_numpy(seg_logits_cropped_np).to(device).unsqueeze(0).unsqueeze(0)

        return final_seg_logits, final_cls_logits