import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

# Предполагается, что все ваши модели, конфиги и DEVICE уже определены
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
def predict_full_volume(
    full_image, 
    coarse_model, 
    fine_model, 
    config
):
    """
    Выполняет предсказание для одного полного 3D-объема с использованием
    sliding window inference для сглаживания и сборки патчей.

    Args:
        full_image (torch.Tensor): Входной 3D-объем (1, C, D, H, W) на DEVICE.
        coarse_model (nn.Module): Обученная Coarse-модель.
        fine_model (nn.Module): Обученная Fine-модель (MultiTask).
        config (dict): Словарь с конфигурацией (PATCH_SIZE, COARSE_UNIFIED_SIZE и т.д.).

    Returns:
        torch.Tensor: Финальная маска сегментации (1, 1, D, H, W).
    """
    # Переводим модели в режим инференса
    coarse_model.eval()
    fine_model.eval()
    
    # Отключаем расчет градиентов для экономии памяти и ускорения
    with torch.no_grad():
        # --- Шаг 1: Получаем "подсказку" от Coarse-модели ---
        coarse_input = F.interpolate(
            full_image, 
            size=config['COARSE_UNIFIED_SIZE'], 
            mode='trilinear', 
            align_corners=False
        )
        coarse_pred = coarse_model(coarse_input)
        
        # Растягиваем "подсказку" до полного размера
        upsampled_coarse_pred = F.interpolate(
            coarse_pred, 
            size=full_image.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Объединяем входы для Fine-модели
        fine_input_full = torch.cat([full_image, upsampled_coarse_pred], dim=1)
        
        # --- Шаг 2: Sliding Window Inference для Fine-модели ---
        patch_size = config['PATCH_SIZE']
        # Для инференса можно использовать большее перекрытие для лучшего качества
        patch_overlap = tuple(p // 2 for p in patch_size) 
        stride = [s - o for s, o in zip(patch_size, patch_overlap)]
        
        *_, D, H, W = fine_input_full.shape
        
        # Создаем холсты
        prediction_canvas = torch.zeros_like(full_image, dtype=torch.float32, device=DEVICE)
        weights_canvas = torch.zeros_like(full_image, dtype=torch.float32, device=DEVICE)
        
        # Создаем весовую маску (гауссово окно)
        # Мы создаем ее на CPU с NumPy/SciPy, а затем переносим на GPU
        gaussian_weights = np.zeros(patch_size)
        center_coords = [p // 2 for p in patch_size]
        gaussian_weights[center_coords[0], center_coords[1], center_coords[2]] = 1
        # Сигма ~ 1/8 размера патча - хороший выбор
        sigma = [p / 8 for p in patch_size]
        gaussian_weights = gaussian_filter(gaussian_weights, sigma, mode='constant', cval=0)
        # Нормализуем
        gaussian_weights /= np.max(gaussian_weights)
        gaussian_weights = torch.from_numpy(gaussian_weights).to(DEVICE).float()

        # Цикл по патчам (исправленный, с паддингом для покрытия краев)
        pad_d = (stride[0] - (D - patch_size[0]) % stride[0]) % stride[0]
        pad_h = (stride[1] - (H - patch_size[1]) % stride[1]) % stride[1]
        pad_w = (stride[2] - (W - patch_size[2]) % stride[2]) % stride[2]
        padded_input = F.pad(fine_input_full, (0, pad_w, 0, pad_h, 0, pad_d))
        
        *_, D_pad, H_pad, W_pad = padded_input.shape

        for d in range(0, D_pad - patch_size[0] + 1, stride[0]):
            for h in range(0, H_pad - patch_size[1] + 1, stride[1]):
                for w in range(0, W_pad - patch_size[2] + 1, stride[2]):
                    patch = padded_input[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]
                    
                    # Получаем предсказание для патча
                    seg_logits_patch, _ = fine_model(patch)
                    
                    # Применяем sigmoid, чтобы получить вероятности
                    seg_probs_patch = torch.sigmoid(seg_logits_patch)
                    
                    # Добавляем взвешенное предсказание на холст
                    prediction_canvas[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += seg_probs_patch * gaussian_weights
                    weights_canvas[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += gaussian_weights
        
        # Нормализуем и обрезаем до исходного размера
        final_prediction_probs = (prediction_canvas / (weights_canvas + 1e-8))[:, :, :D, :H, :W]
        
        # Применяем порог для получения бинарной маски
        final_mask = (final_prediction_probs > 0.5).to(torch.uint8)
        
        return final_mask

# --- Пример использования после тренировки ---
if __name__ == '__main__':
    # 1. Загрузите ваши обученные модели
    # coarse_model.load_state_dict(torch.load("coarse_model_epoch_100.pth"))
    # fine_model.load_state_dict(torch.load("fine_model_epoch_100.pth"))
    
    # 2. Возьмите один сэмпл для предсказания из вашего датасета
    # (здесь нужна логика загрузки одного скана, например, из H5)
    # dataset = ModalityAwareH5Dataset(...)
    # sample = dataset[0]
    # full_scan = sample['image'].unsqueeze(0).to(DEVICE) # Добавляем batch измерение
    
    # 3. Соберите словарь с конфигурацией
    # config = {
    #     'PATCH_SIZE': PATCH_SIZE,
    #     'COARSE_UNIFIED_SIZE': COARSE_UNIFIED_SIZE
    # }
    
    # 4. Вызовите функцию предсказания
    # final_mask = predict_full_volume(full_scan, coarse_model, fine_model, config)
    
    # print(f"Предсказание завершено. Форма финальной маски: {final_mask.shape}")
    # Теперь `final_mask` можно сохранить в файл NIfTI или визуализировать.
    pass