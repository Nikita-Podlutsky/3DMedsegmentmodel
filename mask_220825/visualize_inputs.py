import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path
from typing import Tuple, Dict, Any

# --- Импорты из вашего проекта ---
try:
    from data_units import FullImageDataset
    from models import CoarseUNet_Medium, MultiTask_FineUNet_MoE
    from inference import UnifiedPatchedModel
except ImportError as e:
    print(f"Ошибка импорта: {e}\nУбедитесь, что все необходимые файлы (.py) находятся в той же директории.")
    exit()


def inspect_full_prediction_with_threshold(config: Dict[str, Any]):
    device = torch.device(config['device'])
    checkpoint_path = Path(config['checkpoint_path'])
    if not checkpoint_path.exists():
        print(f"Ошибка: Файл чекпоинта не найден: {checkpoint_path}")
        return


    print(f"Загрузка чекпоинта из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint['config']
    
    coarse_model = CoarseUNet_Medium(base_filters=16).to(device)
    fine_in_channels = 2 + (3 if train_config.get('use_coord_maps', False) else 0)
    fine_model = MultiTask_FineUNet_MoE(in_channels=fine_in_channels, num_classes=8).to(device)

    inference_model = UnifiedPatchedModel(
        coarse_model=coarse_model,
        fine_model=fine_model,
        coarse_size=train_config['coarse_input_size'],
        patch_size=train_config['fine_patch_size'],
        patch_overlap=train_config['fine_patch_overlap'],
        use_coord_maps=train_config.get('use_coord_maps', False)
    )
    
    inference_model.coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
    inference_model.fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
    inference_model.eval()
    print("Модели успешно загружены.")

    dataset = FullImageDataset(config['h5_path'])
    sample = dataset[config['sample_idx']]
    full_image = sample['image'].unsqueeze(0).to(device)
    full_mask_np = sample['mask'][0].cpu().numpy()
    full_image_np = sample['image'][0].cpu().numpy()
    print(f"Загружен сэмпл #{config['sample_idx']} (модальность: {sample['modality_name']})")

    print("Выполнение предсказания на полном 3D-изображении...")
    with torch.no_grad():
        seg_logits, _ = inference_model(full_image)

    predicted_probs_np = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    print("Предсказание завершено.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    plt.subplots_adjust(left=0.1, bottom=0.25) 
    fig.suptitle(f"Сравнение Результатов (Сэмпл #{config['sample_idx']}, Модальность: {sample['modality_name']})", fontsize=16)

    images = {}
    current_axis = 'Аксиальный (D)'
    axis_map = {'Аксиальный (D)': 0, 'Сагиттальный (H)': 1, 'Корональный (W)': 2}
    
    # --- Функция обновления ---
    def update_plots(val):
        
        slice_idx = int(slice_slider.val)
        threshold = threshold_slider.val
        
        axis_idx = axis_map[current_axis]
        
        pred_mask_binary = (predicted_probs_np > threshold).astype(np.uint8)

        if axis_idx == 0:
            img_slice = full_image_np[slice_idx, :, :]
            mask_slice = full_mask_np[slice_idx, :, :]
            pred_slice = pred_mask_binary[slice_idx, :, :]
        elif axis_idx == 1:
            img_slice = full_image_np[:, slice_idx, :]
            mask_slice = full_mask_np[:, slice_idx, :]
            pred_slice = pred_mask_binary[:, slice_idx, :]
        else:
            img_slice = full_image_np[:, :, slice_idx]
            mask_slice = full_mask_np[:, :, slice_idx]
            pred_slice = pred_mask_binary[:, :, slice_idx]
            
        images[0].set_data(img_slice)
        images[1].set_data(mask_slice)
        images[2].set_data(pred_slice)
        
        axes[0].set_title(f'Исходное Изображение (срез {slice_idx})')
        axes[2].set_title(f'Предсказание Модели (Порог = {threshold:.2f})')
        
        fig.canvas.draw_idle()


    D, H, W = full_image_np.shape
    initial_slice = D // 2
    initial_threshold = 0.5
    
    initial_pred_mask = (predicted_probs_np > initial_threshold).astype(np.uint8)
    
    images[0] = axes[0].imshow(full_image_np[initial_slice, :, :], cmap='gray', aspect='auto')
    axes[0].set_title(f'Исходное Изображение (срез {initial_slice})')
    
    images[1] = axes[1].imshow(full_mask_np[initial_slice, :, :], cmap='gray', aspect='auto')
    axes[1].set_title('Правильная Маска (Эталон)')
    
    images[2] = axes[2].imshow(initial_pred_mask[initial_slice, :, :], cmap='gray', aspect='auto')
    axes[2].set_title(f'Предсказание Модели (Порог = {initial_threshold:.2f})')


    ax_slice_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slice_slider = Slider(ax_slice_slider, 'Срез', 0, D - 1, valinit=initial_slice, valstep=1)


    ax_threshold_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    threshold_slider = Slider(
        ax=ax_threshold_slider,
        label='Порог',
        valmin=0.0,
        valmax=1.0,
        valinit=initial_threshold,
        valfmt='%0.2f'
    )
    
    ax_radio = plt.axes([0.01, 0.7, 0.07, 0.2])
    radio = RadioButtons(ax_radio, ('Аксиальный (D)', 'Сагиттальный (H)', 'Корональный (W)'), active=0)

    def select_axis(label):
        nonlocal current_axis
        current_axis = label
        axis_idx = axis_map[current_axis]
        max_slices = full_image_np.shape[axis_idx]
        slice_slider.valmax = max_slices - 1
        slice_slider.set_val(max_slices // 2)

    slice_slider.on_changed(update_plots)
    threshold_slider.on_changed(update_plots)
    radio.on_clicked(select_axis)

    plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])
    plt.show()
if __name__ == '__main__':

    INSPECTION_CONFIG = {
        'h5_path': r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_prepared_golden.h5",
        'checkpoint_path': r"C:\Users\pniki\Documents\Programs\ML\Исследования\MEd\mask_220825\checkpoints\checkpoint_epoch_30.pth",
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sample_idx': 20
    }
    
    inspect_full_prediction_with_threshold(INSPECTION_CONFIG)


# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from typing import Tuple, Dict, Any, List
# from tqdm import tqdm

# # --- Импорты из вашего проекта ---
# try:
#     from data_units import FullImageDataset
#     from models import CoarseUNet_Medium, MultiTask_FineUNet_MoE
#     from inference import UnifiedPatchedModel
# except ImportError as e:
#     print(f"Ошибка импорта: {e}\nУбедитесь, что все необходимые файлы (.py) находятся в той же директории.")
#     exit()

# # --- Вспомогательные функции ---

# def dice_score_numpy(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
#     """Вычисляет Dice Score для бинарных numpy массивов."""
#     pred = pred.astype(bool)
#     target = target.astype(bool)
    
#     intersection = np.sum(pred & target)
#     union = np.sum(pred) + np.sum(target)
    
#     if union == 0: return 1.0
#     return (2. * intersection + smooth) / (union + smooth)

# def calculate_optimal_threshold(probs_np: np.ndarray, mask_np: np.ndarray, num_steps: int = 100) -> Tuple[float, float, np.ndarray, np.ndarray]:
#     """Находит оптимальный порог и возвращает данные для построения кривой."""
#     thresholds = np.linspace(0.01, 0.99, num_steps)
#     dice_scores = np.array([dice_score_numpy((probs_np > t), mask_np) for t in thresholds])
    
#     best_idx = np.argmax(dice_scores)
#     return thresholds[best_idx], dice_scores[best_idx], thresholds, dice_scores

# def plot_multiple_dice_curves(
#     thresholds: np.ndarray, 
#     all_dice_curves: List[np.ndarray], 
#     avg_best_threshold: float, 
#     avg_best_dice: float):
#     """Строит график с индивидуальными и усредненной кривыми."""
#     plt.figure(figsize=(12, 7))
    
#     # Рисуем все индивидуальные кривые полупрозрачными
#     for curve in all_dice_curves:
#         plt.plot(thresholds, curve, color='gray', alpha=0.3, lw=1)
    
#     # Считаем и рисуем усредненную кривую жирной линией
#     mean_curve = np.mean(all_dice_curves, axis=0)
#     plt.plot(thresholds, mean_curve, color='blue', lw=3, label='Средний Dice Score по выборке')
    
#     # Находим оптимум на усредненной кривой
#     best_t_on_avg = thresholds[np.argmax(mean_curve)]
#     best_d_on_avg = np.max(mean_curve)

#     plt.axvline(avg_best_threshold, color='red', linestyle='--', label=f'Средний оптим. порог = {avg_best_threshold:.3f}')
#     plt.axvline(best_t_on_avg, color='green', linestyle=':', label=f'Оптим. порог для ср. кривой = {best_t_on_avg:.3f}')

#     plt.title(f'Анализ Порога по {len(all_dice_curves)} сэмплам', fontsize=16)
#     plt.xlabel('Порог', fontsize=12)
#     plt.ylabel('Dice Score', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.ylim(bottom=max(0, np.min(mean_curve)-0.1)) # Динамический масштаб
    
#     # Аннотация с результатами
#     text_str = (f"Средний макс. Dice (по сэмплам): {avg_best_dice:.4f}\n"
#                 f"Макс. Dice на средней кривой: {best_d_on_avg:.4f}")
#     plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
#              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
#     plt.show()

# # --- Основная функция ---

# def analyze_thresholds_for_dataset(config: Dict[str, Any]):
#     device = torch.device(config['device'])
#     checkpoint_path = Path(config['checkpoint_path'])
#     if not checkpoint_path.exists():
#         print(f"Ошибка: Файл чекпоинта не найден: {checkpoint_path}")
#         return

#     # 1. Загрузка
#     print("Загрузка моделей и данных...")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     train_config = checkpoint['config']
    
#     # Инициализация моделей
#     coarse_model = CoarseUNet_Medium(base_filters=16).to(device)
#     fine_in_channels = 2 + (3 if train_config.get('use_coord_maps', False) else 0)
#     fine_model = MultiTask_FineUNet_MoE(in_channels=fine_in_channels, num_classes=8).to(device)
#     inference_model = UnifiedPatchedModel(
#         coarse_model=coarse_model, fine_model=fine_model,
#         coarse_size=train_config['coarse_input_size'],
#         patch_size=train_config['fine_patch_size'],
#         patch_overlap=train_config['fine_patch_overlap'],
#         use_coord_maps=train_config.get('use_coord_maps', False)
#     )
#     inference_model.coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
#     inference_model.fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
#     inference_model.eval()

#     # 2. Определение валидационной выборки
#     dataset = FullImageDataset(config['h5_path'])
#     val_split = train_config['validation_split']
#     indices = list(range(len(dataset)))
#     split_idx = int(np.floor(val_split * len(dataset)))
    
#     # Используем тот же seed, что и при обучении, для консистентности (если он был)
#     np.random.seed(42) 
#     np.random.shuffle(indices)
#     val_indices = indices[:split_idx]
    
#     # Выбираем n случайных сэмплов из валидационной выборки
#     num_samples_to_analyze = min(config['num_random_samples'], len(val_indices))
#     selected_indices = np.random.choice(val_indices, num_samples_to_analyze, replace=False)
#     print(f"Будет проанализировано {num_samples_to_analyze} случайных сэмплов из валидационной выборки.")

#     # 3. Цикл анализа
#     all_best_thresholds = []
#     all_best_dices = []
#     all_dice_curves = []

#     for sample_idx in tqdm(selected_indices, desc="Анализ сэмплов"):
#         sample = dataset[sample_idx]
#         full_image = sample['image'].unsqueeze(0).to(device)
#         full_mask_np = sample['mask'][0].cpu().numpy().astype(np.uint8)

#         with torch.no_grad():
#             seg_logits, _ = inference_model(full_image)
#         predicted_probs_np = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        
#         best_t, best_d, thresholds, dice_curve = calculate_optimal_threshold(predicted_probs_np, full_mask_np)
        
#         all_best_thresholds.append(best_t)
#         all_best_dices.append(best_d)
#         all_dice_curves.append(dice_curve)

#     # 4. Агрегация результатов
#     avg_best_threshold = np.mean(all_best_thresholds)
#     std_best_threshold = np.std(all_best_thresholds)
#     avg_best_dice = np.mean(all_best_dices)

#     print("\n" + "="*50)
#     print("      СТАТИСТИЧЕСКИЙ РЕЗУЛЬТАТ АНАЛИЗА ПОРОГА")
#     print("="*50)
#     print(f"Проанализировано сэмплов:            {num_samples_to_analyze}")
#     print(f"Средний оптимальный порог:           {avg_best_threshold:.4f} (± {std_best_threshold:.4f})")
#     print(f"Средний макс. достижимый Dice:     {avg_best_dice:.4f}")
#     print("="*50)
#     print("\nОтображение графика...")
    
#     # 5. Визуализация
#     plot_multiple_dice_curves(thresholds, all_dice_curves, avg_best_threshold, avg_best_dice)



# if __name__ == '__main__':
#     # --- НАСТРОЙТЕ ЭТИ ПАРАМЕТРЫ ---
#     ANALYSIS_CONFIG = {
#         'h5_path': r"C:\Users\pniki\Documents\Programs\Datasets\synthstrip_prepared_golden.h5",
#         'checkpoint_path': r"C:\Users\pniki\Documents\Programs\ML\Исследования\MEd\mask_220825\checkpoints\best_model.pth",
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'num_random_samples': 10
#     }
    
#     analyze_thresholds_for_dataset(ANALYSIS_CONFIG)





# # File: visualize_nifti_prediction.py

# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, RadioButtons
# from pathlib import Path
# from typing import Tuple

# def visualize_scan_and_mask(image_path: str, mask_path: str):
#     """
#     Создает интерактивную панель для сравнения NIfTI изображения и его маски.

#     Args:
#         image_path (str): Путь к исходному NIfTI файлу.
#         mask_path (str): Путь к предсказанному NIfTI файлу маски.
#     """
#     image_p = Path(image_path)
#     mask_p = Path(mask_path)

#     if not image_p.exists() or not mask_p.exists():
#         print("Ошибка: Один или оба файла не найдены.")
#         print(f"Проверьте пути:\n- Изображение: {image_p}\n- Маска: {mask_p}")
#         return

#     # 1. Загрузка данных
#     print("Загрузка NIfTI файлов...")
#     try:
#         image_nii = nib.load(image_p)
#         mask_nii = nib.load(mask_p)
        
#         image_data = image_nii.get_fdata()
#         mask_data = mask_nii.get_fdata()

#         # Убедимся, что маска бинарная
#         mask_data = (mask_data > 0).astype(np.uint8)
#     except Exception as e:
#         print(f"Не удалось загрузить файлы. Ошибка: {e}")
#         return
        
#     print("Данные успешно загружены.")
#     print(f"Размер изображения: {image_data.shape}")

#     # 2. Создание интерактивной панели
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#     plt.subplots_adjust(left=0.1, bottom=0.2)
#     fig.suptitle("Интерактивный просмотр предсказания", fontsize=16)

#     # Глобальные переменные для управления состоянием
#     current_axis_name = 'Аксиальный (Z)'
#     # Nibabel обычно загружает данные в порядке (X, Y, Z)
#     axis_map = {'Аксиальный (Z)': 2, 'Сагиттальный (X)': 0, 'Корональный (Y)': 1}
    
#     # --- Функция обновления, вызываемая виджетами ---
#     def update_plots(val):
#         slice_idx = int(slice_slider.val)
#         axis_idx = axis_map[current_axis_name]
        
#         # Выбираем срезы в зависимости от оси
#         if axis_idx == 2: # Аксиальный
#             img_slice = image_data[:, :, slice_idx].T
#             mask_slice = mask_data[:, :, slice_idx].T
#         elif axis_idx == 0: # Сагиттальный
#             img_slice = image_data[slice_idx, :, :].T
#             mask_slice = mask_data[slice_idx, :, :].T
#         else: # Корональный
#             img_slice = image_data[:, slice_idx, :].T
#             mask_slice = mask_data[:, slice_idx, :].T
            
#         # Создаем маску с прозрачностью, где 0 - полностью прозрачный
#         # Это нужно, чтобы фон не закрашивался
#         alpha_mask = np.where(mask_slice > 0, 0.4, 0) # 40% непрозрачности для маски

#         # Обновляем данные на графиках
#         images['image'].set_data(img_slice)
#         images['overlay_img'].set_data(img_slice)
#         images['overlay_mask'].set_data(mask_slice)
#         images['overlay_mask'].set_alpha(alpha_mask) # Обновляем прозрачность
        
#         axes[0].set_title(f'Исходное Изображение (срез {slice_idx})')
#         axes[1].set_title(f'Предсказание (наложение)')
        
#         fig.canvas.draw_idle()

#     # --- Первичная отрисовка ---
#     initial_axis_idx = axis_map[current_axis_name]
#     initial_slice_idx = image_data.shape[initial_axis_idx] // 2
    
#     # Получаем начальные срезы
#     initial_img_slice = image_data[:, :, initial_slice_idx].T
#     initial_mask_slice = mask_data[:, :, initial_slice_idx].T
#     initial_alpha_mask = np.where(initial_mask_slice > 0, 0.4, 0)
    
#     images = {}
#     axes[0].set_title(f'Исходное Изображение (срез {initial_slice_idx})')
#     images['image'] = axes[0].imshow(initial_img_slice, cmap='gray', aspect='equal')
    
#     axes[1].set_title(f'Предсказание (наложение)')
#     images['overlay_img'] = axes[1].imshow(initial_img_slice, cmap='gray', aspect='equal')
#     images['overlay_mask'] = axes[1].imshow(initial_mask_slice, cmap='viridis', alpha=initial_alpha_mask, aspect='equal')
    
#     # --- Создание виджетов ---
#     ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
#     slice_slider = Slider(
#         ax=ax_slider,
#         label='Срез',
#         valmin=0,
#         valmax=image_data.shape[initial_axis_idx] - 1,
#         valinit=initial_slice_idx,
#         valstep=1
#     )
    
#     ax_radio = plt.axes([0.02, 0.4, 0.07, 0.2])
#     radio = RadioButtons(ax_radio, ('Аксиальный (Z)', 'Корональный (Y)', 'Сагиттальный (X)'), active=0)

#     # --- Функция для смены оси ---
#     def select_axis(label):
#         nonlocal current_axis_name
#         current_axis_name = label
#         axis_idx = axis_map[current_axis_name]
        
#         max_slices = image_data.shape[axis_idx] - 1
#         slice_slider.valmax = max_slices
#         slice_slider.set_val(max_slices // 2) # Устанавливаем на центральный срез новой оси

#     # --- Подключение функций к виджетам ---
#     slice_slider.on_changed(update_plots)
#     radio.on_clicked(select_axis)

#     plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])
#     plt.show()

# if __name__ == '__main__':
#     # --- НАСТРОЙТЕ ЭТИ ДВА ПУТИ ---
    
#     INPUT_NIFTI_FILE = r"C:\Users\pniki\Documents\Programs\Datasets\SynthRad\23\1BA082\mr.nii.gz"
#     PREDICTED_MASK_FILE = r"C:\Users\pniki\Documents\Programs\ML\Исследования\MEd\mask_220825\ct.nii.gz"

#     # --- Запуск визуализатора ---
#     visualize_scan_and_mask(INPUT_NIFTI_FILE, PREDICTED_MASK_FILE)