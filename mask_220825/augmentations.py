# augmentations.py
import torchio as tio

def get_augmentations_transform():
    """Создает конвейер мощных 3D-аугментаций."""
    # Эти аугментации будут применяться случайным образом к каждому сэмплу
    augmentations = {
        tio.RandomFlip(axes=('LR', 'AP', 'IS')), # Отражение по осям
        tio.RandomAffine(
            scales=(0.9, 1.2),      # Масштабирование
            degrees=15,             # Поворот
            translation=10,         # Сдвиг
            default_pad_value=0
        ),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5), # Эластичная деформация
        tio.RandomNoise(std=0.1),               # Гауссов шум
        tio.RandomBlur(std=(0, 1)),             # Размытие
        tio.RandomGamma(log_gamma=(-0.3, 0.3))  # Изменение контрастности
    }
    
    # Собираем все в один конвейер. OneOf применяет только одну из группы.
    # Это позволяет избежать слишком сильных искажений.
    transform = tio.Compose([
        tio.OneOf(augmentations, p=0.8), # Применяем одну из аугментаций с вероятностью 80%
    ])
    
    return transform