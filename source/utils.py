import torch as _torch
import numpy as _np
from scipy import io as _io
from PIL import Image as _Image


def get_refraction_index(wavelength: float | _torch.Tensor | _np.ndarray) -> float | _torch.Tensor | _np.ndarray:
    """
    Функция получения показателя преломления силикатного стекла согласно длине волны.
 
    Параметры:
        wavelength: длина волны в нм.
 
    Returns:
        Показатель преломления.
    """
    i1 = (0.6961663 * wavelength ** 2) / (wavelength ** 2 - 0.0684043 ** 2)
    i2 = (0.4079426 * wavelength ** 2) / (wavelength ** 2 - 0.1162414 ** 2)
    i3 = (0.8974794 * wavelength ** 2) / (wavelength ** 2 - 9.896161 ** 2)
    n = (i1 + i2 + i3 + 1) ** 0.5
    return n


def save_image(filename: str, data: _np.ndarray, mode: str = 'L'):
    """
    Функция сохранения массива в качестве изображения в файл.

    Параметры:
        filename: название файла для сохранения изображения.
        data: массив данных для сохранения в виде изображения.
        mode: режим сохранения "RGB" - цветное изображение,
            "P",
            "L" - 8бит изображение,
            "1" - 1бит изображение.
    """
    im = _Image.fromarray(data)
    im.convert(mode = mode).save(filename)


def load_image(filename: str) -> _np.ndarray:
    """
    Функция получения массива данных из файла изображения.
 
    Параметры:
        filename: название файла для загрузки изображения.
 
    Returns:
        Массив данных изображения.
    """
    image = _Image.open(fp = filename)
    return _np.asarray(image)


def save_mat(filename: str, data: dict[str, _np.ndarray]):
    """
    Функция сохранения словаря данных в виде mat файла.

    Параметры:
        filename: название файла для сохранения данных.
        data: словарь данных.
    """
    _io.savemat(file_name = filename,
               mdict = data)


def load_mat(filename: str) -> dict[str, _np.ndarray]:
    """
    Функция получения словаря данных из mat файла.
 
    Параметры:
        filename: название файла для загрузки данных.
 
    Returns:
        Словарь данных.
    """
    return _io.loadmat(file_name = filename)


def save_tensor(filename: str, data: _torch.Tensor):
    """
    Функция сохранения тензора данных в файл.

    Параметры:
        filename: название файла для сохранения данных.
        data: тензор данных.
    """
    _torch.save(data.cpu().detach(), filename)


def load_tensor(filename: str) -> _torch.Tensor:
    """
    Функция получения тензора данных из файла.
 
    Параметры:
        filename: название файла для загрузки данных.
 
    Returns:
        Тезор данных.
    """
    return _torch.load(filename)


def get_phase_of_ideal_lens(K: float | _torch.Tensor,
                            distance: float,
                            X: _torch.Tensor,
                            Y: _torch.Tensor,
                            offset_X: _torch.Tensor,
                            offset_Y: _torch.Tensor) -> _torch.Tensor:
    """
    Функция получения фазовой функции идеальной линзы.
 
    Параметры:
        K: волновое число.
        distance: дистанция фокусировки идеальной линзы.
        X: координаты абсциссы вычислительной области.
        Y: координаты ординаты вычислительной области.
        offset_X: координаты смещения по абсцисе.
        offset_Y: координаты смещения по ординате.
 
    Returns:
        Фазовая функция идеальной линзы.
    """
    first_part = (-K / 2 / distance) * (X**2 + Y**2)
    second_part = (K / distance) * (X * offset_X + Y * offset_Y)
    return _torch.exp(1j * (first_part + second_part))


def get_masks_main_petal_of_ideal_lens(wavelength: float | _torch.Tensor,
                                       distance: float,
                                       aperture_size: float,
                                       R: _torch.Tensor | _np.ndarray) -> _torch.Tensor | _np.ndarray:
    """
    Функция получения области нахождения главного лепестка фокусировки
    идельной линзы.
 
    Параметры:
        wavelength: длина волны.
        distance: дистанция фокусировки идеальной линзы.
        aperture_size: размер апертуры вычислительной области.
        R: координаты радиуса вычислительной области.
 
    Returns:
        Тензор области главного лепестка фокусировки идельной линзы.
    """
    coef = wavelength * distance / aperture_size
    return R < coef


def get_max_intensity_of_ideal_lens(R: float,
                                    wavelength: float | _torch.Tensor | _np.ndarray,
                                    distance: float) -> float | _torch.Tensor | _np.ndarray:
    """
    Функция получения пикового значения фокусировки идельной линзы.
 
    Параметры:
        R: радиус входного пучка.
        wavelength: длина волны.
        distance: дистанция фокусировки идеальной линзы.
 
    Returns:
        Значение максимума интенсивности в выходной плоскости.
    """
    area = _np.pi * R**2
    return (area / wavelength / distance)**2


def init_batch_generator(dataloader: _torch.utils.data.DataLoader):
    """
    Функция создания генератора данных.

    Параметры:
        dataloader: загрузчик данных.

    Returns:
        Функция получения новой порции данных.
    """
    def f():
        while True:
            for i, (images, labels) in enumerate(dataloader):
                yield images, labels
    return f()