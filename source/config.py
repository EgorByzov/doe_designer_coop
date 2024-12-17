import torch as _torch
import numpy as _np
from .utils import get_refraction_index as _get_refraction_index

class Config:
    """
    Класс конфигурации хранит информацию о расчётной системе.
 
    Поля:
        array_size: размер вычислительной области в пикселях.
        pixel_size: дискретизация вычислительной области.
        doe_size: размер ДОЭ в пикселях.
        doe_pixel_size: дискретизация ДОЭ.
        doe_count: колличество ДОЭ в вычислительной системе.
        wavelength: длины волн используемых в системе.
        K: волновые числа.
        aperture_size: апертура вычислительной области.
        image_size: размер входного изображения в пикселях.
        distance: дистанция распространения светового поля между плоскостями.
        N: показатели преломления волн.
        H_max: максимальная высота микрорельефа ДОЭ.
        device: устройство, выполняющее расчёты.
        X: координаты абсциссы вычислительной области.
        Y: координаты ординаты вычислительной области.
        R: координаты радиуса вычислительной области.
    """
    def __init__(self, array_size: int = 1024,
                 pixel_size: float = 10e-6,
                 doe_size: int = 1024,
                 doe_pixel_size: float = 10e-6,
                 doe_count: int = 1,
                 image_size: int = 1024,
                 wavelength: _torch.Tensor = _torch.Tensor(_np.linspace(430e-9, 700e-9, 10)),
                 distance: float = 0.08,
                 H_max: float = 6e-6,
                 refraction_index: None | float | _torch.Tensor = None,
                 device: _torch.device = _torch.device('cpu')):
        """
        Конструктор класса.
 
        Параметры:
            array_size: размер вычислительной области в пикселях.
            pixel_size: дискретизация вычислительной области.
            doe_size: размер ДОЭ в пикселях.
            doe_pixel_size: дискретизация ДОЭ.
            doe_count: колличество ДОЭ в вычислительной системе.
            image_size: размер входного изображения в пикселях.
            wavelength: длины волн используемых в системе.
            distance: дистанция распространения светового поля между плоскостями.
            refraction_index: показатели преломления волн.
            H_max: максимальная высота микрорельефа ДОЭ.
            device: устройство, выполняющее расчёты.
        """
        self.array_size: int = array_size
        self.pixel_size: float = pixel_size
        self.doe_size: int = doe_size
        self.doe_pixel_size: float = doe_pixel_size
        self.doe_count: int = doe_count
        self.wavelength: _torch.Tensor = wavelength
        self.K = 2 * _np.pi / self.wavelength
        self.aperture_size: float = self.array_size * self.pixel_size
        self.image_size: int = image_size
        self.distance: float = distance
        self.N = refraction_index
        if self.N is None:
            self.N = _get_refraction_index(self.wavelength*1e6)
        self.H_max: float = H_max
        self.device: _torch.device = device
        x = _torch.linspace(-self.aperture_size / 2,
                            self.aperture_size / 2,
                            self.array_size + 1)[:self.array_size]
        x = x + self.pixel_size/2
        self.Y, self.X = _torch.meshgrid(x, x, indexing='ij')
        self.R = (self.X**2 + self.Y**2)**0.5

    def save(self, filename: str = "config.pth"):
        """
        Метод сохранения параметров конфигурации в файл.
 
        Параметры:
            filename: название файла с параметрами конфигурации.
        """
        _torch.save(self, filename)

    def load(self, filename: str = "config.pth"):
        """
        Метод загрузки параметров конфигурации из файла.
 
        Параметры:
            filename: название файла с параметрами конфигурации.
        """
        self = _torch.load(filename)