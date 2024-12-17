import torch as _torch
import torch.nn as _nn
from .config import Config as _Config

class ONN(_nn.Module):
    """
    Класс дифракционной нейронной сети.
 
    Поля:
        gamma: коэффициенты выборки.
        selection: список номеров используемых длин волн.
    """
    def __init__(self,
                 config: _Config,
                 propagation_sequence: _nn.Sequential):
        """
        Конструктор класса.
 
        Параметры:
            config: конфигурация расчётной системы.
            propagation_sequence: последовательность распространения.
            светового поля сквозь систему.
        """
        super(ONN, self).__init__()
        self.__config: _Config = config
        self.__propagation_sequence: _nn.Sequential = propagation_sequence
        zero_count = int((config.array_size - config.image_size) / 2)
        self.__pad = _nn.ZeroPad2d(zero_count)
        self.set_selection(_torch.arange(self.__config.wavelength.size(0)))
        self.gamma = 1

    def save_weights(self, filename: str = "wights.pth"):
        """
        Метод сохранения рассчитанных весов системы.
 
        Параметры:
            filename: название файла для сохранения данных.
        """
        _torch.save(self.state_dict(), filename)

    def save(self, filename: str = "model.pth"):
        """
        Метод полного сохранения системы.
 
        Параметры:
            filename: название файла для сохранения данных.
        """
        _torch.save(self, filename)

    def load_weights(self, filename: str = "wights.pth"):
        """
        Метод загрузки весов системы.
 
        Параметры:
            filename: название файла с сохранёнными данными.
        """
        self.load_state_dict(_torch.load(filename))

    def load(self, filename: str = "model.pth"):
        """
        Метод загрузки всей системы.
 
        Параметры:
            filename: название файла с сохранёнными данными.
        """
        self = _torch.load(filename)

    def set_selection(self, selection: _torch.Tensor):
        """
        Метод установки списка нужных выборок.
 
        Параметры:
            selection: список номеров используемых длин волн.
        """
        self.selection = selection
        for propagator in self.__propagation_sequence:
            propagator.selection = selection

    def get_distance(self) -> float:
        """
        Метод получения дистанции распространения светового поля.
        
        Returns:
            Дистанция распространения светового поля.
        """
        distance: float = 0.0
        for propagator in self.__propagation_sequence:
            distance += propagator.get_distance()
        return distance

    def prepare(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки входного изображения к подаче на вход системе.

        Параметры:
            data: входное изображение.
        
        Returns:
            Комплексная амплитуда входного сигнала.
        """
        data = _nn.functional.interpolate(data,
                                          [self.__config.image_size, self.__config.image_size],
                                          mode = "nearest") 
        # data = data / ((data**2).sum((2, 3))[:, :, None, None]**0.5) * (self.gamma**0.5)
        data = self.__pad(data)
        return data
    
    def forward(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод прямого распространения входного изображения
        через оптическую нейронную сеть.

        Параметры:
            data: входное изображение.
        
        Returns:
            Выход нейронной сети.
        """
        field = self.prepare(data.to(self.__config.device))
        field = self.__propagation_sequence(field)
        return field.abs()**2