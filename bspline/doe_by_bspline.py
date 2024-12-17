"""
Модуль представляет обработчик тензорных данных, как коэффициентов B-сплайна,
для расчёта фазовой функции ДОЭ.
"""

import torch as _torch
from bspline.bspline_v2 import bspline_v2 as _bs
from source.config import Config as _Config
from collections.abc import Callable as _Callable


class DOE_by_bspline:
    """
    Класс оператора обработки значений тензора, как коэффициентов B-сплайна.

    Properties:
        knotes_count_by_x: число узлов по горизонтали,
        knotes_count_by_y: число узлов по вертикали,
        degree: степень B-сплайнов.
    """
    def __init__(self, config: _Config,
                 knotes_count_by_x: int,
                 knotes_count_by_y: int,
                 degree: int,
                 function: _Callable[[_torch.Tensor, _Config, _torch.Tensor],
                                     _torch.Tensor] | None,
                 koef: float = 1.0):
        """
        Конструктор класса.

        Args:
            config: конфигурация расчётной системы,
            knotes_count_by_x: число узлов по горизонтали,
            knotes_count_by_y: число узлов по вертикали,
            degree: степень B-сплайнов,
            function: функция обработки коэффициентов B-сплайна,
            koef: коэффициент размера результата (1.0 - данные без изменений).
        """
        self.__config = config
        self.__nc_x = knotes_count_by_x
        self.__nc_y = knotes_count_by_y
        self.__degree = degree
        self.__function = function
        self.__koef = koef
        self.__set_knotes_by_x()
        self.__set_knotes_by_y()
        self.__calculate_bspline()

    @property
    def knotes_count_by_x(self) -> int:
        """
        Возвращает число узлов по горизонтали.
        """
        return self.__nc_x

    @knotes_count_by_x.setter
    def knotes_count_by_x(self, data: int):
        """
        Устанавливает число узлов по горизонтали.

        Args:
            data: новое узлов по горизонтали.
        """
        self.__nc_x = data
        self.__bspline_x = self.__set_knotes_by_x()
        self.__calculate_bspline_by_dim(self.__nc_x, self.__config.X[0])

    @property
    def knotes_count_by_y(self) -> int:
        """
        Возвращает число узлов по вертикали.
        """
        return self.__knotes_count_by_y

    @knotes_count_by_y.setter
    def knotes_count_by_y(self, data: int):
        """
        Устанавливает число узлов по вертикали.

        Args:
            data: новое узлов по вертикали.
        """
        self.__knotes_count_by_y = data
        self.__bspline_y = self.__set_knotes_by_y()
        self.__calculate_bspline_by_dim(self.__nc_y, self.__config.Y[:, 0])

    @property
    def degree(self) -> int:
        """
        Возвращает степень B-сплайнов.
        """
        return self.__degree

    @degree.setter
    def degree(self, data: int):
        """
        Устанавливает степень B-сплайнов.

        Args:
            data: новая степень B-сплайнов.
        """
        self.__degree = data
        self.__set_knotes_by_x()
        self.__set_knotes_by_y()
        self.__calculate_bspline()

    def __set_knotes_by_x(self):
        self.revers_nc_x = round(self.__config.doe_size / self.__nc_x + 0.5)

    def __set_knotes_by_y(self):
        self.revers_nc_y = round(self.__config.doe_size / self.__nc_y + 0.5)

    def __calculate_bspline(self):
        self.__bspline_x = self.__calculate_bspline_by_dim(self.__nc_x,
                                                           self.__config.X[0])
        self.__bspline_y = self.__calculate_bspline_by_dim(self.__nc_y,
                                                           self.__config.Y[:, 0])

    def __calculate_bspline_by_dim(self, nc: int, data: _torch.Tensor) -> _torch.Tensor:
        knots = _torch.linspace(-self.__config.aperture_size / 2,
                                self.__config.aperture_size / 2,
                                nc - self.__degree + 1)
        min_knots = knots[0]
        max_knots = knots[-1]
        tx = _torch.cat((_torch.ones(self.__degree) * min_knots,
                         knots,
                         _torch.ones(self.__degree) * max_knots), dim=0)
        return _bs(data, nc, self.__degree, tx)

    def __call__(self, data: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Args:
            data: параметры поверхности ДОЭ,
            config: конфигурация расчётной системы,
            selection: номера выбранных длин волн.

        Returns:
            Распределение фазы ДОЭ.
        """
        return (self.__bspline_x.to(config.device)
                @ (self.__function(data[...,
                                        ::self.revers_nc_x,
                                        ::self.revers_nc_y].to(config.device),
                                   config,
                                   selection)
                   * self.__koef)
                @ self.__bspline_y.to(config.device).T)
