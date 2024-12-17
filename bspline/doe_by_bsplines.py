import torch as _torch
from source.config import Config as _Config
from bspline.doe_by_bspline import DOE_by_bspline as _DBB


class DOE_by_bsplines:
    """
    Класс оператора обработки значений тензора, как сумму B-сплайнов
    с пересекающимися коэфициентами.

    Properties:
        bsplines: операторы B-сплайнов.
    """
    def __init__(self, bsplines: list[_DBB]):
        """
        Конструктор класса.

        Args:
            bsplines: операторы B-сплайнов.
        """
        self.bsplines: list[_DBB] = bsplines

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
        result = _torch.zeros_like(data).to(config.device)
        for bspline in self.bsplines:
            result = result + bspline(data, config, selection)
        return result
