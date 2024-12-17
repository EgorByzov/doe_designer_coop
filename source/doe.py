import torch as _torch
import torch.nn as _nn
from torch.nn.functional import interpolate as _interpolate
from .config import Config as _Config


class DOE(_nn.Module):
    """
    Класс дифракционного оптического элемента (ДОЭ).

    Поля:
        surface: параметры ДОЭ.
        weights: ссылка на поле surface.
        selection: список номеров используемых длин волн.
    """
    def __init__(self, operator,
                 config: _Config = _Config()):
        """
        Конструктор класса.

        Параметры:
            operator: оператор обработки surface.
            config: конфигурация расчётной системы.
        """
        super().__init__()
        self.set_config(config)
        self.surface = _nn.Parameter(
            _torch.zeros((1, 1, config.doe_size, config.doe_size),
                         dtype=_torch.float,
                         device=config.device))
        self.weights = self.surface
        self.selection = _torch.arange(self.__config.wavelength.size(0))
        self.__operator = operator
        # self.__dropout = _torch.nn.Dropout()

    def set_config(self, config: _Config):
        """
        Метод замены конфигурационных данных.

        Параметры:
            config: конфигурация расчётной системы.
        """
        self.__config: _Config = config
        self.__scale_factor = config.doe_pixel_size / config.pixel_size
        zero_count = (config.array_size - self.__scale_factor * config.doe_size) / 2
        self.__pad = _nn.ZeroPad2d(int(zero_count))

    def set_operator(self, operator):
        """
        Метод замены оператора обработки surface.

        Параметры:
            operator: оператор обработки surface.
        """
        self.__operator = operator

    def get_distance(self) -> float:
        """
        Метод получения дистанции распространения светового поля сквозь элемент.

        Returns:
            Дистанция распространения светового поля сквозь элемент.
        """
        return 0.0

    def get_phase(self, device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Параметры:
            device: устройство хранения данных.

        Returns:
            Распределение фазы ДОЭ.
        """
        return self.__operator.get_phase(self.surface, self.__config, device)

    def get_microrelief(self,
                        device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения высоты микрорельефа ДОЭ.

        Параметры:
            device: устройство хранения данных.

        Returns:
            Высота микрорельефа ДОЭ.
        """
        return self.__operator.get_microrelief(self.surface, self.__config, device)

    def get_ratio(self, device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения коэфициентов высоты микрорельефа ДОЭ.

        Параметры:
            device: устройство хранения данных.

        Returns:
            Коэфициенты высота микрорельефа ДОЭ.
        """
        return self.__operator.get_ratio(self.surface, self.__config, device)

    def get_phase_in_area(self,
                          device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ в расчётной области.

        Параметры:
            device: устройство хранения данных.

        Returns:
            Распределение фазы ДОЭ в расчётной области.
        """
        data = self.__operator.get_phase(self.surface, self.__config, device)
        return self.__to_area(data)

    def get_microrelief_in_area(self,
                                device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения высоты микрорельефа ДОЭ в расчётной области.

        Параметры:
            device: устройство хранения данных.

        Returns:
            Высота микрорельефа ДОЭ в расчётной области.
        """
        data = self.__operator.get_microrelief(self.surface, self.__config, device)
        return self.__to_area(data)

    def get_ratio_in_area(self,
                          device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения коэфициентов высоты микрорельефа ДОЭ в расчётной области.

        Параметры:
            device: устройство хранения данных.

        Returns:
            Коэфициенты высота микрорельефа ДОЭ в расчётной области.
        """
        data = self.__operator.get_ratio(self.surface, self.__config, device)
        return self.__to_area(data)

    def __interpolate(self, data: _torch.Tensor) -> _torch.Tensor:
        return _interpolate(data, scale_factor=self.__scale_factor, mode="nearest")

    def __to_area(self, data: _torch.Tensor) -> _torch.Tensor:
        data = self.__interpolate(data)
        return self.__pad(data)

    def __get_phase(self, data: _torch.Tensor) -> _torch.Tensor:
        # data = self.__dropout(data)
        angle = self.__operator(data, self.__config, self.selection)
        angle = self.__to_area(angle)
        return _torch.exp(1j * angle)

    def propagation(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод прохождения светового поля свкозь ДОЭ.

        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после прохождения сквозь ДОЭ.
        """
        return (field.to(self.__config.device)
                * self.__get_phase(self.surface.to(self.__config.device)))

    def forward(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод прохождения светового поля свкозь ДОЭ.

        Параметры:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после прохождения сквозь ДОЭ.
        """
        return self.propagation(field)


class DOE_by_phase:
    """
    Класс оператора обработки данных как параметры распределения фазы.
    """
    def get_phase(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Распределение фазы ДОЭ.
        """
        return data.detach().to(device)

    def get_microrelief(self,
                        data: _torch.Tensor,
                        config: _Config,
                        device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения высоты микрорельефа ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Высота микрорельефа ДОЭ.
        """
        return data.detach().to(device) / config.K / (config.N - 1)

    def get_ratio(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения коэфициентов высоты микрорельефа ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Коэфициенты высота микрорельефа ДОЭ.
        """
        microrelief = self.get_microrelief(data, config).to(device)
        return (microrelief / config.H_max).arcsin()

    def __call__(self,
                 data: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            selection: номера выбранных длин волн.

        Returns:
            Распределение фазы ДОЭ.
        """
        return data


class DOE_by_microrelief:
    def get_phase(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ в расчётной области.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Распределение фазы ДОЭ в расчётной области.
        """
        return (data.detach().to(device)
                * (config.N[:, None, None, None].to(device) - 1)
                * config.K[:, None, None, None].to(device))

    def get_microrelief(self,
                        data: _torch.Tensor,
                        config: _Config,
                        device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения высоты микрорельефа ДОЭ в расчётной области.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Высота микрорельефа ДОЭ в расчётной области.
        """
        return data.detach().to(device)

    def get_ratio(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения коэфициентов высоты микрорельефа ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Коэфициенты высота микрорельефа ДОЭ.
        """
        return (data.detach() / config.H_max).arcsin().to(device)

    def __call__(self,
                 data: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            selection: номера выбранных длин волн.

        Returns:
            Распределение фазы ДОЭ.
        """
        return (data.to(config.device)
                * (config.N[selection, None, None, None].to(config.device) - 1)
                * config.K[selection, None, None, None].to(config.device))


class DOE_by_ratio:
    def get_phase(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ в расчётной области.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Распределение фазы ДОЭ в расчётной области.
        """
        return (data.detach().to(device).sin().abs()
                * config.H_max
                * (config.N[:, None, None, None].to(device) - 1)
                * config.K[:, None, None, None].to(device))

    def get_microrelief(self,
                        data: _torch.Tensor,
                        config: _Config,
                        device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения высоты микрорельефа ДОЭ в расчётной области.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Высота микрорельефа ДОЭ в расчётной области.
        """
        return data.detach().sin().abs().to(device) * config.H_max

    def get_ratio(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения коэфициентов высоты микрорельефа ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Коэфициенты высота микрорельефа ДОЭ.
        """
        return data.detach().to(device)

    def __call__(self,
                 data: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            selection: номера выбранных длин волн.

        Returns:
            Распределение фазы ДОЭ.
        """
        return (data.to(config.device).sin().abs() * config.H_max
                * (config.N[selection, None, None, None].to(config.device) - 1)
                * config.K[selection, None, None, None].to(config.device))


class DOE_by_quadrics:
    def __init__(self, config: _Config):
        self.__parabolas = -config.K / 2 / config.distance * (config.X - config.Y)**2

    def __get_max(self,
                  quadrics: _torch.Tensor,
                  parabolas: _torch.Tensor):
        return (quadrics[..., None] + parabolas[None, None, :, None, :]).amax((2))

    def __get_phase_internal(self,
                             data: _torch.Tensor,
                             config: _Config):
        device = data.get_device()
        max_parabolas = self.__get_max(data, self.__parabolas.to(device))
        return self.__get_max(max_parabolas, self.__parabolas.to(device))

    def get_phase(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ в расчётной области.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Распределение фазы ДОЭ в расчётной области.
        """
        return self.__get_phase_internal(data.detach().to(device), config)

    def get_microrelief(self,
                        data: _torch.Tensor,
                        config: _Config,
                        device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения высоты микрорельефа ДОЭ в расчётной области.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Высота микрорельефа ДОЭ в расчётной области.
        """
        return (self.__get_phase_internal(data.detach().to(device), config)
                / config.K
                / (config.N - 1))

    def get_ratio(self,
                  data: _torch.Tensor,
                  config: _Config,
                  device: _torch.device = _torch.device('cpu')) -> _torch.Tensor:
        """
        Метод получения коэфициентов высоты микрорельефа ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            device: устройство хранения данных.

        Returns:
            Коэфициенты высота микрорельефа ДОЭ.
        """
        microrelief = self.get_microrelief(data.detach().to(device), config, device)
        return (microrelief / config.H_max).arcsin()

    def __call__(self,
                 data: _torch.Tensor,
                 config: _Config,
                 selection: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения распределения фазы ДОЭ.

        Параметры:
            data: параметры поверхности ДОЭ.
            config: конфигурация расчётной системы.
            selection: номера выбранных длин волн.

        Returns:
            Распределение фазы ДОЭ.
        """
        return self.__get_phase_internal(data, config)
