import torch as _torch
import torch.nn as _nn
from .config import Config as _Config
from .propagator import Propagator as _Propagator
from .doe import DOE as _DOE
from .doe import DOE_by_microrelief as _DOE_by_microrelief
from tqdm.notebook import trange as _trange

def get_analytical_approximation(config: _Config,
                                 field: _torch.Tensor,
                                 propagator: _Propagator,
                                 target: _torch.Tensor,
                                 quantization: int = 256,
                                 gamma: None | _torch.Tensor = None,
                                 consts: None | _torch.Tensor = None,
                                 selection: None | int = None) -> _nn.ModuleList:
    """
    Функция подбора высоты микрорельефа согласно множеству фазовых функций.
 
    Параметры:
        config: конфигурация системы.
        field: распределение комплексной амплитуды входного поля.
        propagator: пропагатор светового поля распределения.
        target:  множество целевых фазовых функций.
        quantization: число уровней квантования микрорельефа линзы.
        gamma: константы важности целевых функций
        consts: константы целевых фазовых функций.
        selection: число дробления множества целевых функций.
 
    Returns:
        Список ДОЭ с подобранным микрорельефом.
    """
    scale = int(config.doe_pixel_size / config.pixel_size)
    avg = _nn.AvgPool2d(scale, stride = scale, padding = 0)
    
    if (gamma is None):
        gamma = _torch.ones((config.doe_count, config.wavelength.size(0), 1, 1, 1))
    else:
        gamma = gamma[..., None, None, None]

    if (consts is None):
        consts = _torch.zeros((config.doe_count, config.wavelength.size(0), 1, 1, 1))
    else:
        consts = consts[..., None, None, None]

    if (selection is None):
        selection = config.wavelength.size(0)

    field = field.cfloat().to(config.device)
    target = target.to(config.device)
    gamma = gamma.to(config.device)
    consts = consts.to(config.device)

    does = _nn.ModuleList([_DOE(operator = _DOE_by_microrelief(),
                              config = config)
                          for i in range(config.doe_count)])
    blank = _DOE(_DOE_by_microrelief(), config)

    count = int(config.wavelength.size(0) / selection + 0.5)
    
    ones = _torch.ones((1, 1, config.doe_size, config.doe_size)).to(config.device)
    idx = _torch.linspace(0, config.H_max, quantization)
    for i in range(len(does)):
        delta_max = 1e9 * _torch.ones((config.doe_size, config.doe_size)).to(config.device)
        progress = _trange(idx.size(0))
        for j in progress:
            blank.surface.data = ones * idx[j]

            delta = 0.0
            angle = blank.get_phase_in_area(config.device)
            for k in range(count):
                delta += _get_delta(gamma[i][k * selection:k * selection + selection],
                                   field[k * selection:k * selection + selection],
                                   angle[k * selection:k * selection + selection],
                                   target[i][k * selection:k * selection + selection],
                                   consts[i][k * selection:k * selection + selection])
            delta = avg(delta)[0]

            mask = delta < delta_max
            delta_max[mask] = delta[mask]
            does[i].surface.data[mask] = blank.surface.data[0, 0][mask]

        for k in range(count):
            does[i].selection = _torch.linspace(k * selection, k * selection + selection - 1, selection).int()
            propagator.selection = _torch.linspace(k * selection, k * selection + selection - 1, selection).int()
            field[k * selection:k * selection + selection] = propagator(does[i](field[k * selection:k * selection + selection]).detach())
    return does

def _get_delta(gamma: _torch.Tensor,
               field: _torch.Tensor,
               angle: _torch.Tensor,
               target: _torch.Tensor,
               consts: _torch.Tensor):
    return (gamma
            * field.abs()**2
            * (_torch.exp(1j * field.angle())
               * _torch.exp(1j * angle)
               - target * _torch.exp(1j * consts)).abs()**2).sum(0)