from typing import List

import torch
from torch import Tensor
from torch.nn import Module, Softmax, Parameter, ParameterList

from bspline.layered_bspline import LayeredBSpline
from schemas import PropagationParameters
from sinc_propagator import propagation_sinc_prepare, propagation_sinc


class LayeredDOE(Module):
    def __init__(self, x, y, degree, c_s, knots_x_s, knots_y_s):
        super().__init__()
        self.x = x
        self.y = y
        self.c_s = ParameterList([Parameter(x) for x in c_s])
        self.layered_spline = LayeredBSpline(d_s=degree, c_s=self.c_s, knots_x_s=knots_x_s, knots_y_s=knots_y_s)

    def forward(self, x: Tensor):
        W = torch.exp(1j * self.get_phase())
        W = W.to(dtype=x.dtype)
        return W * x

    def get_phase(self):
        return self.layered_spline.eval(x=self.x, y=self.y)

    def get_phase_surface(self):
        phi = self.get_phase()
        n = torch.ceil(phi.min().abs() / 2 / torch.pi)
        return torch.fmod(phi + n * 2 * torch.pi, 2 * torch.pi)


class SimpleLayeredDNN(Module):
    def __init__(self,
                 propagation_params: PropagationParameters,
                 target_regions: List[Tensor],
                 layered_doe: LayeredDOE):
        super().__init__()

        self.pre_prod = Softmax(dim=1)
        self.doe = layered_doe
        self.propagation_params = propagation_params
        self.target_regions = target_regions

        self._propagator_params = [propagation_sinc_prepare(field_shape=propagation_params.im_size,
                                                            wavenumber=propagation_params.wave_number,
                                                            side_length=propagation_params.aperture_size,
                                                            propagation_dist=f) for f in propagation_params.dists]

    def trace_field(self, input):
        res = propagation_sinc(u1=input, propagator_params=self._propagator_params[0])
        res = self.doe(res)
        res = propagation_sinc(u1=res, propagator_params=self._propagator_params[1])
        return res

    def forward(self, x):
        x = self.trace_field(x)
        x = abs(x) ** 2
        x = x * self.target_regions
        x = x.sum(dim=(-2, -1))
        x = self.pre_prod(x)
        return x


class CascadedLDNN(Module):
    def __init__(self,
                 propagation_params: PropagationParameters,
                 target_regions: List[Tensor],
                 layered_doe: List[LayeredDOE]):
        super().__init__()
        self.pre_prod = Softmax(dim=1)
        self.doe = ParameterList(layered_doe)
        self.propagation_params = propagation_params
        self.target_regions = target_regions

        self._propagator_params = [propagation_sinc_prepare(field_shape=propagation_params.im_size,
                                                            wavenumber=propagation_params.wave_number,
                                                            side_length=propagation_params.aperture_size,
                                                            propagation_dist=f) for f in propagation_params.dists]

    def trace_field(self, input):
        res = input
        for idx, cur_doe in enumerate(self.doe):
            res = propagation_sinc(u1=res, propagator_params=self._propagator_params[idx])
            res = cur_doe(res)
        res = propagation_sinc(u1=res, propagator_params=self._propagator_params[-1])
        return res

    def forward(self, x):
        x = self.trace_field(x)
        x = abs(x) ** 2
        x = x * self.target_regions
        x = x.sum(dim=(-2, -1))
        x = self.pre_prod(x)
        return x
