from typing import Tuple, Dict

import numpy as np
import torch
from scipy.linalg import toeplitz
from scipy.special import fresnel
from torch import Tensor

from core.config import torch_complex_float_type, device


def propagation_sinc_prepare(field_shape: Tuple[int, int], side_length: float, wavenumber: float,
                             propagation_dist: float) -> Dict:
    """
        Args:
            field_shape: Source plane field size
            side_length: Source and observation plane side length
            wavenumber: ...
            propagation_dist: Propagation distance

        Returns:
            (eikz, ome_x, ome_y)
    """

    m, n = field_shape  # get input field array size
    dx = side_length / m  # sample interval
    bnd_w = 0.5 / dx  # bandwidth

    eikz = np.asarray(np.exp(1j * wavenumber * propagation_dist))
    eikz = torch.complex(
        real=torch.from_numpy(eikz.real),
        imag=torch.from_numpy(eikz.imag)
    )
    sq2p = np.sqrt(2.0 / np.pi)
    sqzk = np.emath.sqrt(2.0 * propagation_dist / wavenumber)

    x_obs = -0.5 * side_length + np.arange(m) * dx
    x_src = x_obs
    xm = x_obs[:, np.newaxis] - x_src[np.newaxis, :]

    # Compute weights for Fresnel diffraction integral
    mu1 = -np.pi * sqzk * bnd_w - xm / sqzk

    smu1_row1, cmu1_row1 = fresnel(sq2p * mu1[0, :])
    smu1_row1 = smu1_row1 / sq2p
    cmu1_row1 = cmu1_row1 / sq2p

    smu1_col1, cmu1_col1 = fresnel(sq2p * mu1[:, 0])
    smu1_col1 = smu1_col1 / sq2p
    cmu1_col1 = cmu1_col1 / sq2p

    smu1 = torch.from_numpy(toeplitz(smu1_col1, smu1_row1))
    cmu1 = torch.from_numpy(toeplitz(cmu1_col1, cmu1_row1))

    smu2 = -smu1.t()
    cmu2 = -cmu1.t()

    ome_x = (dx / torch.pi) / sqzk * torch.exp(
        torch.from_numpy(0.5 * 1j * (xm ** 2) * wavenumber / propagation_dist)) * (
                    cmu2 - cmu1 - 1j * (smu2 - smu1))
    ome_y = ome_x
    return {
        'eikz': eikz.to(dtype=torch_complex_float_type, device=device),
        'ome_x': ome_x.to(dtype=torch_complex_float_type, device=device),
        'ome_y': ome_y.to(dtype=torch_complex_float_type, device=device)
    }


def propagation_sinc(u1: Tensor, propagator_params: Dict) -> Tensor:
    """
        Fresnel propagation - sinc basis approach
        assumes same x and y side lengths and
        uniform sampling

        Args:
            u1: Source plane field
            propagator_params: eikz: exp(ikz), ome_x: X propagation matrix, ome_y: Y propagation matrix

        Returns:
            u2: Observation plane field
    """
    return propagator_params['eikz'] * (propagator_params['ome_x'] @ u1) @ propagator_params['ome_y'].t()


def _toeplitz(c, r=None):
    """
        Construct a Toeplitz matrix.

        The Toeplitz matrix has constant diagonals, with c as its first column
        and r as its first row. If r is not given, ``r == conjugate(c)`` is
        assumed.

        Parameters
        ----------
        c : array_like
            First column of the matrix.  Whatever the actual shape of `c`, it
            will be converted to a 1-D array.
        r : array_like, optional
            First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
            in this case, if c[0] is real, the result is a Hermitian matrix.
            r[0] is ignored; the first row of the returned matrix is
            ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
            converted to a 1-D array.

        Returns
        -------
        A : (len(c), len(r)) ndarray
            The Toeplitz matrix.
    """
    c = c.flatten()
    if r is None:
        r = c.conj()
    else:
        r = r.flatten()

    # Формируем 1-D массив, содержащий перевернутый c, за которым следует r[1:]
    vals = torch.cat((c.flip(0), r[1:]))
    out_shp = (len(c), len(r))

    # Создаем тензор Toeplitz с использованием strided view
    n = vals.stride(0)
    result = vals[len(c) - 1:].as_strided(size=out_shp, stride=(-n, n))

    # return result.clone()  # Возвращаем копию, чтобы избежать изменений в исходных данных
    return result
