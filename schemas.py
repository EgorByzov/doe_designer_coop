from typing import Tuple, List

from pydantic import BaseModel


class PropagationParameters(BaseModel):
    im_size: Tuple[int, int]
    wave_number: float
    aperture_size: float
    dists: List[float]
