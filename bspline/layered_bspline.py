from typing import List

from torch import Tensor

from bspline.spline2_eval import spline2_eval


class LayeredBSpline:
    c_s: List
    d_s: List
    knots_x_s: List
    knots_y_s: List

    def __init__(self, d_s: List | int, c_s: List, knots_x_s: List, knots_y_s: List):
        self.d_s = d_s if d_s is list else [d_s] * len(c_s)
        self.c_s = c_s
        self.knots_x_s = knots_x_s
        self.knots_y_s = knots_y_s

    @property
    def num_layeres(self):
        return len(self.c_s)

    @property
    def get_params(self):
        return self.c_s

    def eval(self, x: Tensor, y: Tensor):
        z = x.new_zeros(x.size())

        for i in range(self.num_layeres):
            z += spline2_eval(x, y, self.c_s[i], self.d_s[i], self.knots_x_s[i], self.knots_y_s[i])

        return z
