import torch

from bspline_v2 import bspline_v2


def spline2_fit(x, y, z, d, knots_x, knots_y, lambda_param=0):
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    xmin = knots_x[0]
    xmax = knots_x[-1]
    tx = torch.cat((x.new_full((1, d), xmin), knots_x.view(1, -1), x.new_full((1, d), xmax)), dim=1)

    ymin = knots_y[0]
    ymax = knots_y[-1]
    ty = torch.cat((x.new_full((1, d), ymin), knots_y.view(1, -1), x.new_full((1, d), ymax)), dim=1)

    ncoeff_x = len(knots_x) + d - 1
    ncoeff_y = len(knots_y) + d - 1

    bspline_x = bspline_v2(x, ncoeff_x, d, tx)[0]
    bspline_y = bspline_v2(y, ncoeff_y, d, ty)[0]

    # b_dense = (x.numel() * ncoeff_x * ncoeff_y < 0.1 * 1024**3 / 8)
    B = x.new_zeros(x.numel(), ncoeff_x * ncoeff_y)

    # if not b_dense:
    #     B = B.to_sparse()
    #     if torch.count_nonzero(bspline_x) / bspline_x.numel() < 0.2:
    #         bspline_x = bspline_x.to_sparse()
    #     if torch.count_nonzero(bspline_y) / bspline_y.numel() < 0.2:
    #         bspline_y = bspline_y.to_sparse()

    for j in range(ncoeff_x):
        for k in range(ncoeff_y):
            B[:, (j * ncoeff_y + k)] = bspline_x[:, j] * bspline_y[:, k]

    # if b_dense:
    if lambda_param != 0:
        c = torch.linalg.lstsq(B.T @ B + lambda_param * torch.eye(ncoeff_x * ncoeff_y), B.T @ z).solution
    else:
        c = torch.linalg.lstsq(B, z).solution
    # else:
    #     c = torch.linalg.solve(B.T @ B + lambda_param * torch.sparse.eye(ncoeff_x * ncoeff_y), B.T @ z)

    return c