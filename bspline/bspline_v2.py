import torch


def bspline_v2(x: torch.Tensor, jmax, d, t):
    cur_f = x.new_zeros(x.numel(), jmax + d)
    t = t.flatten()
    x = x.flatten()

    for j in range(jmax + d):
        if (j + 1) >= t.numel() - d - 1:
            cur_f[x >= t[j], j] = 1
        else:
            cur_f[(x >= t[j]) & (x < t[j + 1]), j] = 1

    for cur_d in range(1, d + 1):
        last_f = cur_f.clone()
        cur_f = x.new_zeros(x.numel(), jmax + d)
        for j in range(jmax + d - cur_d):
            if t[j + cur_d] != t[j]:
                leftc = (x - t[j]) / (t[j + cur_d] - t[j])
                cur_f[:, j] += leftc * last_f[:, j]
            if t[j + cur_d + 1] != t[j + 1]:
                rightc = (t[j + cur_d + 1] - x) / (t[j + cur_d + 1] - t[j + 1])
                cur_f[:, j] += rightc * last_f[:, j + 1]

    f = cur_f[:, :jmax]
    return f
