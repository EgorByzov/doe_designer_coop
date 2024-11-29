import numpy as np
import torch
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# Setup
from spline2_eval import spline2_eval
from spline2_fit import spline2_fit

degree = 3
errAbs = 0.05

def err(s, a, b):
    return a + (b - a) * torch.rand(s)

def f(x, y):
    # return torch.cos(10 * (x**2 + y)) * torch.sin(10 * (x + y**2))
    return x**2 + y**2

def f_err(x, y):
    return f(x, y) + err(x.shape[0], -errAbs / 2, errAbs / 2).reshape(x.size())

xMin = -1
xMax = 1
yMin = -1
yMax = 1
nx = 100
ny = 100
nknots_x = 64
nknots_y = 64

x = torch.linspace(xMin, xMax, nx).view(-1, 1)
y = torch.linspace(yMin, yMax, ny).view(-1, 1)
x, y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='xy')
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

z = f(x, y)
z_err = f_err(x, y)
knots_x = torch.linspace(xMin, xMax, nknots_x)
knots_y = torch.linspace(yMin, yMax, nknots_y)

# Fit
c = spline2_fit(x, y, z_err, degree, knots_x, knots_y)
z_fit = spline2_eval(x, y, c, degree, knots_x, knots_y)

# Plot
x = x.reshape(nx, ny)
y = y.reshape(nx, ny)
z = z.reshape(nx, ny)
z_err = z_err.reshape(nx, ny)
z_fit = z_fit.reshape(nx, ny)

plt.figure()
plt.imshow(z_err.numpy())
plt.figure()
plt.imshow(z_fit.numpy())
plt.figure()
plt.imshow((z - z_fit).numpy())
plt.show()

# Stat
aver = torch.sum(z) / (nx * ny)
rrmse = torch.sqrt(torch.sum((z - z_fit)**2) / (nx * ny)) / aver * 100

