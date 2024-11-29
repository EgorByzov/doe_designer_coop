from typing import List, Union, Tuple

import matplotlib
import torch
from torch import Tensor
from matplotlib import pyplot as plt

from models.quad_solutions.base import QuadricSolution
from services.distribution.planar_distribution import make_2Ddistribution_map


def show_planar_flux(image: Union[Tensor, List[Tensor]]):
    cur_cmap = matplotlib.colormaps['gray_r']
    if type(image) is Tensor:
        aspect = 'equal' if image.size(0) == image.size(1) else 'auto'

        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)

        image_handle = ax.imshow(image.detach().cpu().numpy(), aspect=aspect, cmap=cur_cmap)
        fig.colorbar(image_handle, ax=ax)
    else:
        aspect = 'equal' if image[0].size(0) == image[0].size(1) else 'auto'
        num_imgs = len(image)

        fig, ax = plt.subplots(1, num_imgs)
        fig.set_size_inches(7 * num_imgs, 5)
        image_handle = []

        for i in range(num_imgs):
            image_handle.append(
                ax[i].imshow(image[i].detach().cpu().numpy(), aspect=aspect, cmap=cur_cmap))
            fig.colorbar(image_handle[i], ax=ax[i])

    fig.set_dpi(100)
    plt.show()
    return fig, ax, image_handle


def show_3D_surface(image: Union[Tuple[Tensor,Tensor,Tensor], List[Tuple[Tensor,Tensor,Tensor]]], fig_size:Tuple[int,int]=(7,5)):
    cur_cmap = matplotlib.colormaps['gray_r']
    fig = plt.figure()
    if type(image) is tuple:
        x, y, z = image[0].detach().cpu().numpy(),\
                  image[1].detach().cpu().numpy(),\
                  image[2].detach().cpu().numpy()

        aspect = 'equal' if image[0].size(0) == image[0].size(1) else 'auto'
        ax = fig.add_subplot(projection='3d')
        fig.set_size_inches(fig_size[0], fig_size[1])

        image_handle = ax.plot_surface(x, y, z, linewidth=.5, antialiased=False, alpha=0.8, cmap=cur_cmap)

        # image_handle = ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
        #                 alpha=0.3, cmap=cur_cmap)
        # ax.contour(x, y, z, zdir='z', offset=-100, cmap='coolwarm')
        # ax.contour(x, y, z, zdir='x', offset=-40, cmap='coolwarm')
        # ax.contour(x, y, z, zdir='y', offset=40, cmap='coolwarm')
        fig.colorbar(image_handle, ax=ax)
    else:
        aspect = 'equal' if image[0][0].size(0) == image[0][0].size(1) else 'auto'
        num_imgs = len(image)

        fig.set_size_inches(fig_size[0] * num_imgs, fig_size[1])
        image_handle = []
        ax = []
        for i in range(num_imgs):
            ax.append(fig.add_subplot(1, num_imgs, i+1, projection='3d'))
            x, y, z = image[i][0].detach().cpu().numpy(), \
                      image[i][1].detach().cpu().numpy(), \
                      image[i][2].detach().cpu().numpy()

            image_handle.append(ax[i].plot_surface(x, y, z, linewidth=0, antialiased=False, cmap=cur_cmap))

            # image_handle.append(ax[i].plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
            #                                alpha=0.3, cmap=cur_cmap))
            # ax[i].contour(x, y, z, zdir='z', offset=-100, cmap='coolwarm')
            # ax[i].contour(x, y, z, zdir='x', offset=x.min(), cmap='coolwarm')
            # ax[i].contour(x, y, z, zdir='y', offset=y.max(), cmap='coolwarm')
            fig.colorbar(image_handle[i], ax=ax[i])

    fig.set_dpi(100)
    plt.show()
    return fig, ax, image_handle


def update_map(quad_solution: QuadricSolution, en, fig, ax):
    traces = quad_solution.req_distr.get_traces(en=en)
    distr_map = make_2Ddistribution_map(traces=traces)

    ax.imshow(distr_map)
    fig.canvas.draw_idle()
