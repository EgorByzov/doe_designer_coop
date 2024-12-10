from typing import List, Union

import matplotlib
from matplotlib import pyplot as plt
from torch import Tensor


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
