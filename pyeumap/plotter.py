import matplotlib.pyplot as plt
from typing import Union, List, Tuple
import rasterio as rio
import numpy as np
from pathlib import Path

def plot_rasters(
    *rasters: Union[Tuple[str], Tuple[np.ndarray], Tuple[Path]],
    out_file: Union[str, Path]=None,
    vertical_layout: bool=False,
    figsize: float=10,
    spacing: float=0.01,
    cmaps: Union[str, List[str]]='Spectral',
    titles: List[str]=[],
    dpi: int=150,
    nodata: List[Union[int, float]]=None,
):
    if isinstance(rasters, (str, Path, np.ndarray)):
        rasters = [rasters]
    else:
        rasters = list(rasters)

    if isinstance(cmaps, str):
        cmaps = [cmaps] * len(rasters)

    if nodata is None or isinstance(nodata, (int, float)):
        nodata = [nodata] * len(rasters)

    for i, r in enumerate(rasters):
        if isinstance(r, (str, Path)):
            with rio.open(r) as src:
                rasters[i] = src.read(1)
                if nodata[i] is None:
                    nodata[i] = src.nodata

    if titles and isinstance(titles, str):
        titles = [titles]

    subplot_dims = [1, len(rasters)]

    if vertical_layout:
        subplot_dims = subplot_dims[::-1]
        plot_w = max((r.shape[1] for r in rasters))
        plot_h = sum((r.shape[0] for r in rasters))
        fig_dims = (figsize, figsize*plot_h/plot_w)
    else:
        plot_h = max((r.shape[0] for r in rasters))
        plot_w = sum((r.shape[1] for r in rasters))
        fig_dims = (figsize, figsize*plot_h/plot_w)
    fig, axes = plt.subplots(
        *subplot_dims,
        #figsize=fig_dims,
        frameon=False,
        dpi=dpi,
    )
    fig.subplots_adjust(hspace=spacing, wspace=spacing)
    fig.patch.set_alpha(0)
    if len(rasters) == 1:
        axes = [axes]
    for i, (ax, arr, cmap, nd) in enumerate(zip(axes, rasters, cmaps, nodata)):
        if nd is None:
            alpha = None
        else:
            alpha = np.full_like(arr, 1, dtype=type(nd))
            alpha[arr==nd] = 0
        ax.imshow(arr, alpha=alpha, cmap=cmap)
        ax.axis('off')
        if titles:
            if vertical_layout:
                ax.set_ylabel(titles[i])
            else:
                ax.set_title(titles[i])
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
