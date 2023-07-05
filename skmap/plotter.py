'''
Functions to plot raster data
'''
from typing import Iterable

try:
	import math
	import matplotlib.pyplot as plt
	import skimage.exposure as exposure
	from mpl_toolkits.axes_grid1 import ImageGrid
	from matplotlib.colors import ListedColormap
	from typing import Union, List, Iterable
	import rasterio as rio
	import numpy as np
	from pathlib import Path

	def _percent_clip(data, perc_min, perc_max):
		return (data - np.percentile(data, perc_min))/(np.percentile(data, perc_max) - np.percentile(data, perc_min))

	def _plot_rgb(raster, perc_min=2, perc_max=98):

		bands = range(0, raster.shape[2])
		data_equalized = []
		for band in bands:
			data_equalized.append(_percent_clip(raster[:, :, band], perc_min, perc_max))

		data_equalized = np.stack(data_equalized, axis=2)
		plt.imshow(data_equalized)

	def plot_stac_collection(
		collection, 
		thumb_id='thumbnail',
		ncols = 4, 
		figsize=(15, 25), 
		axes_pad=(0,0.4)
	):
		"""
    
	    Plot the asset thumbnails for all items of a STAC collection.

	    :param collection: STAC collection instance ``pystac.collection.Collection``.
	    :param thumb_id: Asset id of thumbnails.
	    :param ncols: Number of columns used to define the grid.
	    :param figsize: Print size of the horizontal axis of the plot (passed to ``matplotlib``).
		:param axes_pad: Padding space between the plots of the grid.


	    """
		
		items = list(collection.get_all_items())

		nrows = math.ceil(len(items) / 4)

		fig = plt.figure(figsize=figsize)
		fig.tight_layout()
		grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)

		for ax, item in zip(grid, items):
			thumbnail_url = item.assets[thumb_id].href

			start_dt = item.properties['start_datetime']
			end_dt = item.properties['end_datetime']
			title = f'{start_dt} - {end_dt}'

			ax.title.set_text(title)
			ax.get_yaxis().set_ticks([])
			ax.get_xaxis().set_ticks([])

			im = plt.imread(thumbnail_url)
			ax.imshow(im)

		plt.show()

	def plot_rasters(
		*rasters: Union[Iterable[str], Iterable[np.ndarray], Iterable[Path]],
		out_file: Union[str, Path]=None,
		vertical_layout: bool=False,
		figsize: float=10,
		spacing: float=0.01,
		cmaps: Union[str, List[str]]='Spectral',
		titles: List[str]=[],
		dpi: int=150,
		nodata: List[Union[int, float]]=None,
		vmin: List[Union[int, float]]=None,
		vmax: List[Union[int, float]]=None,
		perc_clip: bool=False,
		perc_min: List[Union[int, float]]=2,
		perc_max: List[Union[int, float]]=98,
	):
		"""
		Plots data from one or more rasters.

		Preserves pixel aspect ratio, removes axes and ensures transparency on nodata.

		Uses ``matplotlib.pyplot.imshow`` [1].

		:param *rasters:        List of rasters, passed as either data or file paths. If 3D (multiband) data is passed (as ``numpy`` array(s)), the first axis of the array must correspond to the band index.
		:param out_file:        Path to save figure if not ``None``.
		:param vertical_layout: Produces a vertical array of plots if ``True``, horizontal if ``False`` (default).
		:param figsize:         Print size of the horizontal axis of the plot (passed to ``matplotlib``). The vertical size is calculated automatically.
		:param spacing:         Spacing between raster plots.
		:param cmaps:           Colormap to use for singleband plots, or list of colormaps (applied respectively). Must contain valid ``matplotlib`` colormaps [2]. For rasters with multiple (3 or more) bands, this argument is ignored and RGB plots are produced.
		:param titles:          Titles to produce for each plot.
		:param dpi:             DPI of the figure.
		:param nodata:          Nodata value or list of values respective to each raster. If ``None`` and ``*rasters`` contains file paths, ``nodata`` will be inferred from raster source.
		:param vmin:            Minimum value to clip data.
		:param vmax:            Maximum value to clip data.
		:param perc_clip:       Clips rasters with percentiles if ``True``.
		:param perc_min:        Minimum percentile to clip with if ``perc_clip=True``.
		:param perc_max:        Maximum percentile to clip with if ``perc_clip=True``.

		Examples
		========

		>>> from skmap import plotter
		>>> import numpy as np
		>>>
		>>> singleband = np.random.randint(0, 255, [5, 5])
		>>> multiband = np.random.randint(0, 255, [3, 5, 5])
		>>>
		>>> plotter.plot_rasters(
		>>>     singleband,
		>>>     multiband,
		>>>     titles=['single band', 'RGB'],
		>>>     figsize=4,
		>>>     cmaps='Greens',
		>>> )

		References
		==========

		[1] `Matplotlib imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_

		[2] `Matplotlib colormaps <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_

		"""

		if isinstance(rasters, (str, Path, np.ndarray)):
			rasters = [rasters]
		else:
			rasters = list(rasters)

		if isinstance(cmaps, (str, ListedColormap)):
			cmaps = [cmaps] * len(rasters)

		if not isinstance(vmin, Iterable):
			vmin = [vmin] * len(rasters)

		if not isinstance(vmax, Iterable):
			vmax = [vmax] * len(rasters)

		if not isinstance(nodata, Iterable):
			nodata = [nodata] * len(rasters)

		for i, r in enumerate(rasters):
			if isinstance(r, (str, Path)):
				with rio.open(r) as src:
					rasters[i] = src.read()
					if nodata[i] is None:
						nodata[i] = src.nodata
			if len(rasters[i].shape) < 3:
				rasters[i] = rasters[i].reshape(1, *rasters[i].shape)
			rasters[i] = np.stack(rasters[i], axis=-1)[:, :, :4]
			if perc_clip:
				try:
					bands = range(0, rasters[i].shape[2])
					data_equalized = []
					for band in bands:
						data_equalized.append(_percent_clip(rasters[i][:, :, band], perc_min, perc_max))
					data_equalized = np.stack(data_equalized, axis=-1)
					rasters[i] = data_equalized
				except IndexError:
					pass

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
			figsize=fig_dims,
			frameon=False,
			dpi=dpi,
		)
		fig.subplots_adjust(hspace=spacing, wspace=spacing)
		fig.patch.set_alpha(0)
		if len(rasters) == 1:
			axes = [axes]
		for i, (ax, arr, cmap, nd, _vmin, _vmax) in enumerate(zip(
			axes,
			rasters,
			cmaps,
			nodata,
			vmin,
			vmax,
		)):
			if nd is None:
				alpha = None
			else:
				alpha = np.full_like(arr, 1, dtype='uint8')
				alpha[arr==nd] = 0
				if len(alpha.shape) == 3:
					alpha = alpha[:,:,0]
			ax.imshow(arr, alpha=alpha, cmap=cmap, vmin=_vmin, vmax=_vmax)
			ax.axis('off')
			if titles:
				if vertical_layout:
					ax.set_ylabel(titles[i])
				else:
					ax.set_title(titles[i])
		if out_file is not None:
			plt.savefig(out_file, bbox_inches='tight')

except ImportError as e:
	from skmap.misc import _warn_deps
	_warn_deps(e, 'plotter')
