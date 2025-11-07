# Fee's ramble on getting to terms with the C++ source code

So there's no doxygen, I kind of recall having sorta made doxygen for [openlisem](https://github.com/vjetten/openlisem) that at least auto-generated the class inheritance structure and such... lemme check...

pointers:

- https://github.com/openlandmap/scikit-map/blob/develop/skmap/tiled_data.py check where "sb." is called, all functions are in https://github.com/openlandmap/scikit-map/blob/develop/skmap/src/skmap_bindings.cpp
- cpp files are in https://github.com/openlandmap/scikit-map/tree/develop/skmap/src and headers in https://github.com/openlandmap/scikit-map/tree/develop/skmap/include
- http://192.168.1.54:8817/lab/tree/mnt/europa/global_soc/scikit-map/test_ai4sh_shdc_props.ipynb

but actually the `docs/notebooks` will be like my main notebooks pointers.

## Mandatory [Richdem design principles](https://richdem.readthedocs.io/en/latest/philosophy.html):

The design of RichDEM is guided by these principles:

- **Algorithms will be well-tested.** Every algorithm is verified by a rigorous testing procedure. See below.
- **Algorithms will be fast, without compromising safety and accuracy.** The algorithms used in RichDEM are state of the art, permitting analyses that would take days on other systems to be performed in hours, or even minutes.
- **Algorithms will be available as libraries, whenever possible.** RichDEM is designed as a set of header-only C++ libraries, making it easy to include in your projects and easy to incorporate into other programming languages. RichDEM also includes apps, which are simple wrappers around the algorithms, and a limited, but growing, set of algorithms which may have special requirements, like MPI, that make them unsuitable as libraries. These are available as programs.
- **Programs will have a command-line interface, not a GUI.** Command-line interfaces are simple to use and offer extreme flexibility for both users and programmers. They are available on every type of operating system. RichDEM does not officially support any GUI. Per the above, encapsulating RichDEM in a high-level interface of your own is not difficult.
- **Algorithms and programs will be portable.** Linux, Mac, and Windows should all be supported.
- **The code will be beautiful.** RichDEM’s code utilizes sensible variable names and reasonable abstractions to make it easy to understand, use, and design algorithms. The code contains extensive internal documentation which is DOxygen compatible.
- **Programs and algorithms will provide useful feedback.** Progress bars will appear if desired and the output will be optimized for machine parsing.
- **Analyses will be reproducible.** Every time you run a RichDEM command that command is logged and timestamped in the output data, along with the version of the program you created the output with. Additionally, a history of all previous manipulations to the data is kept. Use rd_view_processing_history to see this.**


## making documentation

### Quickstart

### Tutorial: machine learning pipeline

![](../../docs/img/pipeline.svg)


### In-depth space-time overlay

#### Whales

### In-depth: target-scale predictions

In modern computing workloads, parallelization is a huge speedup. For optimal performance, operations should work on contiguous blocks of memory. Data is loaded from a backend using GDAL VRTs and whales into a `TiledData` array. 

![](../../docs/img/overview_non-edit.svg)

There is some change going on, because the main thing changed to doing the io part of data mangling really really fast, but also like metadata and pre-processing of data

GDAL also has [some options](https://gdal.org/en/stable/drivers/raster/vrt.html#derived-bands-pixel-functions) for that in VRTs🐋

### Contributing

TODO: put [logo readme](../../logo/readme.md) here with like color guidelines and such.

### C++

#### Doxygen

So,

```bash
sudo dnf install doxygen
doxygen -g # create Doxyfile
doxygen Doxyfile # create documentation
```
then in a browser `Ctrl-o` and select `html/index.html` or `$ firefox html/index.html`
But this is still very empty, so change in doxyfile:

```
EXTRACT_ALL             = YES
EXTRACT_LOCAL_CLASSES   = YES
HIDE_UNDOC_NAMESPACES   = NO
RECURSIVE               = YES
```

then again `doxygen Doxyfile` and refresh

#### Clangd language server

So I wanted to get nice code completion and such, so that's what clangd can help with, 

#### Unit tests?

Would be great to have like unit tests for most things

## checking where `sb.` is called:

`sb_arr` 

1. `warp_tile`: warps tile or fills with all zeroes
2. `TiledData.convert_nan_to_value(value)`: sb.maskNan
3. `TiledData.convert_nan_to_median`: `sb.computePercentiles`

At this point I was like: hey, I can add docstrings and installed [autodocstring](https://open-vsx.org/vscode/item?itemName=njpwerner.autodocstring).

4. 

## doing the installation thing

so yeah, we're doing the conda+pip thing, but also need to do some system-level packages and the problem is that the c++ part gets compiled against system-gdal, which mismatches with the pinned gdal in `conda_env.yml` and then we're sad.

[sidenote](https://stackoverflow.com/a/79805924/14681457)

sooo, 

```
$ gdalinfo --version
GDAL 3.10.3, released 2025/04/01
```

modify `conda_env.yml` to reflect this...  
also needs `python=3.9`...  

Forget that, I think we can just install everything with conda/micromamba and then have everything use the local versions. That's nice... Except build isolation and we should really be moving to a `pyproject.toml`. but anyways:

```
pip install --no-build-isolation -e .[full]
```

But then for some reason the `skmap.io.process.SeasConvFill` didn't exist???? even after adding `.[full]` in stead of `.`... whatevs...



## memmap?

So we create like binary files for arrays using `np.memmap`, idk if that'll be important later on, but it seems different than a rust-like memmap, which is access to the direct io buffers for e.g. network requests that I didn't use in `tiff2` because there's lots of misaligned data there.

apparently its old and we don't care


