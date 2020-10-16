# pyeumap

pyeumap, the python package for eumap library, implements the follow workflow:

![pyeumap Workflow](../img/pyeumap_workflow.png)

## Installing

With `pip`:
```
python -m pip install -U git+https://gitlab.com/geoharmonizer_inea/eumap.git#pyeumap
```
## Usage

The follow code demonstration are based in this [benchmark dataset for land-cover classification](http://doi.org/10.5281/zenodo.4058447), available for different areas of the EU.

* [Temporal Gap-filling Demonstration](../demo/python/01_temporal_gapfilling.ipynb)
* [Overlay Demonstration](../demo/python/02_overlay.ipynb)
* [Land-Cover Mapping](../demo/python/03_landcover_mapping.ipynb)