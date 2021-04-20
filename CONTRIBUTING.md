# Contributing

## Design principles and intended scope

The `eumap` software aims to provide:
  1. ergonomically designed abstractions around [pan-european datasets](https://maps.opendatascience.eu) produced within the scope of the [GeoHarmonizer project](https://opendatascience.eu/geoharmonizer-project/) and beyond, to catalyze large-scale future research
  2. a high-level, user-friendly and generalized utility package for geodata analytics, with functionality that is felt to be either entirely missing from the present ecosystem, or can be improved upon both in terms of ease of use and easy scalability to large datasets (e.g. performing machine learning tasks against terabyte-scale geodata)

All contributions to this software should fall within the scope described above, unless discussed beforehand by the core development team.

## Issue reporting and suggestions

For reporting issues and making feature suggestions please refer to the [issue tracker](https://gitlab.com/geoharmonizer_inea/eumap/-/issues).

## Contributing code and documentation

### Initial setup

1. Clone the repo: `git clone https://geoharmonizer_inea/eumap`
2. Install dependencies (not needed if only contributing documentation):
  - for the Python package: `pip install -r requirements.txt` into a Python 3.6+ environment
  <!-- needs verification -->
  - for the R package: TBD
  <!-- needs R instructions -->

### Development

All changes to code and documentation should be made in a separate branch, created from an up-to-date local `master`, e.g.:

```
git checkout master
git pull
git branch new_feature
git checkout new_feature
```

When the changes are complete a merge request may be submitted from the development branch (if you have submitted a merge request with incomplete changes, please indicate that the branch is not to be merged yet in the title of the request).

If you do not have the appropriate permissions to submit new branches to the `eumap` repository, you may fork this repository into your own GitLab namespace and submit merge requests from there.

### Code conventions

We strongly prefer to submit code to `pyeumap` with [type hints](https://docs.python.org/3/library/typing.html). Additionally, we support Python versions as low as 3.6 and no code that uses syntax introduced in later versions of Python (e.g. the walrus operator) will be accepted.
<!-- needs verification -->
<!-- needs R instructions -->

There are currently no style restrictions guidelines imposed upon code contributions. This may change at a later date.

### Versioning

We adthere to standard [semantic versioning](https://semver.org/). Since we release from `master` <!-- needs to be discussed -->
all merge requests should be accompanied with a version increment and the responsibility for increasing the version number falls on the contributor merging a branch: when merging a request either increment the MINOR version and reset the PATCH version to zero (if the intent of the merge request is to add new features) or increment the PATCH version (if the merge request only contains bugfixes). When merging a branch made by another contributor (e.g. because they do not have the required permissions to do so) please confirm the intent of the merge request (i.e. which semver number needs to be incremented).

The Python and R packages are to be versioned separately. When incrementing the version of `pyeumap` it is enough to write the version change into [`__init__.py`](https://gitlab.com/geoharmonizer_inea/eumap/-/blob/master/pyeumap/__init__.py) in the appropriate branch.
<!-- needs R instructions -->
