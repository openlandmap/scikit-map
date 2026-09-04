# Contributing

## Design principles and intended scope

The `scikit-map` aims to provide a high-level, user-friendly and generalized utility package for geodata analytics, with functionality that is felt to be either entirely missing from the present ecosystem, or can be improved upon both in terms of ease of use and easy scalability to large datasets (e.g. performing machine learning tasks against terabyte-scale geodata)

All contributions to this software should fall within the scope described above, unless discussed beforehand by the core development team.

## Issue reporting and suggestions

For reporting issues and making feature suggestions please refer to the [issue tracker](https://github.com/openlandmap/scikit-map/issues) using the existing templates:

## Contributing code and documentation

### Initial setup

**Prerequisites (system packages):** `gdal-bin`, `libgdal-dev`, `libproj-dev`, `libgeos-dev`, `doxygen`, `pandoc`, `build-essential`, `cmake`, and `uv` ([install uv](https://docs.astral.sh/uv/getting-started/installation/)).

1. Clone the repo:
   ```bash
   git clone git@github.com:openlandmap/scikit-map.git
   cd scikit-map
   ```
2. Create and activate a `uv` virtual environment (Python 3.10+):
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```
3. Install the package (editable) together with all dev, docs, and full extras:
   ```bash
   uv pip install -e ".[full,dev,docs]"
   ```
   This builds the C++ bindings (`skmap_bindings`) via CMake — the build fetches Eigen and pybind11 automatically, so network access is required.

Verify the install:
   ```bash
   python -c "import skmap, skmap_bindings; print('ok')"
   ```

### Building the documentation

The docs use Sphinx with Breathe (C++ Doxygen autodoc) and nbsphinx (Jupyter notebooks).

1. Regenerate the C++ XML that Breathe consumes:
   ```bash
   doxygen
   ```
2. Build the HTML:
   ```bash
   sphinx-build docs/ _build/
   ```
   Open `_build/index.html` to view.

> **Don't use `make -C docs html`** — the Makefile's `jupytext --to notebook notebooks/*.py` step fails when there are no `.py` notebooks. Use `sphinx-build docs/ _build/` directly.

For live-updates during development:
   ```bash
   sphinx-autobuild docs/ _build/ --watch xml/
   ```
   (Run `doxygen` in a separate terminal if you change C++ docstrings.)

### Development: Git

All changes to code and documentation should be made in a separate branch, created from an up-to-date local `main`. The **branch name** must refer a open issue (``i{ISSUE_ID}``):

```
git checkout main
git pull
git checkout -b i0

git add [CHANGED FILES]
git commit -m "closes #0; [GENERAL COMMENT]"

git checkout main
git pull
git merge i0
git push

git branch -d i0
```

When the changes are complete a merge request may be submitted from the development branch (if you have submitted a merge request with incomplete changes, please indicate that the branch is not to be merged yet in the title of the request).

If you do not have the appropriate permissions to submit new branches to the `scikit-map` repository, you may fork this repository into your own Github namespace and submit merge requests from there.

### Commit conventions

All commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

``` 

**Type**

Must be one of the following:

- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **test**: Adding missing tests or correcting existing tests

Based on [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

### Documentation

`scikit-map` is a mixed Python/C++ project. It auto-builds documentation from Python and C++ docstrings. See [Building the documentation](#building-the-documentation) above for the build steps.

### Python code


#### Code conventions

We strongly prefer to submit code to `scikit-map` with [type hints](https://docs.python.org/3/library/typing.html). Additionally, we support Python 3.10+ (`requires-python = ">=3.10"` in `pyproject.toml`).

Python code is formatted using ruff:

```bash
ruff format
```

There are currently no style restrictions guidelines imposed upon code contributions. This may change at a later date.

### C++ code

C++ code is formatted with clang-format:

```bash
# specify directories to omit build directory
find {skmap/,tests/} -iname "*.cpp" -o -iname "*.h" | xargs clang-format --verbose -i
```
#### Clangd language server

For code completion and intellisense. It needs a `compile_commands.json` at top-level. generate with:

```bash
cmake -B build -DTESTS=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .
mv build/compile_commands.json .
```
#### Unit tests

Unit tests with [`gtest`](https://google.github.io/googletest/quickstart-cmake.html) are included.

To run:

```bash
cmake -B build -DTESTS=1 . # only have to run once
cmake --build build -j && ./build/tests/src/unit_tests 
```

### Versioning

We adhere to standard [semantic versioning](https://semver.org/). Since we release from `main` <!-- needs to be discussed -->
all merge requests should be accompanied with a version increment and the responsibility for increasing the version number falls on the contributor merging a branch: when merging a request either increment the MINOR version and reset the PATCH version to zero (if the intent of the merge request is to add new features) or increment the PATCH version (if the merge request only contains bugfixes). When merging a branch made by another contributor (e.g. because they do not have the required permissions to do so) please confirm the intent of the merge request (i.e. which semver number needs to be incremented).

When incrementing the version of `scikit-map` it is enough to write the version change into [`pyproject.toml`](./pyproject.toml) in the appropriate branch.
