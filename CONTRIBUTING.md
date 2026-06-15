# Contributing

## Design principles and intended scope

The `scikit-map` aims to provide a high-level, user-friendly and generalized utility package for geodata analytics, with functionality that is felt to be either entirely missing from the present ecosystem, or can be improved upon both in terms of ease of use and easy scalability to large datasets (e.g. performing machine learning tasks against terabyte-scale geodata)

All contributions to this software should fall within the scope described above, unless discussed beforehand by the core development team.

## Issue reporting and suggestions

For reporting issues and making feature suggestions please refer to the [issue tracker](https://github.com/openlandmap/scikit-map/issues) using the existing templates:

## Contributing code and documentation

### Initial setup

1. Clone the repo: `git clone git@github.com:openlandmap/scikit-map.git`
2. Install dependencies (not needed if only contributing documentation):
   - Documentation: `uv pip install .[full,dev,docs] sphinx-autobuild`
   - Code: `uv` doesn't like editable installs with our setup, after the above, re-run: `pip install -e .[full,dev,docs]` into a Python 3.6+ environment

This will ensure that a change to python files triggers a rebuild, as well as running `doxygen` again.

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

`scikit-map` is a mixed Python/C++ project. It auto-builds documentation from Python and C++ docstrings.

1. For Python documentation live-updates run
   ```bash
   sphinx-autobuild docs/ _build/ --watch xml/
   ```
2. For C++: in the top-level directory, run `doxygen` to create xml documentation for `skmap_bindings`
3. If you change C++ documentation, in a separate terminal run `doxygen` again and it will auto-update.

### Python code


#### Code conventions

We strongly prefer to submit code to `scikit-map` with [type hints](https://docs.python.org/3/library/typing.html). Additionally, we support Python versions as low as 3.8 and no code that uses syntax introduced in later versions of Python (e.g. the walrus operator) will be accepted.

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
