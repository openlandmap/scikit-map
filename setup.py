import setuptools
from pyeumap import __version__
from pathlib import Path

root_dir = Path(__file__).parent

with open(root_dir.joinpath('pyeumap', 'README.md'), 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyeumap',
    version=__version__,
    description='eumap Python package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/geoharmonizer_inea/eumap',
    packages=setuptools.find_packages(),
    package_data={},
    scripts=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
