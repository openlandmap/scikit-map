import setuptools
from skmap import __version__
from pathlib import Path

root_dir = Path(__file__).parent

with open(root_dir.joinpath('skmap', 'README.md'), 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='scikit-map',
    version=__version__,
    description='scikit-learn applied to mapping and spatial prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/scikit-map/scikit-map',
    packages=setuptools.find_packages(),
    package_data={},
    scripts=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'GDAL>=3.1',
        'affine>=2.3',
        'geopandas>=0.8',
        'joblib>=1.1.0',
        'numpy>=1.19',
        'pyproj>=3.1',
        'OWSLib==0.22',
        'pandas>=1.1',
        'requests>=2.24',
        'scikit_learn>=0.24',
        'rasterio>=1.1',
        'psutil>=5.8'
    ],
    extras_require={
        'full': [
            'Bottleneck>=1.3',
            'gspread>=5.3.2',
            'matplotlib>=3.3',
            'minio>=7.1.0',
            'opencv_python>=4.5',
            'pqdm>=0.1',
            'pygeos>=0.8',
            'pystac>=1.4.0',
            'pyts>=0.11',
            'pytz>=0.11',
            'scikit_image>=0.17',
            'Shapely>=1.7',
            'sentinelhub==3.4.4',
            'whoosh>=2.7.4'
        ],
    },
)
