import os
import subprocess
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension


module_skmap_bindings = setuptools.Extension('skmap_bindings',
                                             sources=['skmap_bindings/src/skmap_bindings.cpp',
                                                      'skmap_bindings/src/ParArray.cpp',
                                                      'skmap_bindings/src/io/IoArray.cpp',
                                                      'skmap_bindings/src/transform/TransArray.cpp'],
                                             include_dirs=[np.get_include(), 'pybind11/include', 'skmap_bindings', 'skmap_bindings/include', 'skmap_bindings/src'],
                                             extra_compile_args=['-fopenmp', '-std=c++17', '-std=gnu++17'],
                                             extra_link_args=['-lgomp'],
                                             libraries=['fftw3_threads', 'fftw3', 'm', 'gomp', 'gdal'])

# ext_modules = [
#     Pybind11Extension(
#         'skmap_bindings',
#         ['skmap_bindings/skmap_bindings.cpp'],
#         include_dirs=[pybind11.get_include()],
# library_dirs = [ '/home/dconsoli/Documents/skmap-devel/skmap_bindings/cmake-build-debug'],
# libraries = ['skmap_bindings'],
# extra_compile_args = ['-std=c++17'],
# ),
# ]


setuptools.setup(
    name='scikit-map',
    version='0.8.1',
    description='scikit-learn applied to mapping and spatial prediction',
    long_description="Python module to produce maps using machine learning, reference samples and raster data.",
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
        'geopandas>=0.13',
        'joblib>=1.1.0',
        'numpy>=1.19',
        'pyproj>=3.1',
        'pandas>=2.0',
        'requests>=2.24',
        'scikit-learn>=1.3',
        'rasterio>=1.1'
    ],
    extras_require={
        'full': [
            'Bottleneck>=1.3',
            'gspread>=5.3.2',
            'matplotlib>=3.7.3',
            'minio>=7.1.0',
            'pqdm>=0.1',
            'pystac>=1.4.0',
            'pyts>=0.11',
            'pyfftw>=0.13',
            'scikit_image>=0.20',
            'Shapely>=1.7',
            'whoosh>=2.7.4'
        ],
    },
    ext_modules=[module_skmap_bindings],
    # ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
