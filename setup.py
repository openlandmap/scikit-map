import setuptools
import numpy as np
from pathlib import Path
from setuptools.command.build_ext import build_ext

module_skmap_bindings = setuptools.Extension('skmap_bindings',
                                       sources=['skmap/src/skmap_bindings.cpp',
                                                'skmap/src/ParArray.cpp',
                                                'skmap/src/io/IoArray.cpp',
                                                'skmap/src/transform/TransArray.cpp'],
                                       include_dirs=[np.get_include(), 'pybind11/include', 'skmap/include', 'skmap/src'],
                                       extra_compile_args=['-fopenmp'],
                                       extra_link_args=['-lgomp', '-std=c++11'],
                                       libraries=['fftw3_threads', 'fftw3', 'm', 'gomp', 'gdal'])

setuptools.setup(
    name='scikit-map',
    version='0.7.3',
    description='scikit-learn applied to mapping and spatial prediction',
    long_description="Python module to produce maps using machine learning, reference samples and raster data.",
    long_description_content_type='text/markdown',
    url='https://github.com/scikit-map/scikit-map',
    packages=setuptools.find_namespace_packages(),
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
    ext_modules = [module_skmap_bindings],
    cmdclass={'build_ext': build_ext},
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
)


