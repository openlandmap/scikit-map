import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Release" if not self.debug else "Debug"
        build_args = ["--config", cfg]

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        # Propagate the detected system GDAL version so CMake can verify that
        # the C++ libgdal it links against matches the Python GDAL/rasterio
        # wheels (an ABI mismatch here causes hard-to-debug runtime crashes).
        if _sys_gdal_version:
            cmake_args.append(f"-DSKMAP_EXPECTED_GDAL_VERSION={_sys_gdal_version}")

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


install_requires = [
    "affine>=2.4.0",
    "geopandas>=0.13.2",
    "joblib>=1.3.2",
    "ray>=2.9.0",
    "numpy>=1.24.3",
    "pandas>=2.0.2",
    "requests>=2.31.0",
    "scikit-learn>=1.3.2",
    "rasterio>=1.3.6",
    "cmake>=3.15",
    "minio>=7.1.5",
    "gspread>=5.3.2",
    "tomli>=2.0.1",
    "PyYAML>=6.0",
]


# Detect system libgdal version (via gdal-config) and build the
# install_requires list accordingly. If detection fails, fall back to a
# permissive GDAL requirement.
def _detect_system_gdal_version():
    if shutil.which("gdal-config"):
        try:
            # text=True requires Python 3.7+; keep compatibility by decoding
            out = subprocess.check_output(["gdal-config", "--version"]).decode()
            return out.strip()
        except Exception:
            return None
    return None


_sys_gdal_version = _detect_system_gdal_version()


if _sys_gdal_version:
    # Pin GDAL to the system library version to avoid mismatches
    install_requires.append(f"GDAL=={_sys_gdal_version}")
else:
    raise RuntimeError("GDAL lib not found in the system")

setup(
    name="scikit-map",
    version="0.9.1",
    packages=["skmap"],
    ext_modules=[CMakeExtension("skmap_bindings", ".")],
    cmdclass={"build_ext": CMakeBuild},
    data_files=[("", ["skmap_bindings.pyi"])],
    zip_safe=False,
    install_requires=install_requires,
)
