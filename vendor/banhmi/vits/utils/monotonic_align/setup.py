"""Build the `core` Cython extension in place:
    python setup.py build_ext --inplace
Requires Cython and numpy (both already in requirements.txt) and a C
compiler (MSVC Build Tools on Windows).
"""
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

_DIR = Path(__file__).parent

# Extension name is the unqualified "core" (not a dotted package path) so
# `build_ext --inplace` always drops the built .pyd/.so next to core.pyx,
# regardless of where banhmi_train/vits/monotonic_align sits in the
# package tree or what directory this script is invoked from.
_extension = Extension(name="core", sources=[str(_DIR / "core.pyx")])

setup(
    name="monotonic_align",
    ext_modules=cythonize([_extension], language_level=3),
    include_dirs=[numpy.get_include()],
)
