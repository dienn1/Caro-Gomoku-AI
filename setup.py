from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("caro.pyx", compiler_directives={'boundscheck': False})
)