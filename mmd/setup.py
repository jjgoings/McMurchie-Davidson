from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + numpy.get_include() 


setup(
    #ext_modules=cythonize("hermite.pyx"),
    ext_modules=cythonize("*.pyx"),
    include_dirs=[numpy.get_include()],
)

