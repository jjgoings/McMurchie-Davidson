from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
#from Cython.Distutils import build_ext
import numpy
import os

os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + numpy.get_include() 
#os.environ["CC"] = "gcc-6" 
#os.environ["CXX"] = "gcc-6"

my_hermite = Extension('mmd.hermite',['hermite.pyx'])
#                 extra_compile_args=['-fopenmp'],
#                 extra_link_args=['-fopenmp'])

setup(
   #ext_modules=cythonize("hermite.pyx"),
    ext_modules=cythonize(["numeri.pyx",
                          "magnus.pyx",
                           my_hermite]),
    include_dirs=[numpy.get_include()],
)


