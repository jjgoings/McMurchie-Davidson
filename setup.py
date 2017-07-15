from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + numpy.get_include() 

my_integrals = [Extension('mmd.integrals.onee',['cython/onee.pyx']),
                Extension('mmd.integrals.twoe',['cython/twoe.pyx']),
                Extension('mmd.integrals.grad',['cython/grad.pyx']),
#                 extra_compile_args=['-fopenmp'],
#                 extra_link_args=['-fopenmp'])
                ]

setup(
    name='mmd',
    version='0.0.1',
    packages=['mmd'],
    package_data = {'mmd' : ['mmd/basis']},
    license='BSD-3',
    install_requires=[
          'cython',
          'numpy',
          'scipy',
    ],
    long_description=open('README.md').read(),
    ext_modules=cythonize(my_integrals),
    include_dirs=[numpy.get_include()],
    include_package_data = True,
)


