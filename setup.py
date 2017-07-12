from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

my_integrals = Extension('mmd.integrals',['mmd/integrals.pyx'])
#                 extra_compile_args=['-fopenmp'],
#                 extra_link_args=['-fopenmp'])

setup(
    name='mmd',
    version='0.1dev0',
    packages=['mmd'],
    package_data = {'mmd' : ['mmd/basis']},
    license='BSD-3',
    install_requires=[
          'cython',
          'numpy',
          'scipy',
    ],
    long_description=open('README.md').read(),
    ext_modules=cythonize([my_integrals]),
    include_dirs=[numpy.get_include()],
    include_package_data = True,
)


