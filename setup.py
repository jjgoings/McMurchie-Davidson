from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CPPFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + numpy.get_include() 

my_integrals = [Extension('mmd.integrals.onee',['cython/onee.pyx']),
                Extension('mmd.integrals.twoe',['cython/twoe.pyx']),
                Extension('mmd.integrals.grad',['cython/grad.pyx']),
                Extension('mmd.integrals.fock',['cython/fock.pyx']),
#                 extra_compile_args=['-fopenmp'],
#                 extra_link_args=['-fopenmp'])
                ]

setup(
    name='mmd',
    version='0.0.1',
    packages=find_packages(),
    package_data = {'mmd' : ['mmd/basis']},
    license='BSD-3',
    python_requires=">=3.4",
    install_requires=[
          'cython',
          'numpy',
          'scipy',
          'bitstring',
    ],
    long_description=open('README.md').read(),
#   linetrace directive for cython profiling
    ext_modules=cythonize(my_integrals,
                          compiler_directives={'linetrace': True, 'language_level' : '3'}),
#    ext_modules=cythonize(my_integrals),
    include_dirs=[numpy.get_include()],
    include_package_data = True,
)


