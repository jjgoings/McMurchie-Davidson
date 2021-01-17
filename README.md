[![Build Status](https://travis-ci.org/jjgoings/McMurchie-Davidson.svg?branch=master)](https://travis-ci.org/jjgoings/McMurchie-Davidson) 
[![codecov](https://codecov.io/gh/jjgoings/McMurchie-Davidson/branch/master/graph/badge.svg)](https://codecov.io/gh/jjgoings/McMurchie-Davidson)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/jjgoings/McMurchie-Davidson.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jjgoings/McMurchie-Davidson/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jjgoings/McMurchie-Davidson.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jjgoings/McMurchie-Davidson/context:python)

# McMurchie-Davidson

This contains some simple routines to compute one and two electron integrals 
necessary for Hartree Fock calculations using the McMurchie-Davidson algorithm.
The Hartree-Fock code can only  handle closed shell molecules at the moment. 
It's not fast (though getting faster with Cython interface), but should be 
somewhat readable. 

I'm slowly porting the integrals over to Cython and reoptimizing. I'm also 
thinking about reorganizing so as to make it easy to add functionality in the 
future.

## Installation
Installation should be simple, just

```
python setup.py build_ext --inplace
```

### Dependencies
You'll need `numpy`, `scipy`, and `cython` (for the integrals). The install script should yell at you if you don't have the requisite dependencies. You can install them all at once if you have `pip`:

```
pip install numpy scipy cython
```

### Testing
You can test the install with `nosetests`. In the head directory, just do

```
nosetests tests
```

it should take a few seconds, but everything should pass.

## Running
Once you've installed, you can try running the input script `sample-input.py`:

```
python sample-input.py
```

which should do an SCF on water with an STO-3G basis and dump out to your terminal:

```
E(SCF)    =  -74.942079928060 in 10 iterations
  Convergence:
    FPS-SPF  =  3.377372360032819e-13
    RMS(P)   =  2.35e-12
    dE(SCF)  =  -9.95e-10
  Dipole X =  0.00000000
  Dipole Y =  1.53400931
  Dipole Z =  -0.00000000
E(MP2) =  -74.99122956422062
```

## Input file specification

The input is fairly straightforward. Here is a minimal example using water.

```
from mmd.molecule import *
from mmd.scf import *

water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

# init molecule and build integrals
mol = Molecule(geometry=water,basis='sto-3g')

# do the SCF
mol.RHF()
```

The first lines import the `molecule` and `scf` modules, which we need to specify our molecule and the SCF routines. The molecular geometry follows afterward and is specified by the stuff in triple quotes. The first line is charge and multiplicity, followed by each atom and its Cartesian coordinates (in Angstrom).

```
water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""
```

Then we generate create the molecule (`Molecule` object) and build the integrals, and finish by running the SCF.

At any point you can inspect the molecule. For example, you can dump out the (full) integral arrays:

```
print(mol.S)
```

which dumps out the overlap matrix:

```
[[ 1.     0.237  0.     0.     0.     0.038  0.038]
 [ 0.237  1.     0.     0.     0.     0.386  0.386]
 [ 0.     0.     1.     0.     0.     0.268 -0.268]
 [ 0.     0.     0.     1.     0.     0.21   0.21 ]
 [ 0.     0.     0.     0.     1.     0.     0.   ]
 [ 0.038  0.386  0.268  0.21   0.     1.     0.182]
 [ 0.038  0.386 -0.268  0.21   0.     0.182  1.   ]]
```

There is also some limited post-SCF functionality, hopefully more useful as a 
template for adding later post-SCF methods.

```
# do MP2
mp2 = PostSCF(mol)
mp2.MP2()
```

## Examples
In the `examples` folder you can find some different scripts for setting up more complex jobs. For example, there is a simple script that does Born-Oppenheimer molecular dynamics on minimal basis hydrogen, aptly titled `bomd.py`. There is also some real-time code that allows you to craft an arbitrary pulse sequence, in the example `pulses.py`. Feel free to try them out.
