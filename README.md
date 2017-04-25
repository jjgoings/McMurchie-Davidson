# McMurchie-Davidson

This contains some simple routines to compute one and two electron integrals 
necessary for Hartree Fock calculations using the McMurchie-Davidson algorithm.
Some of the code (esp. basis function classes and basis set definitions)
 is borrowed heavily from PyQuante2, but the integral evaluation routines over 
primitives, as well as the SCF code, are my own. The Hartree-Fock code can only 
handle closed shell molecules at the moment. It's not fast (though getting 
faster with Cython interface), but should be somewhat readable. 

I'm slowly porting the integrals over to Cython and reoptimizing. I'm also 
thinking about reorganizing so as to make it easy to add functionality in the 
future.

If you clone (and are still in) the directory, you can try the SCF like so:

```
from mmd.molecule import *
from mmd.scf import *
from mmd.postscf import *

# read in geometry
geometry = './geoms/h2o.dat'

# init molecule and build integrals
mol = Molecule(filename=geometry,basis='sto-3g')
mol.build()

# do the SCF
scf = SCF(mol)
scf.RHF()
```

You can also dump out the (full) integral arrays:

```
print h2o.S
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

