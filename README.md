# McMurchie-Davidson

This contains some simple routines to compute one and two electron integrals 
necessary for Hartree Fock calculations using the McMurchie-Davidson algorithm.
Some of the code (esp. basis function classes and basis set definitions)
 is borrowed heavily from PyQuante2, but the integral evaluation routines over 
primitives, as well as the SCF code are my own. The Hartree-Fock code can only 
handle closed shell molecules at the moment.

If you close the directory, you can try the SCF like so:
```
from molecule import *
h2o = Molecule('h2o.dat',basis='sto-3g')
h2o.SCF()
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




