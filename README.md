# McMurchie-Davidson

This contains some simple routines to compute one and two electron integrals 
necessary for Hartree Fock calculations using the McMurchie-Davidson algorithm.
Some of the code (esp. basis function classes) is borrowed heavily from 
PyQuante2, but the integral evaluation routines over primitives are my own. The
Hartree-Fock code can only handle closed shell molecules at the moment.
