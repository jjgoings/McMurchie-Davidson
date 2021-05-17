from __future__ import division
import sys
import os
import numpy as np
from mmd.integrals.onee import S,T,Mu,V,RxDel
from mmd.integrals.twoe import doERIs, ERI
from scipy.linalg import fractional_matrix_power as mat_pow
from mmd.scf import SCF
from mmd.forces import Forces
from mmd.integrals.twoe import Basis
import itertools

class Atom(object):
    """Class for an atom"""
    def __init__(self,charge,mass,origin=np.zeros(3)):
        self.charge = charge
        self.origin = origin
        self.mass   = mass
        # contains forces (not mass-weighted)
        self.forces      = np.zeros(3)
        self.saved_forces  = np.zeros(3)
        self.velocities  = np.zeros(3)

class Molecule(SCF,Forces):
    """Class for a molecule object, consisting of Atom objects
       Requres that molecular geometry, charge, and multiplicity be given as
       input on creation.
    """
    def __init__(self,geometry,basis='sto-3g'):
        # geometry is now specified in imput file
        charge, multiplicity, atomlist = self.read_molecule(geometry)
        self.charge = charge
        self.multiplicity = multiplicity
        self.atoms = atomlist
        self.nelec = sum([atom.charge for atom in atomlist]) - charge 
        self.nocc  = self.nelec//2
        self.is_built = False
        
        # Read in basis set data
        import os
        cur_dir = os.path.dirname(__file__)
        basis_path = 'basis/'+str(basis).lower()+'.gbs'
        basis_file = os.path.join(cur_dir, basis_path)
        self.basis_data = self.getBasis(basis_file)
        self.formBasis()

    @property
    def _forces(self):
        # FIXME: assumes forces have been computed!
        F = []
        for atom in range(len(self.atoms)):
            F.append(self.atoms[atom].forces)
        return np.concatenate(F).reshape(-1,3)

         

    def formBasis(self):
        """Routine to create the basis from the input molecular geometry and
           basis set. On exit, you should have a basis in self.bfs, which is a 
           list of BasisFunction objects. This routine also defines the center
           of nuclear charge. 
        """
        self.bfs = []
        for atom in self.atoms:
            for momentum,prims in self.basis_data[atom.charge]:
                exps = [e for e,c in prims]
                coefs = [c for e,c in prims]
                for shell in self.momentum2shell(momentum):
                    #self.bfs.append(BasisFunction(np.asarray(atom.origin),\
                    #    np.asarray(shell),np.asarray(exps),np.asarray(coefs)))
                    self.bfs.append(Basis(np.asarray(atom.origin),
                        np.asarray(shell),len(exps),np.asarray(exps),np.asarray(coefs)))
        self.nbasis = len(self.bfs)
        # create masking vector for geometric derivatives
        idx = 0
        for atom in self.atoms:
            atom.mask = np.zeros(self.nbasis)
            for momentum,prims in self.basis_data[atom.charge]:
                for shell in self.momentum2shell(momentum):
                    atom.mask[idx] = 1.0
                    idx += 1

        # note this is center of positive charge (atoms only, no electrons)
        self.center_of_charge =\
            np.asarray([sum([atom.charge*atom.origin[0] for atom in self.atoms]),
                        sum([atom.charge*atom.origin[1] for atom in self.atoms]),
                        sum([atom.charge*atom.origin[2] for atom in self.atoms])])\
                        * (1./sum([atom.charge for atom in self.atoms]))

    def build(self,direct=False):
        """Routine to build necessary integrals"""
        self.one_electron_integrals()
        if direct:
            # populate dict for screening
            self.screen = {}
            for p in range(self.nbasis):
                for q in range(p + 1):
                    pq = p*(p+1)//2 + q
                    self.screen[pq] = ERI(self.bfs[p],self.bfs[q],self.bfs[p],self.bfs[q])
        else:
            self.two_electron_integrals()
        self.is_built = True

    def momentum2shell(self,momentum):
        """Routine to convert angular momentum to Cartesian shell pair in order
           to create the appropriate BasisFunction object (e.g. form px,py,pz)
        """
        shells = {
            'S' : [(0,0,0)],
            'P' : [(1,0,0),(0,1,0),(0,0,1)],
            'D' : [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
            'F' : [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
                   (0,3,0),(0,2,1),(0,1,2), (0,0,3)]
            }
        return shells[str(momentum)]
        
    def sym2num(self,sym):
        """Routine that converts atomic symbol to atomic number"""
        symbol = [
            "X","H","He",
            "Li","Be","B","C","N","O","F","Ne",
            "Na","Mg","Al","Si","P","S","Cl","Ar",
            "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
            "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
            "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Sb", "Te", "I", "Xe",
            "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",  "Eu",
            "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl","Pb","Bi","Po","At","Rn"]
        return symbol.index(str(sym))

    def getBasis(self,filename):
        """Routine to read the basis set files (EMSL Gaussian 94 standard)
           The file is first split into atoms, then iterated through (once).
           At the end we get a basis, which is a dictionary of atoms and their
           basis functions: a tuple of angular momentum and the primitives

           Return: {atom: [('angmom',[(exp,coef),...]), ('angmom',[(exp,...} 
        """
        basis = {}
    
        with open(filename, 'r') as basisset:
            data = basisset.read().split('****')
   
        # Iterate through all atoms in basis set file 
        for i in range(1,len(data)):
            atomData = [x.split() for x in data[i].split('\n')[1:-1]]
            for idx,line in enumerate(atomData):
                # Ignore empty lines
                if not line:
                   pass
                # first line gives atom
                elif idx == 0:
                    assert len(line) == 2
                    atom = self.sym2num(line[0])
                    basis[atom] = []
                    # now set up primitives for particular angular momentum
                    newPrim = True
                # Perform the set up once per angular momentum 
                elif idx > 0 and newPrim:
                    momentum  = line[0]
                    numPrims  = int(line[1])
                    newPrim   = False
                    count     = 0
                    prims     = []
                    prims2    = [] # need second list for 'SP' case
                else:
                   # Combine primitives with its angular momentum, add to basis
                   if momentum == 'SP':
                       # Many basis sets share exponents for S and P basis 
                       # functions so unfortunately we have to account for this.
                       prims.append((float(line[0].replace('D', 'E')),float(line[1].replace('D', 'E'))))
                       prims2.append((float(line[0].replace('D', 'E')),float(line[2].replace('D', 'E'))))
                       count += 1
                       if count == numPrims:
                           basis[atom].append(('S',prims))
                           basis[atom].append(('P',prims2))
                           newPrim = True
                   else:
                       prims.append((float(line[0].replace('D', 'E')),float(line[1].replace('D', 'E'))))
                       count += 1
                       if count == numPrims:
                           basis[atom].append((momentum,prims))
                           newPrim = True
    
        return basis
        
    def read_molecule(self,geometry):
        """Routine to read in the charge, multiplicity, and geometry from the 
           input script. Coordinates are assumed to be Angstrom.
           Example:

           geometry = '''
                      0 1
                      H  0.0 0.0 1.2
                      H  0.0 0.0 0.0
                      '''
           self.read_molecule(geometry)

        """
        # atomic masses (isotop avg)
        masses = [0.0,1.008,4.003,6.941,9.012,10.812,12.011,14.007,5.999,
                  18.998,20.180,22.990,24.305,26.982,28.086,30.974,32.066,
                  35.453,39.948]
        f = geometry.split('\n')
        # remove any empty lines
        f = filter(None,f) 
        # First line is charge and multiplicity
        atomlist = []
        for line_number,line in enumerate(f):
            if line_number == 0:
                assert len(line.split()) == 2
                charge = int(line.split()[0])
                multiplicity = int(line.split()[1])
            else: 
                if len(line.split()) == 0: break
                assert len(line.split()) == 4
                sym = self.sym2num(str(line.split()[0]))
                mass = masses[sym]
                # Convert Angstrom to Bohr (au)
                x   = float(line.split()[1])/0.52917721092
                y   = float(line.split()[2])/0.52917721092
                z   = float(line.split()[3])/0.52917721092
                # Convert amu to atomic units
                mass *= 1822.8885
                atom = Atom(charge=sym,mass=mass,
                            origin=np.asarray([x,y,z]))
                atomlist.append(atom)
    
        return charge, multiplicity, atomlist

    def one_electron_integrals(self):
        """Routine to set up and compute one-electron integrals"""
        N = self.nbasis
        # core integrals
        self.S = np.zeros((N,N)) 
        self.V = np.zeros((N,N)) 
        self.T = np.zeros((N,N)) 
        # dipole integrals
        self.M = np.zeros((3,N,N)) 
        self.mu = np.zeros(3,dtype='complex') 
 
        # angular momentum
        self.L = np.zeros((3,N,N)) 

        self.nuc_energy = 0.0
        # Get one electron integrals
        #print "One-electron integrals"

        for i in (range(N)):
            for j in range(i+1):
                self.S[i,j] = self.S[j,i] \
                    = S(self.bfs[i],self.bfs[j])
                self.T[i,j] = self.T[j,i] \
                    = T(self.bfs[i],self.bfs[j])
                self.M[0,i,j] = self.M[0,j,i] \
                    = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'x')
                self.M[1,i,j] = self.M[1,j,i] \
                    = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'y')
                self.M[2,i,j] = self.M[2,j,i] \
                    = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'z')
                for atom in self.atoms:
                    self.V[i,j] += -atom.charge*V(self.bfs[i],self.bfs[j],atom.origin)
                self.V[j,i] = self.V[i,j]

                # RxDel is antisymmetric
                self.L[0,i,j] \
                    = RxDel(self.bfs[i],self.bfs[j],self.center_of_charge,'x')
                self.L[1,i,j] \
                    = RxDel(self.bfs[i],self.bfs[j],self.center_of_charge,'y')
                self.L[2,i,j] \
                    = RxDel(self.bfs[i],self.bfs[j],self.center_of_charge,'z')
                self.L[:,j,i] = -1*self.L[:,i,j] 

        # Compute nuclear repulsion energy 
        for pair in itertools.combinations(self.atoms,2):
            self.nuc_energy += pair[0].charge*pair[1].charge \
                              / np.linalg.norm(pair[0].origin - pair[1].origin)
           
        # Preparing for SCF
        self.Core       = self.T + self.V
        self.X          = mat_pow(self.S,-0.5)
        self.U          = mat_pow(self.S,0.5)

    def two_electron_integrals(self):
        """Routine to setup and compute two-electron integrals"""
        N = self.nbasis
        self.TwoE = np.zeros((N,N,N,N))  
        self.TwoE = doERIs(N,self.TwoE,self.bfs)
        self.TwoE = np.asarray(self.TwoE)

    def save_integrals(self,folder=None):
        """Routine to save integrals for SCF in Crawford group format"""
        if folder is None:
            sys.exit('Please provide a folder to save the integrals.')
        else:
            if not self.is_built:
                self.build()
            os.makedirs(folder,exist_ok=True) # careful! will overwrite. 
            
            np.savetxt(folder + '/enuc.dat',self.nuc_energy.reshape(1,))
            np.savetxt(folder + '/nbf.dat',np.asarray(self.nbasis,dtype=int).reshape(1,),fmt='%d')
            np.savetxt(folder + '/nelec.dat',np.asarray(self.nelec,dtype=int).reshape(1,),fmt='%d')
            np.savetxt(folder + '/s.dat',self.S)
            np.savetxt(folder + '/t.dat',self.T)
            np.savetxt(folder + '/v.dat',self.V)
            with open(folder + '/eri.dat','w') as f:
                for i,j,k,l in itertools.product(range(self.nbasis),range(self.nbasis),range(self.nbasis),range(self.nbasis)):
                    print(i+1,j+1,k+1,l+1,self.TwoE[i,j,k,l],file=f)
       
            
            

        
            


