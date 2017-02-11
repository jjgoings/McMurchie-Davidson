import numpy as np
from integrals import *
from scipy.misc import factorial2 as fact2
from scipy.linalg import fractional_matrix_power as mat_pow
import itertools
from tqdm import tqdm, trange 

class BasisFunction(object):
    def __init__(self,origin=(0,0,0),shell=(0,0,0),exps=[],coefs=[]):
        assert len(origin)==3
        assert len(shell)==3
        self.origin = np.asarray(origin,'d')
        self.shell = shell
        self.exps  = exps
        self.coefs = coefs
        self.normalize()

    def normalize(self):
        l,m,n = self.shell
        # self.norm is a list of length number primitives
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                        np.power(self.exps,l+m+n+1.5)/
                        fact2(2*l-1)/fact2(2*m-1)/
                        fact2(2*n-1)/np.power(np.pi,1.5))
        return

class Molecule(object):
    def __init__(self,filename,basis='sto3g'):
        charge, multiplicity, atomlist = self.read_molecule(filename)
        self.charge = charge
        self.multiplicity = multiplicity
        self.atoms = atomlist
        self.nelec = sum([atom[0] for atom in atomlist]) - charge 
        self.nocc  = self.nelec//2
        self.bfs = []
        try:
            import data
        except ImportError:
            print "No basis set data"
            sys.exit(0)

        basis_data = data.basis[basis]
        for atom in self.atoms:
            for momentum,prims in basis_data[atom[0]]:
                exps = [e for e,c in prims]
                coefs = [c for e,c in prims]
                for shell in self.momentum2shell(momentum):
                    #self.bfs.append(BasisFunction(atom[1],shell,exps,coefs))
                    self.bfs.append(BasisFunction(atom[1],shell,exps,coefs))
        self.nbasis = len(self.bfs)
        # note this is center of positive charge
        self.center_of_charge = np.asarray([sum([x[0]*x[1][0] for x in self.atoms]),
                                            sum([x[0]*x[1][1] for x in self.atoms]),
                                            sum([x[0]*x[1][2] for x in self.atoms])])\
                                         * (1./sum([x[0] for x in self.atoms]))
        self.one_electron_integrals()
        self.two_electron_integrals()




    def momentum2shell(self,momentum):
        shells = {
            'S' : [(0,0,0)],
            'P' : [(1,0,0),(0,1,0),(0,0,1)],
            'D' : [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
            'F' : [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
                   (0,3,0),(0,2,1),(0,1,2), (0,0,3)]
            }
        return shells[str(momentum)]
        
    def sym2num(self,sym):
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
        
    def read_molecule(self,filename):
        with open(filename) as f:
            atomlist = []
            for line_number,line in enumerate(f):
                if line_number == 0:
                    assert len(line.split()) == 1
                    natoms = int(line.split()[0])
                elif line_number == 1:
                    assert len(line.split()) == 2
                    charge = int(line.split()[0])
                    multiplicity = int(line.split()[1])
                else: 
                    if len(line.split()) == 0: break
                    assert len(line.split()) == 4
                    sym = self.sym2num(str(line.split()[0]))
                    x   = float(line.split()[1])
                    y   = float(line.split()[2])
                    z   = float(line.split()[3])
                    #atomlist.append((sym,(x,y,z)))
                    atomlist.append((sym,np.asarray([x,y,z])))
    
        return charge, multiplicity, atomlist

    def one_electron_integrals(self):
        N = self.nbasis
        self.S = np.zeros((N,N)) 
        self.V = np.zeros((N,N)) 
        self.T = np.zeros((N,N)) 
        # dipole integrals
        self.Mx = np.zeros((N,N)) 
        self.My = np.zeros((N,N)) 
        self.Mz = np.zeros((N,N)) 
        self.nuc_energy = 0.0
        # Get one electron integrals
        print "One-electron integrals"
        for i in tqdm(range(N)):
            for j in range(N):
                self.S[i,j] = S(self.bfs[i],self.bfs[j])
                self.T[i,j] = T(self.bfs[i],self.bfs[j])
                self.Mx[i,j] = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'x')
                self.My[i,j] = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'y')
                self.Mz[i,j] = Mu(self.bfs[i],self.bfs[j],self.center_of_charge,'z')
                for atom in self.atoms:
                    self.V[i,j] += -atom[0]*V(self.bfs[i],self.bfs[j],atom[1])
        # Also populate nuclear repulsion at this time
        for pair in itertools.combinations(self.atoms,2):
            self.nuc_energy += pair[0][0]*pair[1][0]/np.linalg.norm(pair[0][1] - pair[1][1])

        # preparing for SCF
        self.Core       = self.T + self.V
        self.X          = mat_pow(self.S,-0.5)
        self.U          = mat_pow(self.S,0.5)
        print "\n"

    def two_electron_integrals(self):
        N = self.nbasis
        self.TwoE = np.zeros((N,N,N,N))  
        print "Two-electron integrals"
        for i in trange(N,desc='First loop'):
            for j in trange(N,desc='Second loop'):
                for k in trange(N,desc='Third loop'):
                    for l in trange(N,desc='Fourth loop'):
                        if i >= j:
                            if k >= l:
                                if (i*(i+1)//2 + j) >= (k*(k+1)//2 + l):
                                    val = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l])
                                    self.TwoE[i,j,k,l] = val
                                    self.TwoE[k,l,i,j] = val
                                    self.TwoE[j,i,l,k] = val
                                    self.TwoE[l,k,j,i] = val
                                    self.TwoE[j,i,k,l] = val
                                    self.TwoE[l,k,i,j] = val
                                    self.TwoE[i,j,l,k] = val
                                    self.TwoE[k,l,j,i] = val
        print "\n\n"
    def SCF(self):
        self.delta_energy = 1e20
        self.P_RMS        = 1e20
        self.P_old        = None
        En = []
        maxiter = 80
        for step in xrange(maxiter):
            if step == 0:
                self.F = self.Core
            else:
                self.P_old      = self.P
                energy_old = self.energy
                self.buildFock()
            self.FO     = np.dot(self.X.T,np.dot(self.F,self.X))
            E,self.CO   = np.linalg.eigh(self.FO)
            C      = np.dot(self.X,self.CO)
            self.P = np.einsum('pi,qi->pq', C[:,:self.nocc], C[:,:self.nocc])
           # if step == 0:
           #     np.save("Po.npy",self.P)

            self.el_energy = np.einsum('pq,pq',self.P,self.Core+self.F)
            self.energy    = self.el_energy + self.nuc_energy
            #print self.energy
            En.append(self.energy)
            if step > 0:
                self.delta_energy = self.energy - energy_old
                self.P_RMS        = np.std(self.P - self.P_old)
            if np.abs(self.delta_energy) < 1e-14 or self.P_RMS < 1e-12 or step == (maxiter - 1):
                if step == (maxiter - 1):
                    print "NOT CONVERGED"
                print "E(SCF)    = ", "{0:.12f}".format(self.energy.real)+ \
                      " in "+str(step)+" iterations"
                print " RMS(P)  = ", "{0:.2e}".format(self.P_RMS.real)
                print " dE(SCF) = ", "{0:.2e}".format(self.delta_energy.real)
                self.computeDipole()
                print " Dipole X = ", "{0:.8f}".format(self.mu_x)
                print " Dipole Y = ", "{0:.8f}".format(self.mu_y)
                print " Dipole Z = ", "{0:.8f}".format(self.mu_z)
                #import matplotlib.pyplot as plt
                #plt.plot(range(len(En[1:])),En[1:])
                #plt.show()
                break

    def buildFock(self):
        self.J = np.einsum('pqrs,rs->pq', self.TwoE,self.P)
        self.K = np.einsum('prqs,rs->pq', self.TwoE,self.P)
        self.G = 2.*self.J - self.K
        self.GO = np.dot(self.X.T,np.dot(self.G,self.X))
        self.F = self.Core + self.G

    def orthoFock(self):
        self.FO = np.dot(self.X.T,np.dot(self.F,self.X))

    def orthoDen(self):
        self.PO = np.dot(self.U,np.dot(self.P,self.U.T))

    def computeEnergy(self):
        self.el_energy = np.einsum('pq,pq',self.P,self.Core+self.F)
        self.energy    = self.el_energy + self.nuc_energy

    def computeDipole(self):
        self.mu_x = -2*np.trace(np.dot(self.P,self.Mx)) + sum([x[0]*(x[1][0]-self.center_of_charge[0]) for x in self.atoms])  
        self.mu_y = -2*np.trace(np.dot(self.P,self.My)) + sum([x[0]*(x[1][1]-self.center_of_charge[1]) for x in self.atoms])  
        self.mu_z = -2*np.trace(np.dot(self.P,self.Mz)) + sum([x[0]*(x[1][2]-self.center_of_charge[2]) for x in self.atoms])  
        # to debye
        self.mu_x *= 2.541765
        self.mu_y *= 2.541765
        self.mu_z *= 2.541765
         


        

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    filename = 'h2o.dat'
    h2o = Molecule(filename,basis='sto-3g')
    h2o.SCF()
    

