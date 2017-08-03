#from __future__ import division
#import numpy as np
#from joblib import Parallel, delayed
#from integrals import *
##from magnus import updateM4
#from hermite import ERI, doERIs, do2eGIAO
#from scipy.misc import factorial2 as fact2
#from scipy.linalg import fractional_matrix_power as mat_pow
#from scipy.linalg import expm,expm2,schur,lstsq
#from scipy.misc import factorial
#from scipy.fftpack import fft,fftfreq,ifft
#from scipy.signal import gaussian
#import itertools
#from tqdm import tqdm, trange 
#import matplotlib
#matplotlib.use('TkAgg')
#
#class BasisFunction(object):
#    def __init__(self,origin=(0,0,0),shell=(0,0,0),exps=[],coefs=[]):
#        assert len(origin)==3
#        assert len(shell)==3
#        self.origin = np.asarray(origin,'d')#*1.889725989 # to bohr
#        self.shell = np.asarray(shell).astype(int)
#        self.exps  = exps
#        self.coefs = coefs
#        self.normalize()
#
#    def normalize(self):
#        l,m,n = self.shell
#        # self.norm is a list of length number primitives
#        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
#                        np.power(self.exps,l+m+n+1.5)/
#                        fact2(2*l-1)/fact2(2*m-1)/
#                        fact2(2*n-1)/np.power(np.pi,1.5))
#        return
#
#class Molecule(object):
#    def __init__(self,filename,basis='sto3g',load=False,gauge=None):
#        self.load = load
#        charge, multiplicity, atomlist = self.read_molecule(filename)
#        self.charge = charge
#        self.multiplicity = multiplicity
#        self.atoms = atomlist
#        self.nelec = sum([atom[0] for atom in atomlist]) - charge 
#        self.nocc  = self.nelec//2
#        self.bfs = []
#        try:
#            import data
#        except ImportError:
#            print "No basis set data"
#            sys.exit(0)
#
#        basis_data = data.basis[basis]
#        for atom in self.atoms:
#            for momentum,prims in basis_data[atom[0]]:
#                exps = [e for e,c in prims]
#                coefs = [c for e,c in prims]
#                for shell in self.momentum2shell(momentum):
#                    #self.bfs.append(BasisFunction(atom[1],shell,exps,coefs))
#                    self.bfs.append(BasisFunction(atom[1],shell,exps,coefs))
#        self.nbasis = len(self.bfs)
#        # note this is center of positive charge
#        self.center_of_charge = np.asarray([sum([x[0]*x[1][0] for x in self.atoms]),
#                                            sum([x[0]*x[1][1] for x in self.atoms]),
#                                            sum([x[0]*x[1][2] for x in self.atoms])])\
#                                         * (1./sum([x[0] for x in self.atoms]))
#        if not gauge:
#            self.gauge_origin = self.center_of_charge
#        else:
#            self.gauge_origin = np.asarray(gauge)
#            
#        if self.load:
#            self.nuc_energy = np.load('enuc.npy')
#            self.nelec      = np.load('nelec.npy')
#            self.S          = np.load('S.npy')
#            self.T          = np.load('T.npy')
#            self.V          = np.load('V.npy')
#            self.Mx         = np.load('Mx.npy')
#            self.My         = np.load('My.npy')
#            self.Mz         = np.load('Mz.npy')
#            self.L          = np.load('L.npy') 
#            self.Sb         = np.load('dsdb.npy')
#            self.dhdb       = np.load('dhdb.npy')
#            self.dgdb       = np.load('dgdb.npy')
#            self.TwoE       = np.load('ERI.npy')
#            self.Core       = self.T + self.V
#            self.X          = mat_pow(self.S,-0.5)
#            self.U          = mat_pow(self.S,0.5)
#        else:
#            self.one_electron_integrals()
#            self.two_electron_integrals()
#            #self.GIAO_two_electron_integrals()
#            print "\n"
#
#
#
#    def momentum2shell(self,momentum):
#        shells = {
#            'S' : [(0,0,0)],
#            'P' : [(1,0,0),(0,1,0),(0,0,1)],
#            'D' : [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
#            'F' : [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
#                   (0,3,0),(0,2,1),(0,1,2), (0,0,3)]
#            }
#        return shells[str(momentum)]
#        
#    def sym2num(self,sym):
#        symbol = [
#            "X","H","He",
#            "Li","Be","B","C","N","O","F","Ne",
#            "Na","Mg","Al","Si","P","S","Cl","Ar",
#            "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
#            "Co", "Ni", "Cu", "Zn",
#            "Ga", "Ge", "As", "Se", "Br", "Kr",
#            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
#            "Rh", "Pd", "Ag", "Cd",
#            "In", "Sn", "Sb", "Te", "I", "Xe",
#            "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",  "Eu",
#            "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
#            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
#            "Tl","Pb","Bi","Po","At","Rn"]
#        return symbol.index(str(sym))
#        
#    def read_molecule(self,filename):
#        with open(filename) as f:
#            atomlist = []
#            for line_number,line in enumerate(f):
#                if line_number == 0:
#                    assert len(line.split()) == 1
#                    natoms = int(line.split()[0])
#                elif line_number == 1:
#                    assert len(line.split()) == 2
#                    charge = int(line.split()[0])
#                    multiplicity = int(line.split()[1])
#                else: 
#                    if len(line.split()) == 0: break
#                    assert len(line.split()) == 4
#                    sym = self.sym2num(str(line.split()[0]))
#                    x   = float(line.split()[1])*1.889725989
#                    y   = float(line.split()[2])*1.889725989
#                    z   = float(line.split()[3])*1.889725989
#                    #atomlist.append((sym,(x,y,z)))
#                    atomlist.append((sym,np.asarray([x,y,z])))
#    
#        return charge, multiplicity, atomlist
#
#    def one_electron_integrals(self):
#        N = self.nbasis
#        # core integrals
#        self.S = np.zeros((N,N)) 
#        self.rH = np.zeros((3,N,N)) 
#        self.V = np.zeros((N,N)) 
#        self.T = np.zeros((N,N)) 
#        # dipole integrals
#        self.Mx = np.zeros((N,N)) 
#        self.My = np.zeros((N,N)) 
#        self.Mz = np.zeros((N,N)) 
# 
#        # GIAO quasienergy dipole intgrals
#        self.rDipX = np.zeros((3,N,N))
#        self.rDipY = np.zeros((3,N,N))
#        self.rDipZ = np.zeros((3,N,N))
#        
#        # angular momentum
#        self.L = np.zeros((3,N,N)) 
#
#        #GIAO overlap
#        self.Sb = np.zeros((3,N,N))
#
#        # derivative of one-electron GIAO integrals wrt B at B = 0.
#        self.rH = np.zeros((3,N,N)) 
#
#        # London Angular momentum L_N
#        self.Ln = np.zeros((3,N,N))
#
#        self.dhdb = np.zeros((3,N,N))
#
#        self.nuc_energy = 0.0
#        # Get one electron integrals
#        print "One-electron integrals"
#
#        for i in tqdm(range(N)):
#            for j in range(i+1):
#                self.S[i,j] = self.S[j,i] \
#                    = S(self.bfs[i],self.bfs[j])
#                self.T[i,j] = self.T[j,i] \
#                    = T(self.bfs[i],self.bfs[j])
#                self.Mx[i,j] = self.Mx[j,i] \
#                    = Mu(self.bfs[i],self.bfs[j],'x',gOrigin=self.gauge_origin)
#                self.My[i,j] = self.My[j,i] \
#                    = Mu(self.bfs[i],self.bfs[j],'y',gOrigin=self.gauge_origin)
#                self.Mz[i,j] = self.Mz[j,i] \
#                    = Mu(self.bfs[i],self.bfs[j],'z',gOrigin=self.gauge_origin)
#               # self.Mx[i,j] = self.Mx[j,i] \
#               #     = S(self.bfs[i],self.bfs[j],n=(1,0,0)) 
#               # self.My[i,j] = self.My[j,i] \
#               #     = S(self.bfs[i],self.bfs[j],n=(0,1,0)) 
#               # self.Mz[i,j] = self.Mz[j,i] \
#               #     = S(self.bfs[i],self.bfs[j],n=(0,0,1)) 
#                for atom in self.atoms:
#                    self.V[i,j] += -atom[0]*V(self.bfs[i],self.bfs[j],atom[1])
#                self.V[j,i] = self.V[i,j]
#        # Also populate nuclear repulsion at this time
#        for pair in itertools.combinations(self.atoms,2):
#            self.nuc_energy += pair[0][0]*pair[1][0]/np.linalg.norm(pair[0][1] - pair[1][1])
#           
#        # Do GIAOs (no symmetry) 
#        print "GIAO one-electron integrals"
#        for i in tqdm(range(N)):
#            for j in range(N):
#                self.L[0,i,j] \
#                    = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'x')
#                self.L[1,i,j] \
#                    = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'y')
#                self.L[2,i,j] \
#                    = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'z')
#                #QAB matrix elements
#                XAB = self.bfs[i].origin[0] - self.bfs[j].origin[0]
#                YAB = self.bfs[i].origin[1] - self.bfs[j].origin[1]
#                ZAB = self.bfs[i].origin[2] - self.bfs[j].origin[2]
#                # GIAO T
#                self.rH[0,i,j] = T(self.bfs[i],self.bfs[j],n=(1,0,0),gOrigin=self.gauge_origin)
#                self.rH[1,i,j] = T(self.bfs[i],self.bfs[j],n=(0,1,0),gOrigin=self.gauge_origin)
#                self.rH[2,i,j] = T(self.bfs[i],self.bfs[j],n=(0,0,1),gOrigin=self.gauge_origin)
#
#                # testing GIAO quasienergy dipole contribs
#                self.rDipX[0,i,j] = Mu(self.bfs[i],self.bfs[j],'x',n=(1,0,0),gOrigin=self.gauge_origin)
#                self.rDipX[1,i,j] = Mu(self.bfs[i],self.bfs[j],'x',n=(0,1,0),gOrigin=self.gauge_origin)
#                self.rDipX[2,i,j] = Mu(self.bfs[i],self.bfs[j],'x',n=(0,0,1),gOrigin=self.gauge_origin)
#
#                self.rDipY[0,i,j] = Mu(self.bfs[i],self.bfs[j],'y',n=(1,0,0),gOrigin=self.gauge_origin)
#                self.rDipY[1,i,j] = Mu(self.bfs[i],self.bfs[j],'y',n=(0,1,0),gOrigin=self.gauge_origin)
#                self.rDipY[2,i,j] = Mu(self.bfs[i],self.bfs[j],'y',n=(0,0,1),gOrigin=self.gauge_origin)
#
#                self.rDipZ[0,i,j] = Mu(self.bfs[i],self.bfs[j],'z',n=(1,0,0),gOrigin=self.gauge_origin)
#                self.rDipZ[1,i,j] = Mu(self.bfs[i],self.bfs[j],'z',n=(0,1,0),gOrigin=self.gauge_origin)
#                self.rDipZ[2,i,j] = Mu(self.bfs[i],self.bfs[j],'z',n=(0,0,1),gOrigin=self.gauge_origin)
#
#                for atom in self.atoms:
#                    # GIAO V
#                    self.rH[0,i,j] += -atom[0]*V(self.bfs[i],self.bfs[j],atom[1],n=(1,0,0),gOrigin=self.gauge_origin)
#                    self.rH[1,i,j] += -atom[0]*V(self.bfs[i],self.bfs[j],atom[1],n=(0,1,0),gOrigin=self.gauge_origin)
#                    self.rH[2,i,j] += -atom[0]*V(self.bfs[i],self.bfs[j],atom[1],n=(0,0,1),gOrigin=self.gauge_origin)
#
#                # Some temp copies for mult with QAB matrix 
#                xH = self.rH[0,i,j]
#                yH = self.rH[1,i,j]
#                zH = self.rH[2,i,j]
#               
#                # add QAB contribution 
#                self.rH[0,i,j] = 0.5*(-ZAB*yH + YAB*zH)
#                self.rH[1,i,j] = 0.5*( ZAB*xH - XAB*zH)
#                self.rH[2,i,j] = 0.5*(-YAB*xH + XAB*yH)
#
#                # ditto for GIAO quasienergy dipoles
#                xDipX = self.rDipX[0,i,j]
#                yDipX = self.rDipX[1,i,j]
#                zDipX = self.rDipX[2,i,j]
#
#                xDipY = self.rDipY[0,i,j]
#                yDipY = self.rDipY[1,i,j]
#                zDipY = self.rDipY[2,i,j]
#
#                xDipZ = self.rDipZ[0,i,j]
#                yDipZ = self.rDipZ[1,i,j]
#                zDipZ = self.rDipZ[2,i,j]
#
#                self.rDipX[0,i,j] = 0.5*(-ZAB*yDipX + YAB*zDipX)
#                self.rDipX[1,i,j] = 0.5*( ZAB*xDipX - XAB*zDipX)
#                self.rDipX[2,i,j] = 0.5*(-YAB*xDipX + XAB*yDipX)
#
#                self.rDipY[0,i,j] = 0.5*(-ZAB*yDipY + YAB*zDipY)
#                self.rDipY[1,i,j] = 0.5*( ZAB*xDipY - XAB*zDipY)
#                self.rDipY[2,i,j] = 0.5*(-YAB*xDipY + XAB*yDipY)
#
#                self.rDipZ[0,i,j] = 0.5*(-ZAB*yDipZ + YAB*zDipZ)
#                self.rDipZ[1,i,j] = 0.5*( ZAB*xDipZ - XAB*zDipZ)
#                self.rDipZ[2,i,j] = 0.5*(-YAB*xDipZ + XAB*yDipZ)
#
#                # add QAB contribution for overlaps 
#                #C = np.asarray([0,0,0])
#                Rx = S(self.bfs[i],self.bfs[j],n=(1,0,0),gOrigin=self.gauge_origin)
#                Ry = S(self.bfs[i],self.bfs[j],n=(0,1,0),gOrigin=self.gauge_origin)
#                Rz = S(self.bfs[i],self.bfs[j],n=(0,0,1),gOrigin=self.gauge_origin)
#                self.Sb[0,i,j] = 0.5*(-ZAB*Ry + YAB*Rz)
#                self.Sb[1,i,j] = 0.5*( ZAB*Rx - XAB*Rz)
#                self.Sb[2,i,j] = 0.5*(-YAB*Rx + XAB*Ry)
#
#                # now do Angular London Momentum
#                self.Ln[0,i,j] = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'x',london=True)
#                self.Ln[1,i,j] = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'y',london=True)
#                self.Ln[2,i,j] = RxDel(self.bfs[i],self.bfs[j],self.gauge_origin,'z',london=True)
#
#        # below gives dH/dB accoriding to dalton
#        self.dhdb[:] = 0.5*self.Ln[:] + self.rH[:]
#        #self.dhdb[1] = 0.5*self.Ln[1] + self.rH[1] 
#        #self.dhdb[2] = 0.5*self.Ln[2] + self.rH[2]
#
#
#        # preparing for SCF
#        self.Core       = self.T + self.V
#        self.X          = mat_pow(self.S,-0.5)
#        self.U          = mat_pow(self.S,0.5)
#
#
#    def two_electron_integrals(self):
#        N = self.nbasis
#        self.TwoE = np.zeros((N,N,N,N))  
#        print "Two-electron integrals"
#        #self.TwoE = pop_TwoE(N,self.TwoE)
#        self.TwoE = doERIs(N,self.TwoE,self.bfs)
#        self.TwoE = np.asarray(self.TwoE)
#      #  for i in trange(N,desc='First loop'):
#      #      for j in trange(i+1,desc='Second loop'):
#      #          ij = (i*(i+1)//2 + j)
#      #          for k in trange(N,desc='Third loop'):
#      #              for l in trange(k+1,desc='Fourth loop'):
#      #                  kl = (k*(k+1)//2 + l)
#      #                  if ij >= kl:
#      #                     val = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l])
#      #                     self.TwoE[i,j,k,l] = val
#      #                     self.TwoE[k,l,i,j] = val
#      #                     self.TwoE[j,i,l,k] = val
#      #                     self.TwoE[l,k,j,i] = val
#      #                     self.TwoE[j,i,k,l] = val
#      #                     self.TwoE[l,k,i,j] = val
#      #                     self.TwoE[i,j,l,k] = val
#      #                     self.TwoE[k,l,j,i] = val
#
#    def GIAO_two_electron_integrals(self):
#        # note these do not have the same symetry as undifferentiated integrals.
#        # some symmetry exists (see Ruud et al. 'Hartree-Fock limit magnetizabilities')
#        # I have tried to exploit the permutational symmetry accordingly
#        N = self.nbasis
#        self.GR1 = np.zeros((3,N,N,N,N))  
#        self.GR2 = np.zeros((3,N,N,N,N))  
#        self.dgdb = np.zeros((3,N,N,N,N))  
#        print "GIAO two-electron integrals"
#        self.dgdb = do2eGIAO(N,self.GR1,self.GR2,self.dgdb,self.bfs,self.gauge_origin)
#        self.dgdb = np.asarray(self.dgdb)
##        ij = 0
##        for i in trange(N,desc='First loop'):
##            for j in trange(N,desc='Second loop'):
##                ij += 1
##                kl = 0
##                for k in trange(N,desc='Third loop'):
##                    ik = i + k
##                    for l in trange(N,desc='Fourth loop'):
##                        kl += 1
##                        if (ij >= kl and ik >= j+l and not (i==j and k==l)):
##                            #QMN matrix elements
##                            XMN  = self.bfs[i].origin[0] - self.bfs[j].origin[0]
##                            YMN  = self.bfs[i].origin[1] - self.bfs[j].origin[1]
##                            ZMN  = self.bfs[i].origin[2] - self.bfs[j].origin[2]
##                            #QPQ matrix elements
##                            XPQ  = self.bfs[k].origin[0] - self.bfs[l].origin[0]
##                            YPQ  = self.bfs[k].origin[1] - self.bfs[l].origin[1]
##                            ZPQ  = self.bfs[k].origin[2] - self.bfs[l].origin[2]
##
##                            GR1x = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],n1=(1,0,0),n2=(0,0,0), gOrigin=self.gauge_origin)
##                            GR1y = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],n1=(0,1,0),n2=(0,0,0), gOrigin=self.gauge_origin)
##                            GR1z = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],n1=(0,0,1),n2=(0,0,0), gOrigin=self.gauge_origin)
##                            GR2x = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],n1=(0,0,0),n2=(1,0,0), gOrigin=self.gauge_origin)
##                            GR2y = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],n1=(0,0,0),n2=(0,1,0), gOrigin=self.gauge_origin)
##                            GR2z = ERI(self.bfs[i],self.bfs[j],self.bfs[k],self.bfs[l],n1=(0,0,0),n2=(0,0,1), gOrigin=self.gauge_origin)
##
##                            # add QMN contribution
##                            self.GR1[0,i,j,k,l] = 0.5*(-ZMN*GR1y + YMN*GR1z)
##                            self.GR1[1,i,j,k,l] = 0.5*( ZMN*GR1x - XMN*GR1z)
##                            self.GR1[2,i,j,k,l] = 0.5*(-YMN*GR1x + XMN*GR1y)
##                            # add QPQ contribution
##                            self.GR2[0,i,j,k,l] = 0.5*(-ZPQ*GR2y + YPQ*GR2z)
##                            self.GR2[1,i,j,k,l] = 0.5*( ZPQ*GR2x - XPQ*GR2z)
##                            self.GR2[2,i,j,k,l] = 0.5*(-YPQ*GR2x + XPQ*GR2y)
##
##                            self.dgdb[:,i,j,k,l] = self.dgdb[:,k,l,i,j] = self.GR1[:,i,j,k,l] + self.GR2[:,i,j,k,l]
##                            self.dgdb[:,j,i,l,k] = self.dgdb[:,l,k,j,i] = -self.dgdb[:,i,j,k,l] 
##
#
#    def SCF(self,doPrint=True,save=True,method=None):
#        self.delta_energy = 1e20
#        self.P_RMS        = 1e20
#        self.P_old        = np.zeros((self.nbasis,self.nbasis)) 
#        if not method:
#            En = []
#            num_e = 8
#            FockSet = []
#            ErrorSet = []
#            maxiter = 200
#            for step in xrange(maxiter):
#                if step == 0:
#                    #print "Using old P"
#                    self.P_old      = self.P
#                    self.buildFock()
#                    self.computeEnergy()
#                    #print self.energy
#                    #self.F = self.Core
#                else:
#                    self.P_old      = self.P
#                    energy_old = self.energy
#                    self.buildFock()
#                doDIIS = False 
#                if doDIIS:
#                    if step >= 0:
#                        FPS = np.dot(self.F,np.dot(self.P,self.S))
#                        SPF = np.dot(self.S,np.dot(self.P,self.F))
#                        error = FPS - SPF 
#                        if np.linalg.norm(error) < 1e-8:
#                           doDIIS = False 
#                        if len(ErrorSet) < num_e:
#                            FockSet.append(self.F)
#                            ErrorSet.append(error)
#                        elif len(ErrorSet) >= num_e:
#                            del FockSet[0]
#                            del ErrorSet[0]
#                            FockSet.append(self.F)
#                            ErrorSet.append(error)
#                        NErr = len(ErrorSet)
#                        if NErr >= 2:
#                            Bmat = np.zeros((NErr+1,NErr+1),dtype='complex')
#                            ZeroVec = np.zeros((NErr+1))
#                            ZeroVec[-1] = -1.0
#                            for a in range(0,NErr):
#                                for b in range(0,a+1):
#                                    Bmat[a,b] = Bmat[b,a] = np.trace(np.dot(ErrorSet[a].T,ErrorSet[b])) 
#                                    Bmat[a,NErr] = Bmat[NErr,a] = -1.0
#                            try:
#                                coeff = np.linalg.solve(Bmat,ZeroVec)
#                            except np.linalg.linalg.LinAlgError as err:
#                                if 'Singular matrix' in err.message:
#                                    print '\tSingular B matrix, turing off DIIS'
#                                    do_DIIS = False
#                            else:
#                                F = 0.0
#                                for i in range(0,len(coeff)-1):
#                                    F += coeff[i]*FockSet[i]
#                                self.F = F
#
#                self.FO     = np.dot(self.X.T,np.dot(self.F,self.X))
#                E,self.CO   = np.linalg.eigh(self.FO)
#
#                C      = np.dot(self.X,self.CO)
#                self.C      = np.dot(self.X,self.CO)
#                self.MO     = E
#                #self.P = np.einsum('pi,qi->pq', C[:,:self.nocc], np.conjugate(C[:,:self.nocc]))
#                self.P = np.dot(C[:,:self.nocc],np.conjugate(C[:,:self.nocc]).T)
#                #self.P = np.einsum('pi,qi->pq', np.conjugate(C[:,:self.nocc]), C[:,:self.nocc])
#               # if step == 0:
#               #     np.save("Po.npy",self.P)
#
#                #self.buildFock() 
#                self.computeEnergy()
#                #self.el_energy = np.einsum('pq,pq',self.P,self.Core+self.F)
#                #self.energy    = self.el_energy + self.nuc_energy
#                #print self.energy
#                En.append(self.energy)
#                if step > 0:
#                    self.delta_energy = self.energy - energy_old
#                    self.P_RMS        = np.linalg.norm(self.P - self.P_old)
#                FPS = np.dot(self.F,np.dot(self.P,self.S))
#                SPF = np.dot(self.S,np.dot(self.P,self.F))
#                SPF = np.conjugate(FPS).T
#                error = np.linalg.norm(FPS - SPF)
#                #print self.P_RMS 
#                if np.abs(self.delta_energy) < 1e-14 or self.P_RMS < 1e-14 or step == (maxiter - 1):
#                #if self.P_RMS < 1e-8 or step == (maxiter - 1):
#                #if np.abs(error) < 1e-14 or step == (maxiter - 1):
#                    if step == (maxiter - 1):
#                        print "NOT CONVERGED"
#                    elif doPrint:
#                        FPS = np.dot(self.F,np.dot(self.P,self.S))
#                        SPF = np.dot(self.S,np.dot(self.P,self.F))
#                        error = FPS - SPF
#                        print "Error", np.linalg.norm(error)
#                        print "E(SCF)    = ", "{0:.12f}".format(self.energy.real)+ \
#                              " in "+str(step)+" iterations"
#                        print " RMS(P)  = ", "{0:.2e}".format(self.P_RMS.real)
#                        print " dE(SCF) = ", "{0:.2e}".format(self.delta_energy.real)
#                        self.computeDipole()
#                        print " Dipole X = ", "{0:.8f}".format(self.mu_x)
#                        print " Dipole Y = ", "{0:.8f}".format(self.mu_y)
#                        print " Dipole Z = ", "{0:.8f}".format(self.mu_z)
#                        self.orthoDen()
#                        
#                    if save:
#                        np.save('enuc.npy',self.nuc_energy)
#                        np.save('nelec.npy',self.nelec)
#                        np.save('S.npy',self.S)
#                        np.save('T.npy',self.T)
#                        np.save('V.npy',self.V)
#                        np.save('Mx.npy',self.Mx)
#                        np.save('My.npy',self.My)
#                        np.save('Mz.npy',self.Mz)
#                        np.save('ERI.npy',self.TwoE)
#                        np.save('F.npy',self.F)
#                        np.save('P.npy',self.P)
#                    break
#        elif method == 'imagtime':
#            En = []
#            dt = 0.050
#            for step in xrange(1000000):
#                if step == 0:
#                    #self.P = np.random.rand(self.nbasis,self.nbasis)
#                    #self.P = np.zeros((self.nbasis,self.nbasis),dtype='complex')
#                    #self.F = self.Core
#                    self.buildFock()
#                    self.orthoFock()
#                    self.FO,R = np.linalg.qr(self.F)
#                    E,self.CO = np.linalg.eigh(self.FO)
#                    t = 0.0
#                else:
#                    self.buildFock()
#                    self.orthoFock()
#                    self.P_old      = self.P
#                    energy_old = self.energy
#                    #t += dt
#
#                self.timeProp(dt)
#                self.computeEnergy()
#
#                print self.energy, dt
#                #E = np.linalg.eigvalsh(self.FO)
#                if step > 0:
#                    self.delta_energy = self.energy - energy_old
#                    self.P_RMS        = np.std(abs(self.P) - abs(self.P_old))
#                    #En.append(np.log10(abs(self.delta_energy)))
#                    En.append(self.energy)
#
#                if np.abs(self.delta_energy) < 1e-14 or np.abs(self.P_RMS) < 1e-12:
#                    self.FO     = np.dot(self.X.T,np.dot(self.F,self.X))
#                    E,self.CO   = np.linalg.eigh(self.FO)
#
#                    C      = np.dot(self.X,self.CO)
#                    self.C      = np.dot(self.X,self.CO)
#                    self.MO     = E
#                    self.P = np.einsum('pi,qi->pq', C[:,:self.nocc], np.conjugate(C[:,:self.nocc]))
#                    #self.P = np.einsum('pi,qi->pq', np.conjugate(C[:,:self.nocc]), C[:,:self.nocc])
#
#                    if doPrint:
#                        FPS = np.dot(self.F,np.dot(self.P,self.S))
#                        SPF = np.dot(self.S,np.dot(self.P,self.F))
#                        error = FPS - SPF
#                        print "Error", np.linalg.norm(error)
#                        print "E(SCF)    = ", "{0:.12f}".format(self.energy.real)+ \
#                              " in "+str(step)+" iterations"
#                        print " RMS(P)  = ", "{0:.2e}".format(self.P_RMS.real)
#                        print " dE(SCF) = ", "{0:.2e}".format(self.delta_energy.real)
#                        self.computeDipole()
#                        print " Dipole X = ", "{0:.8f}".format(self.mu_x)
#                        print " Dipole Y = ", "{0:.8f}".format(self.mu_y)
#                        print " Dipole Z = ", "{0:.8f}".format(self.mu_z)
#                        self.orthoDen()
#                        self.buildL(direction='z',P=self.P)
#                        Lz = (self.LN)
#                        dSz = -self.dSdB#**2
#                        dHz = -self.dHdB#**2
#                        dGz = -self.dGdB#**2
#                        self.buildL(direction='y',P=self.P)
#                        Ly = (self.LN)
#                        dSy = -self.dSdB#**2
#                        dHy = -self.dHdB#**2
#                        dGy = -self.dGdB#**2
#                        self.buildL(direction='x',P=self.P)
#                        Lx = (self.LN)
#                        dSx = -self.dSdB#**2
#                        dHx = -self.dHdB#**2
#                        dGx = -self.dGdB#**2
#                        print "GIAO Tot L:     ", "{0:.8f}".format(np.sqrt(Lx*Lx+Ly*Ly+Lz*Lz))
#                        print "GIAO Tot Lz:     ", "{0:.8f}".format(Lz)
#                        print "GIAO Tot Ly:     ", "{0:.8f}".format(Ly)
#                        print "GIAO Tot Lx:     ", "{0:.8f}".format(Lx)
#                        nLx = np.trace(np.dot(self.P,1j*(self.L[0])))
#                        nLy = np.trace(np.dot(self.P,1j*(self.L[1])))
#                        nLz = np.trace(np.dot(self.P,1j*(self.L[2])))
#                        print "non-GIAO Tot L: ", "{0:.8f}".format(np.sqrt(nLx*nLx +nLy*nLy+nLz*nLz))
#                        print "non-GIAO Lz: ", "{0:.8f}".format(nLz)
#                        print "non-GIAO Ly: ", "{0:.8f}".format(nLy)
#                        print "non-GIAO Lx: ", "{0:.8f}".format(nLx)
#                        print "dSdBz tot: ", dSz#np.sqrt(dSx + dSy + dSz)
#                        print "dHdBz tot: ", dHz#np.sqrt(dHx + dHy + dHz)
#                        print "dGdBz tot: ", dGz#np.sqrt(dGx + dGy + dGz)
#
#                        
#                    if save:
#                        np.save('enuc.npy',self.nuc_energy)
#                        np.save('nelec.npy',self.nelec)
#                        np.save('S.npy',self.S)
#                        np.save('T.npy',self.T)
#                        np.save('V.npy',self.V)
#                        np.save('Mx.npy',self.Mx)
#                        np.save('My.npy',self.My)
#                        np.save('Mz.npy',self.Mz)
#                        np.save('L.npy',self.L)
#                        np.save('dsdb.npy',self.Sb)
#                        np.save('dhdb.npy',self.dhdb)
#                        np.save('dgdb.npy',self.dgdb)
#                        np.save('ERI.npy',self.TwoE)
#                        np.save('F.npy',self.F)
#                        np.save('P.npy',self.P)
#                    break
#
#    def timeProp(self,dt):
#        C0          = np.copy(self.CO)
#        C           = C0 - dt*np.dot(self.FO,self.CO)
#
#        self.CO, R  = np.linalg.qr(C)
#        C           = np.dot(self.X,self.CO)
#        self.P      = np.einsum('pi,qi->pq', C[:,:self.nocc], np.conjugate(C[:,:self.nocc]))
#        #self.P = np.einsum('pi,qi->pq', np.conjugate(C[:,:self.nocc]), C[:,:self.nocc])
#
#        self.buildFock() 
#        self.orthoFock()
#
#        U           = expm(-dt*(self.FO))
#        C           = np.dot(U,C0)
#        self.CO, R  = np.linalg.qr(C)
#        C           = np.dot(self.X,self.CO)
#        self.P      = np.einsum('pi,qi->pq', C[:,:self.nocc], np.conjugate(C[:,:self.nocc]))
#        #self.P = np.einsum('pi,qi->pq', np.conjugate(C[:,:self.nocc]), C[:,:self.nocc])
#
#
#
#    def MP2(self):
#
#        self.single_bar = np.einsum('mp,mnlz->pnlz',self.C,self.TwoE)
#        temp            = np.einsum('nq,pnlz->pqlz',self.C,self.single_bar)
#        self.single_bar = np.einsum('lr,pqlz->pqrz',self.C,temp)
#        temp            = np.einsum('zs,pqrz->pqrs',self.C,self.single_bar)
#        self.single_bar = temp
#
#        EMP2 = 0.0
#        for i in range(self.nocc):
#            for j in range(self.nocc):
#                for a in range(self.nocc,self.nbasis):
#                    for b in range(self.nocc,self.nbasis):
#                        denom = self.MO[i] + self.MO[j] - self.MO[a] - self.MO[b]
#                        numer = self.single_bar[i,a,j,b]*(2.0*self.single_bar[i,a,j,b] - self.single_bar[i,b,j,a])
#                        EMP2 += numer/denom
#
#        print 'E(MP2) = ', EMP2 + self.energy
#
#    def Stable(self):
#        self.single_bar = np.einsum('mp,mnlz->pnlz',self.C,self.TwoE)
#        temp            = np.einsum('nq,pnlz->pqlz',self.C,self.single_bar)
#        self.single_bar = np.einsum('lr,pqlz->pqrz',self.C,temp)
#        temp            = np.einsum('zs,pqrz->pqrs',self.C,self.single_bar)
#        self.single_bar = temp
#
#
#        self.nvirt = self.nbasis - self.nocc
#        NOV = self.nocc * self.nvirt
#        A = np.zeros((NOV,NOV),dtype='complex')
#        B = np.zeros((NOV,NOV),dtype='complex')
#
#        ia = -1
#        for i in range(self.nocc):
#            for a in range(self.nocc,self.nbasis):
#                ia += 1
#                jb = -1
#                for j in range(self.nocc):
#                    for b in range(self.nocc,self.nbasis):
#                        jb += 1
#                        A[ia,jb] = (self.MO[a] - self.MO[i]) * ( i == j ) * (a == b) + 2*self.single_bar[a,i,j,b] - self.single_bar[a,b,j,i]
#                        B[ia,jb] = 2*self.single_bar[i,a,j,b] - self.single_bar[i,b,j,a]
#        M = np.bmat([[A,B],[np.conjugate(B).T,np.conjugate(A).T]]).astype('complex')
#        # careful with eigensolver...
#        E,C = np.linalg.eigh(M)
#
#        X = C[:NOV,0].reshape(self.nocc,self.nvirt)
#        #Y = C[NOV:,0].reshape(self.nvirt,self.nocc)
#
#        #print np.dot(C[:,0].T,np.dot(M,C[:,0]))*27.2114
#        
#        #print X
#        #print -X.T 
#
#        
#        print E[0:3]*27.2114
#        theta = 0.001 
#        U = expm(-theta*np.bmat([[np.zeros((self.nocc,self.nocc)),X],[-X.T,np.zeros((self.nvirt,self.nvirt))]]))
#        self.P = np.dot(U,np.dot(self.P,np.conjugate(U).T))
#        
#
#     
#        
#
#   
#    def RT(self,numsteps=1000,stepsize=0.1,field=0.0001,direction='x',t2=None):
#        self.SCF(doPrint=False,save=False)
#        self.dipole     = []
#        self.angmom     = []
#        self.Sbmom     = []
#        self.Hbmom     = []
#        self.Gbmom     = []
#        self.Energy     = []
#        self.t2 = t2
#        self.field = field
#        self.stepsize = stepsize
#        self.numSteps = numsteps
#        self.time = np.arange(0,self.numSteps)*self.stepsize
#        self.shape = []
#        #self.Magnus4(direction=direction)
#        self.Magnus2(direction=direction)
#       
#    def buildFock(self):
#        self.J = np.einsum('pqrs,sr->pq', self.TwoE.astype('complex'),self.P)
#        # note indices for K
#        #self.K = np.einsum('prqs,rs->pq', self.TwoE.astype('complex'),self.P)
#        self.K = np.einsum('psqr,sr->pq', self.TwoE.astype('complex'),self.P)
#        self.G = 2.*self.J - self.K
#        #self.GO = np.dot(self.X.T,np.dot(self.G,self.X))
#        self.F = self.Core.astype('complex') + self.G
#        #self.F = np.conjugate(self.Core.astype('complex') + self.G)
#
#    def orthoFock(self):
#        self.FO = np.dot(self.X.T,np.dot(self.F,self.X))
#
#    def unOrthoFock(self):
#        self.F = np.dot(self.U.T,np.dot(self.FO,self.U))
#
#    def orthoDen(self):
#        self.PO = np.dot(self.U,np.dot(self.P,self.U.T))
#
#    def unOrthoDen(self):
#        self.P = np.dot(self.X,np.dot(self.PO,self.X.T))
#
#    def computeEnergy(self):
#        #self.el_energy = np.einsum('pq,pq',self.P.T,self.Core+self.F)
#        self.el_energy = np.einsum('pq,qp',self.Core+self.F,self.P)
#        self.energy    = self.el_energy + self.nuc_energy
#
#    def computeDipole(self):
#        self.mu_x = -2*np.trace(np.dot(self.P,self.Mx)) + sum([x[0]*(x[1][0]-self.gauge_origin[0]) for x in self.atoms])  
#        self.mu_y = -2*np.trace(np.dot(self.P,self.My)) + sum([x[0]*(x[1][1]-self.gauge_origin[1]) for x in self.atoms])  
#        self.mu_z = -2*np.trace(np.dot(self.P,self.Mz)) + sum([x[0]*(x[1][2]-self.gauge_origin[2]) for x in self.atoms])  
#        # to debye
#        self.mu_x *= 2.541765
#        self.mu_y *= 2.541765
#        self.mu_z *= 2.541765
#
#    def adj(self,x):
#        return np.conjugate(x).T       
#
#    def comm(self,A,B):
#        return np.dot(A,B) - np.dot(B,A)
#
#    def updateFock(self):
#        self.unOrthoDen()
#        self.buildFock()
#        self.orthoFock()
#
#    def addField(self,time,addstep=False,direction='x'):
#        #if time == 0.0:
#        #    shape = 1.0
#        #else:
#        #    shape = 0.0
#    
#        #shape = np.cos(0.1*time)
#        #shape = 0.000 
#
#        sigma1 = self.stepsize 
#        a = 1.0/(sigma1*np.sqrt(2*np.pi))
#        t1 = 0.0 
#        omega = 0.833191236 
#        periods = 12
#        #shape  = a*np.exp(-((time-t1)**2)/sigma1)#*np.cos(omega*(time-t1))
#        shape = np.piecewise(time, [time < 0, 2*np.pi*periods/omega > time >= 0, time >= 2*np.pi*periods/omega], [0.0,np.sin(omega*time),0.0])
#
#        t2 = self.t2
#        sigma2 = self.stepsize
#        if t2:
#            #shape  += a*np.exp(-((time-t2)**2)/sigma)*np.cos(0.1*omega*(time-t2))
#            shape += (1.0/(sigma2*np.sqrt(2*np.pi)))*np.exp(-((time-t2)**2)/sigma2)#*np.cos(0.1*omega*(time-t2))
#        if addstep:
#            self.shape.append(shape)
#        else:
#            # this will only be true for isotropic GIAO stuffz
#            if direction.lower() == 'x':
#                #self.F += -self.field*shape*(self.Mx + 1j*self.rDipX[0])
#                self.F += -self.field*shape*self.Mx
#            elif direction.lower() == 'y':
#                #self.F += -self.field*shape*(self.My + 1j*self.rDipY[1])
#                self.F += -self.field*shape*self.My
#            elif direction.lower() == 'z':
#                #self.F += -self.field*shape*(self.Mz + 1j*self.rDipZ[2])
#                self.F += -self.field*shape*self.Mz
#            self.orthoFock()
#
#    def buildL(self,direction='x'):
#        # W1 is equivalent to P*F*P
#        #self.orthoFock()
#        #E,CO = np.linalg.eigh(self.FO)
#        #C      = np.dot(self.X,CO)
#        #W1 = np.zeros((self.nbasis,self.nbasis),dtype='complex')
#        #for mu in range(self.nbasis):
#        #    for nu in range(self.nbasis):
#        #        for i in range(self.nocc):
#        #            W1[mu,nu] += E[i]*np.conjugate(C[mu,i])*C[nu,i] 
#        #print E
#
#        if direction.lower() == 'x':
#            dHdB = 1j*self.dhdb[0]
#            dGdB = 1j*self.dgdb[0]
#            dSdB = 1j*self.Sb[0] 
#        elif direction.lower() == 'y':
#            dHdB = 1j*self.dhdb[1]
#            dGdB = 1j*self.dgdb[1]
#            dSdB = 1j*self.Sb[1] 
#        elif direction.lower() == 'z':
#            dHdB = 1j*self.dhdb[2]
#            dGdB = 1j*self.dgdb[2]
#            dSdB = 1j*self.Sb[2] 
# 
#        J = np.einsum('pqrs,sr->pq', dGdB,self.P)
#        K = np.einsum('psqr,sr->pq', dGdB,self.P)
#        G = 2.*J - K
#        F = dHdB + G 
#        self.LN = np.einsum('pq,qp',self.P,F + dHdB) # Correct for GS
#        PFP = np.dot(self.P,np.dot(self.F,self.P)) 
#        W = PFP
#        self.LN -= 2*np.einsum('pq,qp',dSdB,W)
#
#
#    def Magnus2(self,direction='x'):
#        self.orthoDen()
#        self.orthoFock()
#        h = -1j*self.stepsize
#        for idx,time in enumerate(tqdm(self.time)):
#            if direction.lower() == 'x':
#                self.computeDipole()
#                self.dipole.append(np.real(self.mu_x))
#            elif direction.lower() == 'y':
#                self.computeDipole()
#                self.dipole.append(np.real(self.mu_y))
#            elif direction.lower() == 'z':
#                self.computeDipole()
#                self.dipole.append(np.real(self.mu_z))
#            self.addField(time,addstep=True,direction=direction)
#
#            #self.FO, self.PO = updateM4(time,direction,self.stepsize,self.PO,self.F,self.X,self.TwoE,self.Core,self.Mx,self.My,self.Mz,self.field)
#
#            #curFock = np.copy(self.FO)
#
#            curDen  = np.copy(self.PO)
#      
#            self.addField(time + 0.0*self.stepsize,direction=direction)
#            k1 = h*self.FO 
#            U = expm(k1)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#            
#            self.addField(time + 1.0*self.stepsize,direction=direction)
#            L  = 0.5*(k1 + h*self.FO)
#            U  = expm(L)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#            
#            self.unOrthoFock()    
#            self.unOrthoDen()    
#            self.computeEnergy()
#            #print str(np.real(self.energy))
#            #print np.real(np.trace(np.dot(self.P,self.Mz)))
#            self.Energy.append(np.real(self.energy))
#
#    def Magnus4(self,direction='x'):
#        self.orthoDen()
#        self.orthoFock()
#        h = -1j*self.stepsize
#        for idx,time in enumerate(tqdm(self.time)):
#            if direction.lower() == 'x':
#                self.computeDipole()
#                self.dipole.append(np.real(self.mu_x))
#            elif direction.lower() == 'y':
#                self.computeDipole()
#                self.dipole.append(np.real(self.mu_y))
#            elif direction.lower() == 'z':
#                self.computeDipole()
#                self.dipole.append(np.real(self.mu_z))
#            self.addField(time,addstep=True,direction=direction)
#
#            #self.FO, self.PO = updateM4(time,direction,self.stepsize,self.PO,self.F,self.X,self.TwoE,self.Core,self.Mx,self.My,self.Mz,self.field)
#
#            #curFock = np.copy(self.FO)
#
#            curDen  = np.copy(self.PO)
#      
#            self.addField(time + 0.0*self.stepsize,direction=direction)
#            k1 = h*self.FO 
#            Q1 = k1
#            U = expm(0.5*Q1)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#            
#            self.addField(time + 0.5*self.stepsize,direction=direction)
#            k2 = h*self.FO
#            Q2 = k2 - k1
#            U = expm(0.5*Q1 + 0.25*Q2)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#
#            self.addField(time + 0.5*self.stepsize,direction=direction)
#            k3 = h*self.FO
#            Q3 = k3 - k2
#            U = expm(Q1 + Q2)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#
#            self.addField(time + 1.0*self.stepsize,direction=direction)
#            k4 = h*self.FO
#            Q4 = k4 - 2*k2 + k1
#            L  = 0.5*Q1 + 0.25*Q2 + (1/3.)*Q3 - (1/24.)*Q4
#            L += -(1/48.)*self.comm(Q1,Q2)
#            U  = expm(L)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#           
#            self.addField(time + 0.5*self.stepsize,direction=direction)
#            k5 = h*self.FO
#            Q5 = k5 - k2 
#            L  = Q1 + Q2 + (2/3.)*Q3 + (1/6.)*Q4 - (1/6.)*self.comm(Q1,Q2)
#            U  = expm(L)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
# 
#            self.addField(time + 1.0*self.stepsize,direction=direction)
#            k6 = h*self.FO
#            Q6 = k6 -2*k2 + k1
#            L  = Q1 + Q2 + (2/3.)*Q5 + (1/6.)*Q6
#            L += -(1/6.)*self.comm(Q1, (Q2 - Q3 + Q5 + 0.5*Q6))
#
#            U  = expm(L)
#            self.PO = np.dot(U,np.dot(curDen,self.adj(U))) 
#            self.updateFock()
#            
#            self.unOrthoFock()    
#            self.unOrthoDen()    
#            self.computeEnergy()
#            #print str(np.real(self.energy))
#            #print np.real(np.trace(np.dot(self.P,self.Mz)))
#            self.Energy.append(np.real(self.energy))
#         
#
#
#        
#
#if __name__ == '__main__':
#    np.set_printoptions(precision=8,suppress=True)
#    filename = 'h2o2.dat'
#    h2o = Molecule(filename,basis='sto-3g')
#    h2o.SCF()
##    h2o.MP2()
##    np.save('xH.npy',h2o.rH[0])
##    np.save('yH.npy',h2o.rH[1])
##    np.save('zH.npy',h2o.rH[2])
#    np.save('S.npy',h2o.S)
#    np.save('T.npy',h2o.T)
#    np.save('V.npy',h2o.V)
#    np.save('Mx.npy',h2o.Mx)
#    np.save('My.npy',h2o.My)
#    np.save('Mz.npy',h2o.Mz)
#    np.save('Lx.npy',h2o.L[0])
#    np.save('Ly.npy',h2o.L[1])
#    np.save('Lz.npy',h2o.L[2])
#    np.save('dhdbx.npy',h2o.MagMom[0])
#    np.save('dhdby.npy',h2o.MagMom[1])
#    np.save('dhdbz.npy',h2o.MagMom[2])
#    np.savetxt('dhdbx.csv',h2o.MagMom[0],delimiter=',')
#    np.savetxt('dhdby.csv',h2o.MagMom[1],delimiter=',')
#    np.savetxt('dhdbz.csv',h2o.MagMom[2],delimiter=',')
##    np.save('LNx.npy',h2o.Ln[0])
##    np.save('LNy.npy',h2o.Ln[1])
##    np.save('LNz.npy',h2o.Ln[2])
#    np.save('ERI.npy',h2o.TwoE)
##    np.save('GR1x.npy',h2o.GR1[0])
##    np.save('GR1y.npy',h2o.GR1[1])
##    np.save('GR1z.npy',h2o.GR1[2])
##    np.save('GR2x.npy',h2o.GR2[0])
##    np.save('GR2y.npy',h2o.GR2[1])
##    np.save('GR2z.npy',h2o.GR2[2])
#
#    np.save('dgdbx.npy',h2o.dgdb[0])
#    np.save('dgdby.npy',h2o.dgdb[1])
#    np.save('dgdbz.npy',h2o.dgdb[2])
#    np.save('Sbx.npy',h2o.Sb[0])
#    np.save('Sby.npy',h2o.Sb[1])
#    np.save('Sbz.npy',h2o.Sb[2])
#    np.save('nelec.npy',h2o.nelec)
#    np.save('enuc.npy',h2o.nuc_energy)
#    np.save('P.npy',h2o.P)
#    np.save('F.npy',h2o.F)
#
##    print h2o.S
##    print "KINETIC"
##    print h2o.T
##    print "NUCLEAR"
##    print h2o.V
##    print "ANGMOM"
##    print h2o.L
##    print "ANGLON"
##    print h2o.Ln
##    print "LONMOM"
##    print h2o.rH
# #   print "MAGMOM dH/dB"
##    print h2o.MagMom
##    print "S1Mag"
##    print h2o.Sb
#    # dH/dB should be antisymmetric
#    #print np.allclose(h2o.MagMom[0],-h2o.MagMom[0].T)
#    #print np.allclose(h2o.MagMom[1],-h2o.MagMom[1].T)
#    #print np.allclose(h2o.MagMom[2],-h2o.MagMom[2].T)
#   
#
#
