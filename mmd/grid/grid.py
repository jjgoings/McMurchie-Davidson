import numpy as np
from mmd.grid.atomic_grid import atomic_grid
from mmd.grid.becke import becke_reweight_atoms

class Grid(object):
    def __init__(self,atoms,**kwargs):
        agrids = [atomic_grid(atom,**kwargs) for atom in atoms]
        becke_reweight_atoms(atoms,agrids)
        self.points = np.vstack([agrid.points for agrid in agrids])
        self.npts,sb4 = self.points.shape
        assert sb4==4
        return

    def setbfamps(self,bfs):
        nbf = len(bfs)
        self.bfamps = np.zeros((self.npts,nbf),'d')
        for j,bf in enumerate(bfs):
            for i,(x,y,z,w) in enumerate(self.points):
                self.bfamps[i,j] = bf(x,y,z)
        return

    def getdens(self,D):
        return 2*np.einsum('pI,pJ,IJ->p',self.bfamps,self.bfamps,D)

    def getdens_interpolated(self,D,bbox,npts=50):
        from scipy.interpolate import griddata
        xmin,xmax,ymin,ymax,zmin,zmax = bbox
        xi,yi,zi = np.mgrid[xmin:xmax:(npts*1j),ymin:ymax:(npts*1j),zmin:zmax:(npts*1j)]
        rho = self.getdens(D)
        return griddata(self.points[:,:3],rho,(xi,yi,zi))

