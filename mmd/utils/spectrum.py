import numpy as np

"""Contains some routines to do the (Pade approximant) Fourier transform
   as well as some peak-finding routines, useful for post processing a
   real-time calculation
"""

def genSpectra(time,dipole,signal):

    fw, frequency = pade(time,dipole)
    fw_sig, frequency = pade(time,signal,alternate=True)

    fw_re = np.real(fw)
    fw_im = np.imag(fw)
    fw_abs = fw_re**2 + fw_im**2

    #spectra = (fw_re*17.32)/(np.pi*field*damp_const)
    #spectra = (fw_re*17.32*514.220652)/(np.pi*field*damp_const)
    #numerator = np.imag((fw*np.conjugate(fw_sig)))
    numerator = np.imag(fw_abs*np.conjugate(fw_sig))
    #numerator = np.abs((fw*np.conjugate(fw_sig)))
    #numerator = np.abs(fw)
    denominator = np.real(np.conjugate(fw_sig)*fw_sig)
    #denominator = 1.0 
    spectra = ((4.0*27.21138602*2*frequency*np.pi*(numerator))/(3.0*137.036*denominator))
    spectra *= 1.0/100.0
    #plt.plot(frequency*27.2114,fourier)
    #plt.show()
    return frequency, spectra

def pade(time,dipole):
    damp_const = 100.0
    dipole = np.asarray(dipole) - dipole[0]
      
    stepsize = time[1] - time[0]
    #print dipole
    damp = np.exp(-(stepsize*np.arange(len(dipole)))/float(damp_const))
    dipole *= damp
    M = len(dipole)
    N = int(np.floor(M / 2))

    #print "N = ", N
    num_pts = 20000
    if N > num_pts:
        N = num_pts
    #print "Trimmed points to: ", N

    # G and d are (N-1) x (N-1)
    # d[k] = -dipole[N+k] for k in range(1,N)
    d = -dipole[N+1:2*N]

    try:
        from scipy.linalg import toeplitz, solve_toeplitz
    except ImportError:
        print("You'll need SciPy version >= 0.17.0")

    try:
        # Instead, form G = (c,r) as toeplitz
        #c = dipole[N:2*N-1]
        #r = np.hstack((dipole[1],dipole[N-1:1:-1]))
        b = solve_toeplitz((dipole[N:2*N-1],\
            np.hstack((dipole[1],dipole[N-1:1:-1]))),d,check_finite=False)
    except np.linalg.linalg.LinAlgError:  
        # OLD CODE: sometimes more stable
        # G[k,m] = dipole[N - m + k] for m,k in range(1,N)
        G = dipole[N + np.arange(1,N)[:,None] - np.arange(1,N)]
        b = np.linalg.solve(G,d)

    # Now make b Nx1 where b0 = 1
    b = np.hstack((1,b))

    # b[m]*dipole[k-m] for k in range(0,N), for m in range(k)
    a = np.dot(np.tril(toeplitz(dipole[0:N])),b)
    p = np.poly1d(a)
    q = np.poly1d(b)

    # If you want energies greater than 2*27.2114 eV, you'll need to change
    # the default frequency range to something greater.

    #frequency = np.arange(0.00,2.0,0.00005)
    frequency = np.arange(0.3,0.75,0.0002)

    W = np.exp(-1j*frequency*stepsize)

    fw = p(W)/q(W)

    return fw, frequency 

def peaks(spectra,frequency,number=3,thresh=0.01):
        """ Return the peaks from the Fourier transform
            Variables:
            number:     integer. number of peaks to print.
            thresh:     float. Threshhold intensity for printing.

            Returns: Energy (eV), Intensity (depends on type of spectra)
        """

        from scipy.signal import argrelextrema as pks
        # find all peak indices [idx], and remove those below thresh [jdx]
        idx = pks(np.abs(spectra),np.greater,order=3)
        jdx = np.where((np.abs(spectra[idx]) >= thresh))
        kdx = idx[0][jdx[0]] # indices of peaks matching criteria
        if number > len(kdx):
            number = len(kdx)
        print("First "+str(number)+" peaks (eV) found: ")
        for i in xrange(number):
            print("{0:.4f}".format(frequency[kdx][i]*27.2114),
                  "{0:.4f}".format(spectra[kdx][i]))

            

            

