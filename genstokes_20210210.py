"""
By Marin Anderson
Not sure if this is what's actually used but here's a start.

"""
import numpy as np
from scipy.interpolate import griddata as gd

# DW's beam simulation
# theta,phi = zenith,azimuth [degrees] for single quadrant
# freq [MHz]
beamfile = '/lustre/mmanders/rycbar/Stokes/DW_beamquadranttable20151110.txt'
theta,phi,freq,coreal,coimag,cxreal,cximag,thetareal,thetaimag,phireal,phiimag = \
    np.genfromtxt(beamfile, dtype='float', skip_header=7,unpack=True)
    
frqvals     = np.unique(freq)   # frqvals = np.array([20., 30., 40., 50., 60., 70., 80.])
frqinterval = int(len(freq)/len(frqvals))   # frqinterval = 361

corealnew = coreal.reshape(len(frqvals), frqinterval)   # corealnew.shape = (7, 361)
coimagnew = coimag.reshape(len(frqvals), frqinterval)
cxrealnew = cxreal.reshape(len(frqvals), frqinterval)
cximagnew = cximag.reshape(len(frqvals), frqinterval)

thetanew  = theta.reshape(len(frqvals), frqinterval)
phinew    = phi.reshape(len(frqvals), frqinterval)
freqnew   = freq.reshape(len(frqvals), frqinterval)

# Fill out remaining four quadrants
thetafull = np.concatenate(( thetanew, thetanew, thetanew, thetanew), axis=1)
#phifull   = np.concatenate(( phinew, phinew[::-1]+90, phinew[::-1]-90, phinew-180), axis=1)
phifull   = np.concatenate(( phinew, np.flip(phinew)+90, np.flip(phinew)-90, phinew-180), axis=1)
freqfull  = np.concatenate(( freqnew, freqnew, freqnew, freqnew), axis=1)
lfull     = np.sin(np.radians(thetafull)) * np.sin(np.radians(phifull))
mfull     = np.sin(np.radians(thetafull)) * np.cos(np.radians(phifull))

corealfull = np.concatenate(( corealnew, corealnew, corealnew, corealnew), axis=1)
coimagfull = np.concatenate(( coimagnew, coimagnew, coimagnew, coimagnew), axis=1)
cxrealfull = np.concatenate(( cxrealnew, -cxrealnew, -cxrealnew, cxrealnew), axis=1)
cximagfull = np.concatenate(( cximagnew, -cximagnew, -cximagnew, cximagnew), axis=1)
#
corealfull_rot90  = np.concatenate(( corealnew, corealnew, corealnew, corealnew), axis=1)
coimagfull_rot90  = np.concatenate(( coimagnew, coimagnew, coimagnew, coimagnew), axis=1)
cxrealfull_nrot90 = -np.concatenate(( cxrealnew, -cxrealnew, -cxrealnew, cxrealnew), axis=1)
cximagfull_nrot90 = -np.concatenate(( cximagnew, -cximagnew, -cximagnew, cximagnew), axis=1)

cofull = corealfull + 1.j*coimagfull
cxfull = cxrealfull + 1.j*cximagfull
#
cofull_rot90 = corealfull_rot90 + 1.j*coimagfull_rot90
cxfull_nrot90 = cxrealfull_nrot90 + 1.j*cximagfull_nrot90

# normalize to 1 at beam center
norms = 1/np.nanmax( np.sqrt( np.abs(cofull)**2. + np.abs(cxfull)**2. ), axis=1)
for ind,norm in enumerate(norms):
    cofull[ind,:] *= norm
    cxfull[ind,:] *= norm
    cofull_rot90[ind,:] *= norm
    cxfull_nrot90[ind,:] *= norm
#cofull_rot90  = np.rot90(cofull)
#cxfull_nrot90 = -np.rot90(cxfull)
np.savez('/lustre/mmanders/rycbar/Stokes/peeled/beamLudwig3rd', cofull=cofull, cxfull=cxfull,
    cofull_rot90=cofull_rot90, cxfull_nrot90=cxfull_nrot90, lfull=lfull[0,:], mfull=mfull[0,:], 
    frqvals=frqvals, thetafull=thetafull[0,:], phifull=phifull[0,:])
#    freqfull=freqfull)
import pdb
pdb.set_trace()

# return Jones matrix for specified set of l,m coordinates and frequency
def beamfunc(lval,mval,freqval):
    coval = gd( (lfull.ravel(), mfull.ravel(), freqfull.ravel()), \
                cofull.ravel(), (lval, mval, freqval), method='linear')
    cxval = gd( (lfull.ravel(), mfull.ravel(), freqfull.ravel()), \
                cxfull.ravel(), (lval, mval, freqval), method='linear')
    corot90val  = gd( (lfull.ravel(), mfull.ravel(), freqfull.ravel()), \
                  cofull_rot90.ravel(), (lval, mval, freqval), method='linear')
    cxnrot90val = gd( (lfull.ravel(), mfull.ravel(), freqfull.ravel()), \
                  cxfull_nrot90.ravel(), (lval, mval, freqval), method='linear')
    Jonesmat = np.array([ [coval,       cxval     ], 
                          [cxnrot90val, corot90val] ])
    return Jonesmat
