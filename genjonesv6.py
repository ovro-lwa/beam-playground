#!/usr/bin/env python

from glob import glob
from matplotlib import cm
from colormaps import colormaps as cmaps
from time import time
import numpy as np
import pdb

#execfile('/scr/mmanders/LWA/scripts/image_sub.py')
#execfile('/scr/mmanders/LWA/Stokes/peeled/genstokesv4.py')

def sim_fits(azelgridfile):
    azelgrid = np.load(azelgridfile)
    gridsize = azelgrid.shape[-1]

    Isim = np.zeros((gridsize,gridsize))
    Qsim = np.copy(Isim)
    Usim = np.copy(Isim)
    Vsim = np.copy(Isim)
    # Unpolarized sky = 1 Jy (StokesI=1, StokesQ,U,V=0)
    Isim = np.where(np.isnan(azelgrid[0,:,:]),0,1)

    return Isim,Qsim,Usim,Vsim

def applybeam(freqval,azelfile='/lustre/mmanders/rycbar/Stokes/coordazelvals.npy'):
    azelgrid = np.load(azelfile)
    gridsize = azelgrid.shape[-1]

    beam_L3file  = np.load('/lustre/mmanders/rycbar/Stokes/peeled/beaminterpolation/beam_L3_'+str(freqval)+'.npz')
    covalsnonorm = beam_L3file['arr_0'] # co
    cxvalsnonorm = beam_L3file['arr_1'] # cx
    #covals       = covalsnonorm / np.nanmax( np.sqrt( np.abs(covalsnonorm)**2. + np.abs(cxvalsnonorm)**2. ))
    #cxvals       = cxvalsnonorm / np.nanmax( np.sqrt( np.abs(covalsnonorm)**2. + np.abs(cxvalsnonorm)**2. ))
    #pdb.set_trace()
    covals = covalsnonorm
    cxvals = cxvalsnonorm

    Bsim   = np.array([ [covals, cxvals],
                        [-np.rot90(cxvals), np.rot90(covals) ] ])
    np.savez('/lustre/mmanders/LWA/modules/beam/beamJones_'+str(freqval),J=Bsim)

