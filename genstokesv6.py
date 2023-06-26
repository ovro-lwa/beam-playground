#!/usr/bin/env python

from glob import glob
from matplotlib import cm
from colormaps import colormaps as cmaps
from time import time

execfile('/scr/mmanders/LWA/scripts/image_sub.py')
execfile('/scr/mmanders/LWA/Stokes/peeled/genstokesv4.py')

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

def applybeam(freqval,IQUVglob='',azelfile='/scr/mmanders/LWA/Stokes/coordazelvals.npy'):
    timeStart = time()
    if IQUVglob == '': # Apply beam to simulated StokesI=1 sky at freqval
        Isim,Qsim,Usim,Vsim = sim_fits(azelfile)
    else:
        stokesfiles  = np.sort(glob(IQUVglob))
        Isim,Iheader = readfits(stokesfiles[0])
        Qsim,Qheader = readfits(stokesfiles[1])
        Usim,Uheader = readfits(stokesfiles[2])
        Vsim,Vheader = readfits(stokesfiles[3])
    azelgrid = np.load(azelfile)
    gridsize = azelgrid.shape[-1]

    beam_L3file  = np.load('/scr/mmanders/LWA/Stokes/peeled/beaminterpolation/beam_L3_'+str(freqval)+'.npz')
    beam_L2file  = np.load('/scr/mmanders/LWA/Stokes/peeled/beaminterpolation/beam_L2_47.npz')
    covalsnonorm = beam_L3file['arr_0'] # co
    thetavalsnonorm = beam_L2file['arr_0']  # theta
    cxvalsnonorm = beam_L3file['arr_1'] # cx
    phivalsnonorm = beam_L2file['arr_1']    # phi
    covals       = covalsnonorm / np.nanmax( np.sqrt( np.abs(covalsnonorm)**2. + np.abs(cxvalsnonorm)**2. ))
    thetavals    = thetavalsnonorm / np.nanmax( np.sqrt( np.abs(thetavalsnonorm)**2. + np.abs(phivalsnonorm)**2. ))

    cxvals       = cxvalsnonorm / np.nanmax( np.sqrt( np.abs(covalsnonorm)**2. + np.abs(cxvalsnonorm)**2. ))
    phivals       = phivalsnonorm / np.nanmax( np.sqrt( np.abs(thetavalsnonorm)**2. + np.abs(phivalsnonorm)**2. ))

    Isky   = np.zeros((gridsize,gridsize),dtype=np.complex)
    Qsky   = np.copy(Isky)
    Usky   = np.copy(Isky)
    Vsky   = np.copy(Isky)
    Inosky = np.copy(Isky)
    Qnosky = np.copy(Isky)
    Unosky = np.copy(Isky)
    Vnosky = np.copy(Vsky)

    Bsim   = np.array([ [Isim-Qsim    , Usim-1.j*Vsim],
                        [Usim+1.j*Vsim, Isim+Qsim    ] ])
    # fix_invalid necessary so that np.linalg.inv doesn't break on nans or return Error: Singular Matrix
    Ejones = np.array([ [np.ma.fix_invalid(covals,fill_value=1).data           , np.ma.fix_invalid(cxvals,fill_value=0).data          ],
                        [np.ma.fix_invalid(-np.rot90(cxvals),fill_value=0).data, np.ma.fix_invalid(np.rot90(covals),fill_value=1).data] ])
    Ejones2 = np.array([ [np.ma.fix_invalid(thetavals,fill_value=1).data           , np.ma.fix_invalid(phivals,fill_value=0).data          ],
                        [np.ma.fix_invalid(np.rot90(thetavals),fill_value=0).data, np.ma.fix_invalid(np.rot90(phivals),fill_value=1).data] ])
    # reshape so that np.linalg.inv broadcasts correctly over multidimensional matrices
    Bsimreshape   = Bsim.transpose(2,3,0,1)
    Ejonesreshape = Ejones.transpose(2,3,0,1)
    Ejones2reshape = Ejones2.transpose(2,3,0,1)
    if IQUVglob == '':
        Bapp = np.matmul(Ejonesreshape, np.matmul(Bsimreshape, Ejonesreshape.conj().transpose(0,1,3,2)) )
    else:
        Bapp = np.matmul(np.linalg.inv(Ejonesreshape), np.matmul(Bsimreshape, np.linalg.inv(Ejonesreshape.conj().transpose(0,1,3,2)) ) )
    Inosky = 0.5 * ( np.abs(covals)**2.            + np.abs(cxvals)**2.           \
                   + np.abs(-np.rot90(cxvals))**2. + np.abs(np.rot90(covals))**2. )
    Qnosky = 0.5 * ( np.abs(covals)**2.            + np.abs(cxvals)**2.           \
                   - np.abs(-np.rot90(cxvals))**2. - np.abs(np.rot90(covals))**2. )
    Unosky = 0.5 * 2 * ( np.real(cxvals * np.conj( np.rot90(covals))) \
                       + np.real(covals * np.conj(-np.rot90(cxvals))) )
    Vnosky = 0.5 * 2 * ( np.imag(np.rot90(covals) * np.conj(          cxvals))  \
                       - np.imag(covals           * np.conj(-np.rot90(cxvals))) )
    Isky   =     0.5*(Bapp[:,:,0,0] + Bapp[:,:,1,1]) * Inosky
    Qsky   =     0.5*(Bapp[:,:,0,0] - Bapp[:,:,1,1]) * Inosky
    Usky   =     0.5*(Bapp[:,:,0,1] + Bapp[:,:,1,0]) * Inosky
    Vsky   = 1.j*0.5*(Bapp[:,:,0,1] - Bapp[:,:,1,0]) * Inosky

    print 'Runtime: %0.1f seconds' % (time() - timeStart)

    if not IQUVglob == '':
        writefits(np.real(Isky),Iheader,os.path.splitext(np.sort(glob(IQUVglob))[0])[0]+'_beamapplied.fits')
        writefits(np.real(Qsky),Qheader,os.path.splitext(np.sort(glob(IQUVglob))[1])[0]+'_beamapplied.fits')
        writefits(np.real(Usky),Uheader,os.path.splitext(np.sort(glob(IQUVglob))[2])[0]+'_beamapplied.fits')
        writefits(np.real(Vsky),Vheader,os.path.splitext(np.sort(glob(IQUVglob))[3])[0]+'_beamapplied.fits')
    else:
        np.savez('/scr/mmanders/LWA/modules/beam/beamIQUV_'+str(freqval),I=np.real(Inosky),Q=np.real(Qnosky),U=np.real(Unosky),V=np.real(Vnosky))
        if not os.path.exists('/scr/mmanders/LWA/modules/beam/azelgrid.npy'):
            np.save('/scr/mmanders/LWA/modules/beam/azelgrid',azelgrid*180./np.pi)

    # plots --> np.rot90 so N is up, E is left
    # beam co and cx
    p.close('all')
    p.figure(figsize=(6,9))
    p.subplot(211)
    p.imshow(np.rot90(np.abs(covals)),cmap=cmaps.inferno)
    p.title('Co')
    p.colorbar()
    p.subplot(212)
    p.imshow(np.rot90(np.abs(cxvals)),cmap=cmaps.inferno)
    p.title('Cx')
    p.colorbar()
    # beam co and cx real and imaginary
    p.figure(figsize=(15,9))
    p.subplot(221)
    p.imshow(np.rot90(np.real(covals)),cmap=cmaps.inferno)
    p.title('Re(Co)')
    p.colorbar()
    p.subplot(222)
    p.imshow(np.rot90(np.imag(covals)),cmap=cmaps.inferno)
    p.title('Im(Co)')
    p.colorbar()
    p.subplot(223)
    p.imshow(np.rot90(np.real(cxvals)),cmap=cmaps.inferno)
    p.title('Re(Cx)')
    p.colorbar()
    p.subplot(224)
    p.imshow(np.rot90(np.imag(cxvals)),cmap=cmaps.inferno)
    p.title('Im(Cx)')
    p.colorbar()
    p.savefig('./Ludwig3beam_reim.png', bbox_inches='tight')
    # azimuth and elevation
    p.figure(figsize=(6,9))
    p.subplot(211)
    p.imshow(np.rot90(azelgrid[0,:,:] * 180./np.pi),cmap=cmaps.inferno)
    p.title('azimuth')
    p.colorbar()
    p.subplot(212)
    p.imshow(np.rot90(azelgrid[1,:,:] * 180./np.pi),cmap=cmaps.inferno)
    p.title('elevation')
    p.colorbar()

    # beam theta and phi
    p.figure(figsize=(15,9))
    p.subplot(221)
    p.imshow(np.rot90(np.real(thetavals)),cmap=cmaps.inferno)
    p.title(r'Re($\theta$)')
    p.colorbar()
    p.subplot(222)
    p.imshow(np.rot90(np.imag(thetavals)),cmap=cmaps.inferno)
    p.title(r'Im($\theta$)')
    p.colorbar()
    p.subplot(223)
    p.imshow(np.rot90(np.real(phivals)),cmap=cmaps.inferno)
    p.title(r'Re($\phi$)')
    p.colorbar()
    p.subplot(224)
    p.imshow(np.rot90(np.imag(phivals)),cmap=cmaps.inferno)
    p.title(r'Im($\phi$)')
    p.colorbar()
    p.savefig('./Ludwig2beam_reim.png', bbox_inches='tight')

    # Sky with beam applied
    if IQUVglob == '':
        vminI = 0.0
        vmaxI = 1.0
        vminQ = -0.15
        vmaxQ = 0.15
        vminU = -0.15
        vmaxU = 0.15
        vminV = -0.010
        vmaxV = 0.010
        cmap  = cmaps.inferno
    else:
        vminI = -100
        vmaxI = 200
        vminQ = -20
        vmaxQ = 20
        vminU = -20
        vmaxU = 20
        vminV = -10
        vmaxV = 10
        cmap  = 'gray'
    p.figure(figsize=(15,9))
    p.subplot(221)
    p.imshow(np.rot90(np.real(Isky)),vmin=vminI,vmax=vmaxI,cmap=cmap)
    p.title('I')
    p.colorbar()
    p.subplot(222)
    p.imshow(np.rot90(np.real(Qsky)),vmin=vminQ,vmax=vmaxQ,cmap=cmap)
    p.title('Q')
    p.colorbar()
    p.subplot(223)
    p.imshow(np.rot90(np.real(Usky)),vmin=vminU,vmax=vmaxU,cmap=cmap)
    p.title('U')
    p.colorbar()
    p.subplot(224)
    p.imshow(np.rot90(np.real(Vsky)),vmin=vminV,vmax=vmaxV,cmap=cmap)
    p.title('V')
    p.colorbar()
    # Beam
    p.figure(figsize=(15,9))
    p.subplot(221)
    p.imshow(np.rot90(np.real(Inosky)),vmin=0.0,vmax=1.0,cmap=cmaps.inferno)
    p.title('I')
    p.colorbar()
    p.subplot(222)
    p.imshow(np.rot90(np.real(Qnosky)),vmin=-0.15,vmax=0.15,cmap=cmaps.inferno)
    p.title('Q')
    p.colorbar()
    p.subplot(223)
    p.imshow(np.rot90(np.real(Unosky)),vmin=-0.15,vmax=0.15,cmap=cmaps.inferno)
    p.title('U')
    p.colorbar()
    p.subplot(224)
    p.imshow(np.rot90(np.real(Vnosky)),vmin=-0.010,vmax=0.010,cmap=cmaps.inferno)
    p.title('V')
    p.colorbar()
    pdb.set_trace()
    p.show()
