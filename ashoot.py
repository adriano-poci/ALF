# -*- coding: utf-8 -*-
"""
    ashoot.py
    Adriano Poci
    Durham University
    202

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module contains functions for troubleshooting the various alf fits

    Authors
    -------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:   19 October 2022
"""
from __future__ import print_function, division

# Core modules
import os, io, re
import sys
import traceback
import pdb
import pathlib as plp
import numpy as np
from glob import glob
from copy import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LogNorm
from scipy import stats as scistat
from scipy import interpolate as sint
from skimage import filters as skilters
from tqdm import tqdm
from functools import partial
from astropy.io import fits as pf
from astropy import units as uts
from scipy.special import gammaln
import itertools
from svo_filters import svo

# Custom modules
from alf.Alf import Alf
import alf.alf_MUSE as am
import alf.alf_utils as au
from dynamics.IFU.Galaxy import Redshift, Schwarzschild, Mge, pieceIMF,\
    Photometry
from dynamics.IFU import Constants
from dynamics.IFU.Functions import Plot, Geometric, Mathematical
from cythonModules import C_utils as Cu
from cythonModules import C_GHKinematics as Cgh

# Dynamics modules
from plotbin.display_pixels import display_pixels as dispp
from plotbin.symmetrize_velfield import symmetrize_velfield as svf
from plotbin.plot_velfield import plot_velfield as pvf
from plotbin.sauron_colormap import register_sauron_colormap as srsc
import mgefit.mge_fit_1d as mf1

curdir = plp.Path(__file__).parent
dDir = au._ddir()

UTT = Constants.Units()
UTS = Constants.UnitStr()
CTS = Constants.Constants()
POT = Plot()
GEO = Geometric()
MTH = Mathematical()
SHW = Schwarzschild()
PHT = Photometry()


def NFMvWFM():

    gfs = curdir.parent/'muse'/'obsData'/'SNL1.xz'
    gal = au.Load.lzma(gfs)
    eps = gal['sMGE'].epsE

    aKeys = ['FeH', 'a', 'Na', 'Ti', 'Mg']
    colmar = mc_list = list(itertools.product(
        plt.rcParams['axes.prop_cycle'].by_key()['color'], ['X', 'p', 'D', 'P',
        '*', 'h', 'H', 'o', 'v', '+', 'x', '^', '<', '>', '1', '2', '3', '4',
        '8', 's'][:len(aKeys)+1]))
    main = plt.figure(figsize=plt.figaspect(0.6)*1.3)
    ax = main.gca()
    counter = 0

    nDir = curdir/'SNL1NFM'
    nkfs  = nDir/'kins_SN70_full.xz'
    nsffs = nDir/'pops_SN70_full.xz'
    nSFH = au.Load.lzma(nsffs)
    nKIN = au.Load.lzma(nkfs)

    nxbin, nybin = nKIN['x'], nKIN['y']
    nrade = np.sqrt(nxbin**2 + (nybin/eps)**2)
    nrore = np.argsort(nrade)
    nrade = np.ma.masked_invalid(np.log10(nrade[nrore]))
    nmedBins = np.linspace(np.ma.min(nrade), np.ma.max(nrade), 11)
    ndelta = nmedBins[1:] - nmedBins[:-1]
    nidx = np.digitize(nrade, nmedBins[1:])
    npBins = nmedBins[1:] - ndelta/2

    for ai, key in enumerate(aKeys):
        nabund = nSFH['abundances'][key][nrore]
        nlabel = nSFH['abundances']['labels'][list(
            nSFH['abundances']['keys']).index(key)]
        ncol, nmkr = colmar[counter]
        named = np.array([np.ma.median(nabund[nidx==k]) for k in range(10)])
        naerr = np.array([np.ma.std(nabund[nidx==k]) for k in range(10)])
        ax.errorbar(npBins, named, yerr=naerr, marker=nmkr, mfc=ncol,
            label=nlabel, mew=0.75, mec='k', ecolor=ncol, ms=12, c=ncol,
            zorder=len(aKeys)-ai)
        counter += 1
    nIMF1 = np.ma.masked_equal(nSFH['IMF']['1'], 1)
    nIMF2 = np.ma.masked_equal(nSFH['IMF']['2'], 1)
    im1 = np.ma.getmaskarray(nIMF1)
    im2 = np.ma.getmaskarray(nIMF2)
    nIMF1[im1] = nIMF1.data[im1] + ((np.random.ranf()-0.5)*1e-3)
    nIMF2[im2] = nIMF2.data[im2] + ((np.random.ranf()-0.5)*1e-3)
    nimfs = [pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.0),
        slopes=(x1, x2, 2.3)) for (x1, x2) in zip(nIMF1, nIMF2)]
    nxiTop = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=0.5)[0], nimfs)))
    nxiBot = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=1.0)[0], nimfs)))
    nxi = (nxiTop/nxiBot)[nrore]
    nxia = np.array([np.ma.median(nxi[nidx==k]) for k in range(10)])
    nxie = np.array([np.ma.std(nxi[nidx==k]) for k in range(10)])
    ncol, nmkr = colmar[counter]
    ax.errorbar(npBins, nxia, yerr=nxie, marker=nmkr, mfc=ncol,
            label=r'$\xi$', mew=0.75, mec='k', ecolor=ncol, ms=12, c=ncol,
            zorder=100)
    counter += 1

    wDir = curdir/'SNL1'
    wkfs  = wDir/'kins_SN50_full.xz'
    wsffs = wDir/'pops_SN50_full.xz'
    wSFH = au.Load.lzma(wsffs)
    wKIN = au.Load.lzma(wkfs)

    wxbin, wybin = wKIN['x'], wKIN['y']
    wrade = np.sqrt(wxbin**2 + (wybin/eps)**2)
    wrore = np.argsort(wrade)
    wrade = np.ma.masked_invalid(np.log10(wrade[wrore]))
    wmedBins = np.linspace(np.ma.min(wrade), np.ma.max(wrade), 11)
    wdelta = wmedBins[1:] - wmedBins[:-1]
    widx = np.digitize(wrade, wmedBins[1:])
    wpBins = wmedBins[1:] - wdelta/2

    for ai, key in enumerate(aKeys):
        wabund = wSFH['abundances'][key][wrore]
        wlabel = wSFH['abundances']['labels'][list(
            wSFH['abundances']['keys']).index(key)]
        wcol, wmkr = colmar[counter]
        wamed = np.array([np.ma.median(wabund[widx==k]) for k in range(10)])
        waerr = np.array([np.ma.std(wabund[widx==k]) for k in range(10)])
        ax.errorbar(wpBins, wamed, yerr=waerr, marker=wmkr, mfc=wcol,
            label=wlabel, mew=0.75, mec='k', ecolor=wcol, ms=12, c=wcol,
            zorder=len(aKeys)-ai)
        counter += 1
    wIMF1 = np.ma.masked_equal(wSFH['IMF']['1'], 1)
    wIMF2 = np.ma.masked_equal(wSFH['IMF']['2'], 1)
    im1 = np.ma.getmaskarray(wIMF1)
    im2 = np.ma.getmaskarray(wIMF2)
    wIMF1[im1] = wIMF1.data[im1] + ((np.random.ranf()-0.5)*1e-3)
    wIMF2[im2] = wIMF2.data[im2] + ((np.random.ranf()-0.5)*1e-3)
    wimfs = [pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.0),
        slopes=(x1, x2, 2.3)) for (x1, x2) in zip(wIMF1, wIMF2)]
    wxiTop = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=0.5)[0], wimfs)))
    wxiBot = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=1.0)[0], wimfs)))
    wxi = (wxiTop/wxiBot)[wrore]
    wxia = np.array([np.ma.median(wxi[widx==k]) for k in range(10)])
    wxie = np.array([np.ma.std(wxi[widx==k]) for k in range(10)])
    wcol, wmkr = colmar[counter]
    ax.errorbar(wpBins, wxia, yerr=wxie, marker=wmkr, mfc=wcol,
            label=r'$\xi$', mew=0.75, mec='k', ecolor=wcol, ms=12, c=wcol,
            zorder=100)
    
    ax.legend()
    ax.set_xlim(right=ax.get_xlim()[-1]*1.2)
    ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
    ax.set_ylabel(r'${\rm Abundance}\ [{\rm dex}]$')
    main.savefig(curdir/'NFMvWFM.pdf', format='pdf')
    plt.close('all')

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.gca()
    counter = 0
    nML = nSFH['ML']['F814W'][nrore]
    nMLa = np.array([np.ma.median(nML[nidx==k]) for k in range(10)])
    nMLe = np.array([np.ma.std(nML[nidx==k]) for k in range(10)])
    ncol, nmkr = colmar[counter]
    ax.errorbar(npBins, nMLa, yerr=nMLe, marker=nmkr, mfc=ncol,
        label=r'$M/L_{F814W}$', mew=0.75, mec='k', ecolor=ncol, ms=12, c=ncol,
        zorder=100)
    counter += 10
    wML = wSFH['ML']['F814W'][wrore]
    wMLa = np.array([np.ma.median(wML[widx==k]) for k in range(10)])
    wMLe = np.array([np.ma.std(wML[widx==k]) for k in range(10)])
    wcol, wmkr = colmar[counter]
    ax.errorbar(wpBins, wMLa, yerr=wMLe, marker=wmkr, mfc=wcol,
        label=r'$M/L_{F814W}$', mew=0.75, mec='k', ecolor=wcol, ms=12, c=wcol,
        zorder=100)
    ax.legend()
    ax.set_xlim(right=ax.get_xlim()[-1]*1.2)
    ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
    ax.set_ylabel(r'${\rm M/L}$')
    fig.savefig(curdir/'NFMvWFM_ML.pdf', format='pdf')
    plt.close('all')


def specCal():
    wfm = next((dDir/'MUSECubes').glob(f"*SNL1_WFM_DATACUBE*.fits"))
    wdu = pf.open(wfm)
    wdd = wdu[1].header
    wnL, wnY, wnX = wdd['NAXIS3'], wdd['NAXIS2'], wdd['NAXIS1']
    wxOrg, wyOrg = GEO.genPix(np.arange(wnX), np.arange(wnY))
    wpixs = np.abs(wdd['CD1_1']) * 60. * 60.
    wLamb = wdd['CRVAL3']+np.arange(wnL)*wdd['CD3_3']
    wCube = np.ma.masked_invalid(wdu[1].data)
    wFlux = np.ma.sum(wCube, axis=0)
    wdu.close()
    wxc, wyc, _, _ = PHT.findCentre(wFlux, 'SNL1')
    wxp = (wxOrg-wxc)*wpixs
    wyp = (wyOrg-wyc)*wpixs
    wrp = np.sqrt(wxp**2 + wyp**2)
    
    nfm = next((dDir/'MUSECubes').glob(f"*SNL1_NFM_DATACUBE*.fits"))
    ndu = pf.open(nfm)
    ndd = ndu[0].header
    nnL, nnY, nnX = ndd['NAXIS3'], ndd['NAXIS2'], ndd['NAXIS1']
    nxOrg, nyOrg = GEO.genPix(np.arange(nnX), np.arange(nnY))
    npixs = np.abs(ndd['CD1_1']) * 60. * 60.
    nLamb = ndd['CRVAL3']+np.arange(nnL)*ndd['CD3_3']
    nCube = np.ma.masked_invalid(ndu[0].data)
    nFlux = np.ma.sum(nCube, axis=0)
    ndu.close()
    nxc, nyc, _, _ = PHT.findCentre(nFlux, 'SNL1_NFM')
    nxp = (nxOrg-nxc)*npixs
    nyp = (nyOrg-nyc)*npixs
    nrp = np.sqrt(nxp**2 + nyp**2)

    wrFlux = wFlux.ravel()
    nrFlux = nFlux.ravel()
    
    binNFM = scistat.binned_statistic_2d(nxp, nyp, nrFlux, statistic=np.ma.sum, bins=[np.append(np.unique(wxp), np.max(wxp)+wpixs), np.append(np.unique(wyp), np.max(wyp)+wpixs)])
    bnFlux = binNFM.statistic.T
    bnBN = binNFM.binnumber

    bnCube = (np.ma.ones((nnL, wnY, wnX))*np.nan).reshape(nnL, -1)
    fwCube = wCube.reshape(wnL, -1)
    fnCube = nCube.reshape(nnL, -1)
    for jy in range(bnCube.shape[-1]):
        wwxy = np.where(bnBN == jy)[0]
        if np.any(wwxy):
            speci = np.ma.sum(fnCube[:, wwxy], axis=1)
            bnCube[:, jy] = speci
    bnCube = np.ma.masked_invalid(bnCube)

    plt.clf(); plt.scatter(wrp, wrFlux, s=1, label='WFM'); plt.scatter(nrp, nrFlux, s=1, label='NFM'); plt.legend(); plt.savefig('fluxProfiles')
    plt.clf(); plt.scatter(wrp, wrFlux, s=1, label='WFM'); plt.scatter(wrp, bnFlux, s=1, label='Binned NFM'); plt.legend(); plt.savefig('binFluxProfiles')
    plt.clf(); plt.scatter(np.log10(wrp), np.log10(wrFlux), s=1, label='WFM'); plt.scatter(np.log10(nrp), np.log10(nrFlux), s=1, label='NFM'); plt.legend(); plt.savefig('logFluxProfiles')
    plt.clf(); plt.scatter(np.log10(wrp), np.log10(wrFlux), s=1, label='WFM'); plt.scatter(np.log10(wrp), np.log10(bnFlux), s=1, label='Binned NFM'); plt.legend(); plt.savefig('logBinFluxProfiles')

    crads = np.arange(1.2, 3.8, 0.2)
    cmap = mpl.colormaps['IDLSTDGAMMA']
    figR = plt.figure()
    figN = plt.figure()
    figW = plt.figure()
    axR = figR.gca()
    axN = figN.gca()
    axW = figW.gca()
    for rad in crads:
        r1 = np.where(np.isclose(wrp, rad, atol=wpixs))[0]
        r2 = np.where(np.isclose(nrp, rad, atol=wpixs))[0]
        dwSpec = (np.ma.sum(fwCube[:, r1], axis=1)/np.ma.median(np.ma.sum(fwCube[:, r1], axis=1)))
        dnSpec = (np.ma.sum(fnCube[:, r2], axis=1)/np.ma.median(np.ma.sum(fnCube[:, r2], axis=1)))
        from spectres import spectres
        from scipy.signal import firwin, oaconvolve
        from scipy.interpolate import interp1d
        rdwSpec = spectres(nLamb, wLamb, dwSpec)
        # filt = oaconvolve(rdwSpec/dnSpec, firwin(100, 0.01), mode='same')
        filt = oaconvolve(rdwSpec[1:]/dnSpec[1:], firwin(100, 0.01), mode='same')
        filtN = oaconvolve(dnSpec[1:], firwin(100, 0.01), mode='same')
        filtW = oaconvolve(rdwSpec[1:], firwin(100, 0.01), mode='same')
        axR.plot(nLamb[1:], filt, lw=0.75, c=cmap(rad/np.max(crads)), label=f"{rad:.1f}")
        axN.plot(nLamb[1:], filtN, lw=0.75, c=cmap(rad/np.max(crads)))
        axW.plot(nLamb[1:], filtW, lw=0.75, c=cmap(rad/np.max(crads)))
    # plt.clf(); plt.plot(twLamb, dwSpec[wl], lw=0.25); plt.plot(tnLamb, dnSpec[nl], lw=0.25); plt.plot(tnLamb, filt[nl], lw=0.75); plt.savefig('sameSpec')
    # uncont = rdwSpec/filt
    axR.legend(ncols=3)
    figR.savefig('filts')
    figN.savefig('nfmContinuum')
    figW.savefig('wfmContinuum')
    pdb.set_trace()
    uncont = rdwSpec[1:]/filt
    plt.clf(); plt.plot(wLamb, dwSpec, lw=0.25, label='WFM'); plt.plot(nLamb, dnSpec, lw=0.25, label='NFM'); plt.plot(nLamb[1:], filt, lw=0.75, label='Ratio'); plt.plot(nLamb[1:], uncont, label='Corrected WFM'); plt.legend(); plt.savefig('sameSpec')

    dWave, dLSF = np.loadtxt(dDir/'MUSE.lsf', unpack=True)
    dLSFFunc = interp1d(dWave, dLSF, 'linear', fill_value='extrapolate')
    # museLSF = dLSFFunc(tnLamb)
    museLSF = dLSFFunc(nLamb[1:])
    # velRes = CTS.c/(tnLamb/museLSF)
    velRes = CTS.c/(nLamb[1:]/museLSF)
    weis = np.ones_like(uncont)
    weis[np.where((nLamb[1:] >=7600) & (nLamb[1:] <= 7690))] = 0.0
    plt.clf(); plt.plot(nLamb[1:], uncont); plt.axvspan(7600, 7690, facecolor='k', alpha=0.4); plt.savefig('uncont')
    np.savetxt(curdir/'indata'/"SNL1_corr.dat", np.column_stack((nLamb[1:], uncont, uncont*0.03, weis, velRes)), fmt='%20.10f', header=f"{nLamb[1:][0]*1e-4:.5f} {nLamb[1:][-1]*1e-4:.5f}")
    pdb.set_trace()

def aperSpec(smask=[], rmask=[], variance=True):
    gal = au.Load.lzma(curdir.parent/'muse'/'obsData'/'SNL1.xz')
    if 'z' in gal.keys():
        RZ = Redshift(redshift=gal['z'])
    elif 'distance' in gal.keys():
        RZ = Redshift(distance=gal['distance'])
    else:
        raise RuntimeError('No distance information.')
    print(RZ)

    # wfm = next((dDir/'MUSECubes').glob(f"*SNL1_WFM_DATACUBE*.fits"))
    # wdu = pf.open(wfm)
    # wdd = wdu[1].header
    # wnL, wnY, wnX = wdd['NAXIS3'], wdd['NAXIS2'], wdd['NAXIS1']
    # wxOrg, wyOrg = GEO.genPix(np.arange(wnX), np.arange(wnY))
    # wpixs = np.abs(wdd['CD1_1']) * 60. * 60.
    # lpixs = wdd['CD3_3']
    # wLamb = wdd['CRVAL3']+np.arange(wnL)*lpixs
    # wweights = np.ones_like(wLamb)
    # for pair in smask:
    #     mask = (wLamb >= (pair[0]-lpixs)) & (wLamb <= (pair[1]+lpixs))
    #     wweights[mask] = 0.0
    # for pair in rmask:
    #     mask = (wLamb/(RZ.zShift+1) >= (pair[0]-lpixs)) &\
    #         (wLamb/(RZ.zShift+1) <= (pair[1]+lpixs))
    #     wweights[mask] = 0.0
    # wCube = np.ma.masked_invalid(wdu[1].data)
    # weCube = np.ma.masked_invalid(wdu[2].data)
    # if variance:
    #     weCube = np.ma.sqrt(weCube)
    # wFlux = np.ma.sum(wCube, axis=0)
    # wdu.close()
    # wxc, wyc, _, _, _, _ = PHT.findCentre(wFlux, 'SNL1WFM')
    # wxp = (wxOrg-wxc)*wpixs
    # wyp = (wyOrg-wyc)*wpixs
    # wrp = np.sqrt(wxp**2 + wyp**2)
    
    nfm = next((dDir/'MUSECubes').glob(f"*SNL1_NFMESOouterError_DATACUBE*.fits"))
    ndu = pf.open(nfm)
    ndd = ndu[1].header
    nnL, nnY, nnX = ndd['NAXIS3'], ndd['NAXIS2'], ndd['NAXIS1']
    nxOrg, nyOrg = GEO.genPix(np.arange(nnX), np.arange(nnY))
    npixs = np.abs(ndd['CD1_1']) * 60. * 60.
    lpixs = ndd['CD3_3']
    nLamb = ndd['CRVAL3']+np.arange(nnL)*lpixs
    nweights = np.ones_like(nLamb)
    for pair in smask:
        mask = (nLamb >= (pair[0]-lpixs)) & (nLamb <= (pair[1]+lpixs))
        nweights[mask] = 0.0
    for pair in rmask:
        mask = (nLamb/(RZ.zShift+1) >= (pair[0]-lpixs)) &\
            (nLamb/(RZ.zShift+1) <= (pair[1]+lpixs))
        nweights[mask] = 0.0
    nCube = np.ma.masked_invalid(ndu[1].data)
    neCube = np.ma.masked_invalid(ndu[2].data)
    # if variance:
        # neCube = np.ma.sqrt(neCube)
    nFlux = np.ma.sum(nCube, axis=0)
    ndu.close()
    nxc, nyc, _, _, _, _ = PHT.findCentre(nFlux, 'SNL1NFM')

    # make colour image
    bfil = svo.Filter('WFPC2.F439W')
    rfil = svo.Filter('WFPC2.F814W')
    bWave = bfil.wave.to('angstrom').value.flatten()
    bTrans = bfil.throughput.flatten()
    bUps = sint.interp1d(bWave, bTrans, fill_value='extrapolate')
    bFilt = bUps(nLamb).clip(0.0)
    rWave = rfil.wave.to('angstrom').value.flatten()
    rTrans = rfil.throughput.flatten()
    rUps = sint.interp1d(rWave, rTrans, fill_value='extrapolate')
    rFilt = rUps(nLamb).clip(0.0)
    # collapse data cube after applying filter
    bImg = np.sum(np.multiply(nCube, bFilt[:, np.newaxis, np.newaxis]),
        axis=0)
    rImg = np.sum(np.multiply(nCube, rFilt[:, np.newaxis, np.newaxis]),
        axis=0)
    dImg = bImg - rImg # colour image
    # unsharp mask the colour image
    smooth = skilters.gaussian(dImg, 1.5)
    uMask = dImg - smooth
    dust = np.ma.masked_less(uMask.ravel(), 290.)
    dMask = np.ma.getmaskarray(dust)
    dMask[(nxOrg-nxc)*npixs > 0.075] = True # mask the non-dust
    dMask[np.sqrt(((nxOrg-nxc)*npixs)**2 + ((nyOrg-nyc)*npixs)**2) > 1.5
        ] = True
    # plt.clf(); dpp((xOrgi-xc)*pixs, (yOrgi-yc)*pixs, sele & dMask, pixelsize=pixs); plt.savefig('mask'); plt.close('all')
    nxp = (np.compress(dMask, nxOrg)-nxc)*npixs
    nyp = (np.compress(dMask, nyOrg)-nyc)*npixs
    nrp = np.sqrt(nxp**2 + nyp**2)


    
    pdb.set_trace()

    # fwCube = wCube.reshape(wnL, -1)
    fnCube = nCube.reshape(nnL, -1)
    # fewCube = weCube.reshape(wnL, -1)
    fenCube = neCube.reshape(nnL, -1)
    fnCube = np.compress(dMask, fnCube, axis=1)
    fenCube = np.compress(dMask, fenCube, axis=1)

    # r1 = np.where(wrp <= 2.)[0]
    r2 = np.where(nrp <= 1.0)[0]
    # dwSpec = np.ma.sum(fwCube[:, r1], axis=1)
    dnSpec = np.ma.sum(fnCube[:, r2], axis=1)
    # dewSpec = np.ma.squeeze(np.ma.sqrt(np.ma.sum(fewCube[:, r1]**2, axis=1)))
    denSpec = np.ma.squeeze(np.ma.sqrt(np.ma.sum(fenCube[:, r2]**2, axis=1)))
    # wRelErr = dewSpec / dwSpec
    nRelErr = denSpec / dnSpec
    # dwSpec /= np.ma.median(dwSpec)
    dnSpec /= np.ma.median(dnSpec)
    # dewSpec = np.abs(dwSpec)*wRelErr
    denSpec = np.abs(dnSpec)*nRelErr

    dWave, dLSF = np.loadtxt(dDir/'MUSE.lsf', unpack=True)
    dLSFFunc = sint.interp1d(dWave, dLSF, 'linear', fill_value='extrapolate')
    nMuseLSF = dLSFFunc(nLamb)
    nVelRes = CTS.c/(nLamb/nMuseLSF)
    # wMuseLSF = dLSFFunc(wLamb)
    # wVelRes = CTS.c/(wLamb/wMuseLSF)

    # np.savetxt(curdir/'indata'/'SNL1_WFM_2arcs.dat', np.column_stack((wLamb, dwSpec, dewSpec, wweights, wVelRes)), fmt='%20.10f', header=f"{wLamb[0]*1e-4:.5f} {wLamb[-1]*1e-4:.5f}")
    np.savetxt(curdir/'indata'/'SNL1_NFMESOouterError_1arcs_dust.dat', np.column_stack((nLamb, dnSpec, denSpec, nweights, nVelRes)), fmt='%20.10f', header=f"{nLamb[0]*1e-4:.5f} {nLamb[-1]*1e-4:.5f}")

    # spectrum = 'SNL1_WFM_2arcs'
    # ifn = curdir/'indata'/f"{spectrum}.dat"
    # waves, tPix, spec, err, weights, vel = au.readSpec(ifn)
    # fig = plt.figure(figsize=plt.figaspect(1./10.))
    # ax = fig.gca()
    # for wpair in waves:
    #     ww = np.where((tPix >= wpair[0]*1e4) & (tPix <= wpair[1]*1e4))[0]
    #     ax.plot(tPix[ww], spec[ww], lw=0.4, c='r')
    # ax.fill_between(tPix, weights*spec.max(), alpha=0.2, facecolor='k',
    #     zorder=0)
    # ax.set_ylim(top=(spec*weights).max()*1.1)
    # fig.savefig(f"{spectrum}_input.pdf", format='pdf')

    spectrum = 'SNL1_NFMESOouterError_1arcs_dust'
    ifn = curdir/'indata'/f"{spectrum}.dat"
    waves, tPix, spec, err, weights, vel = au.readSpec(ifn)
    fig = plt.figure(figsize=plt.figaspect(1./10.))
    ax = fig.gca()
    for wpair in waves:
        ww = np.where((tPix >= wpair[0]*1e4) & (tPix <= wpair[1]*1e4))[0]
        ax.plot(tPix[ww], spec[ww], lw=0.4, c='r')
    ax.fill_between(tPix, weights*spec.max(), alpha=0.2, facecolor='k',
        zorder=0)
    ax.set_ylim(top=(spec*weights).max()*1.1)
    fig.savefig(f"{spectrum}_input.pdf", format='pdf')

def inout():
    nalf = au.oneSpec('SNL1_NFMESOouter_05arcs')
    walf = au.oneSpec('SNL1_WFM_2arcs')
    midx = walf.results['Type'].tolist().index('cl50')

    nimf=pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.), slopes=(nalf.results['IMF1'][midx], nalf.results['IMF2'][midx], 2.3))
    wimf=pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.), slopes=(walf.results['IMF1'][midx], walf.results['IMF2'][midx], 2.3))

    nxi = nimf.integrate(mlow=0.2, mhigh=0.5)[0]/nimf.integrate(mlow=0.2, mhigh=1.0)[0]
    wxi = wimf.integrate(mlow=0.2, mhigh=0.5)[0]/wimf.integrate(mlow=0.2, mhigh=1.0)[0]

    labs = ['sigma', 'logage', 'zH', 'FeH', 'Mg', 'Na']
    print(f"{'Prop.': ^12s}| {'NFM': ^12s} | {'WFM': ^12s}")
    for lab in labs:
        print(f"{lab: ^12s}| {nalf.results[lab][midx]: <12.6f} | {walf.results[lab][midx]: <12.6f}")
    print(f"{'xi': ^12s}| {nxi: <12.6f} | {wxi: <12.6f}")

def NFMcube():
    nfm = next((dDir/'MUSECubes').glob(f"*SNL1_NFM_DATACUBE*.fits"))
    ndu = pf.open(nfm)
    ndd = ndu[1].header
    nnL, nnY, nnX = ndd['NAXIS3'], ndd['NAXIS2'], ndd['NAXIS1']
    nxOrg, nyOrg = GEO.genPix(np.arange(nnX), np.arange(nnY))
    npixs = np.abs(ndd['CD1_1']) * 60. * 60.
    lpixs = ndd['CD3_3']
    nLamb = ndd['CRVAL3']+np.arange(nnL)*lpixs
    nCube = np.ma.masked_invalid(ndu[1].data)
    nFlux = np.ma.sum(nCube, axis=0)
    # ndu.close()
    nxc, nyc, theta, _, _, _ = PHT.findCentre(nFlux, 'SNL1NFM', 99.)
    nxp = (nxOrg-nxc)*npixs
    nyp = (nyOrg-nyc)*npixs
    theta += 90.

    sMGE = au.Load.mge('SNL1', 'F814W')
    nrp = np.sqrt(nxp**2 + (nyp/(1.-sMGE.epsE))**2)
    rxp, ryp = GEO.rotate2D(nxp, nyp, theta)
    rrp = np.sqrt(rxp**2 + (ryp/(1.-sMGE.epsE))**2)

    plt.clf(); dispp(nxp[rrp>5], nyp[rrp>5], nFlux.ravel()[rrp>5], pixelsize=npixs, angle=theta); plt.savefig('u')

    background = np.median(nCube.reshape(nnL, -1)[:, rrp>5], axis=1)
    bCube = nCube - background[:, np.newaxis, np.newaxis]

    ndu[1].data = bCube.data
    ndu.writeto(dDir/'MUSECubes'/'SNL1_NFMESOouter_DATACUBE.fits', overwrite=True)

    pdb.set_trace()


def imfCutoffDiffHorizontal():
    labels = ['velz', 'sigma', 'IMF1', 'logage', 'zH', 'FeH', 'a', 'Na', 'Ti',
        'C', 'N', 'Si', 'K', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Sr', 'Ba',
        'Eu', ]
    als = au.oneSpec('SNL1_SN80_aperture_free', labels=labels,
        redshift=0.0312)
    alc = au.oneSpec('SNL1_SN80_aperture_free_cutoff_free_norm', labels=labels,
        redshift=0.0312)
    
    ars = als.results
    types = ars['Type'].tolist()
    bidx = types.index('chi2')
    eidx = types.index('error')
    MLF814Ws = au.getM2L('SNL1_SN80_aperture_free', ars['logage'][bidx],
        ars['zH'][bidx], ars['IMF1'][bidx], ars['IMF1'][bidx], 2.3,
        RZ=Redshift(redshift=0.0312))
    arc = alc.results
    types = arc['Type'].tolist()
    bidx = types.index('chi2')
    eidx = types.index('error')
    MLF814Wc = au.getM2L('SNL1_SN80_aperture_free_cutoff_free_norm',
        arc['logage'][bidx], arc['zH'][bidx], arc['IMF1'][bidx],
        arc['IMF1'][bidx], 2.3, RZ=Redshift(redshift=0.0312),
        imflo=arc['IMF3'][bidx])

    print(f"{'Property': ^12s}| {'No Cutoff': ^12s} | {'Cutoff': ^12s} | {'Abs Diff': ^12s}")
    for lab in labels:
        no_cutoff = als.results[lab][bidx]
        cutoff = alc.results[lab][bidx]
        abs_diff = np.abs(np.diff([no_cutoff, cutoff]))[0]
        print(f"{lab: ^12s}| {no_cutoff: <12.6f} | {cutoff: <12.6f} | {abs_diff: <12.6f}")
    abs_diff_ml = np.abs(np.diff([MLF814Ws, MLF814Wc]))[0]
    print(f"{'M/L F814W': ^12s}| {MLF814Ws: <12.6f} | {MLF814Wc: <12.6f} | {abs_diff_ml: <12.6f}")

    fig = plt.figure(figsize=plt.figaspect(1/(len(labels)*8)))
    gs = gridspec.GridSpec(1, int(len(labels)/2), wspace=1.5)
    ax = fig.add_subplot(gs[:, :2])
    for li, lab in enumerate(labels[:4]):
        _ = ax.scatter(li, np.diff((ars[lab][bidx], arc[lab][bidx])
        )/ars[lab][bidx]*100., c='r', s=20)

    _ = ax.scatter(li+1, (MLF814Ws-MLF814Wc)/MLF814Ws*100., c='r', s=20)
    _ = ax.axhline(0, c='k', lw=0.5)
    _ = ax.set_xticks(np.arange(len(labels[:4])+1))
    plabels = labels[:4]+['M/L F814W']
    _ = ax.set_xticklabels(plabels, rotation=90)
    _ = ax.set_ylabel(r'$\Delta\ [\%]$')
    _ = ax.set_ylim(-50, 50)
    ax = fig.add_subplot(gs[:, 2:])
    for li, lab in enumerate(labels[4:]):
        _ = ax.scatter(li, np.diff((ars[lab][bidx], arc[lab][bidx])), c='r',
            s=20)

    _ = ax.axhline(0, c='k', lw=0.5)
    _ = ax.set_xticks(np.arange(len(labels[4:])))
    plabels = labels[4:]
    plabels[plabels.index('a')] = 'O'
    _ = ax.set_xticklabels(plabels, rotation=90)
    _ = ax.set_ylabel(r'$\Delta$')
    _ = ax.set_ylim(-0.37, 0.37)
    fig.savefig(curdir/'diffs_horizontal.png', format='png')

    pdb.set_trace()

def imfCutoffDiffVertical():
    labels = ['velz', 'sigma', 'IMF1', 'logage', 'zH', 'FeH', 'a', 'Na', 'Ti',
        'C', 'N', 'Si', 'K', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Sr', 'Ba',
        'Eu', ]
    als = au.oneSpec('SNL1_SN80_aperture_free', labels=labels,
        redshift=0.0312)
    alc = au.oneSpec('SNL1_SN80_aperture_free_cutoff', labels=labels,
        redshift=0.0312)
    
    ars = als.results
    types = ars['Type'].tolist()
    bidx = types.index('chi2')
    eidx = types.index('error')
    MLF814Ws = au.getM2L('SNL1_SN80_aperture_free', ars['logage'][bidx],
        ars['zH'][bidx], ars['IMF1'][bidx], ars['IMF1'][bidx], 2.3,
        RZ=Redshift(redshift=0.0312))
    arc = alc.results
    types = arc['Type'].tolist()
    bidx = types.index('chi2')
    eidx = types.index('error')
    MLF814Wc = au.getM2L('SNL1_SN80_aperture_free_cutoff',
        arc['logage'][bidx], arc['zH'][bidx], arc['IMF1'][bidx],
        arc['IMF1'][bidx], 2.3, RZ=Redshift(redshift=0.0312), imflo=0.15)

    print(f"{'Property': ^12s}| {'No Cutoff': ^12s} | {'Cutoff': ^12s} | {'Abs Diff': ^12s}")
    for lab in labels:
        no_cutoff = als.results[lab][bidx]
        cutoff = alc.results[lab][bidx]
        abs_diff = np.abs(np.diff([no_cutoff, cutoff]))[0]
        print(f"{lab: ^12s}| {no_cutoff: <12.6f} | {cutoff: <12.6f} | {abs_diff: <12.6f}")
    abs_diff_ml = np.abs(np.diff([MLF814Ws, MLF814Wc]))[0]
    print(f"{'M/L F814W': ^12s}| {MLF814Ws: <12.6f} | {MLF814Wc: <12.6f} | {abs_diff_ml: <12.6f}")

    fig = plt.figure(figsize=plt.figaspect(len(labels)/8))
    gs = gridspec.GridSpec(int(len(labels)/2), 1, hspace=1.5)
    ax = fig.add_subplot(gs[:4, :])
    for li, lab in enumerate(labels[:4]):
        _ = ax.scatter(np.diff((ars[lab][bidx], arc[lab][bidx])
        )/ars[lab][bidx]*100., li, c='r', s=20)

    _ = ax.scatter((MLF814Ws-MLF814Wc)/MLF814Ws*100., li+1, c='r', s=20)
    _ = ax.axvline(0, c='k', lw=0.5)
    _ = ax.set_yticks(np.arange(len(labels[:4])+1))
    plabels = labels[:4]+['M/L F814W']
    _ = ax.set_yticklabels(plabels)
    _ = ax.set_xlabel(r'$\Delta\ [\%]$')
    _ = ax.set_xlim(-50, 50)
    ax = fig.add_subplot(gs[4:, :])
    for li, lab in enumerate(labels[4:]):
        _ = ax.scatter(np.diff((ars[lab][bidx], arc[lab][bidx])), li, c='r',
            s=20)

    _ = ax.axvline(0, c='k', lw=0.5)
    _ = ax.set_yticks(np.arange(len(labels[4:])))
    plabels = labels[4:]
    plabels[plabels.index('a')] = 'O'
    _ = ax.set_yticklabels(plabels)
    _ = ax.set_xlabel(r'$\Delta$')
    _ = ax.set_xlim(-0.37, 0.37)
    fig.savefig(curdir/'diffs_vertical.png', format='png')

    pdb.set_trace()


def ucmg():
    inpf = curdir/'stacked_fits'
    fits = inpf.glob('*.fits')
    for fit in fits:
        hdul = pf.open(fit)
        hdu = hdul['COADD']
        data = hdu.data
        head = hdu.header
        fDisp = hdul['PRIMARY'].header['HIERARCH SIGMA']
        # final pre-stack dispersion
        hdul.close()

        spix = data['wave']
        alfMask = np.where((spix > 3590.1) & (spix < 11079.6))[0]
        spix = spix[alfMask]
        spec = data['flux'][alfMask]
        stat = 1./data['ivar'][alfMask]
        weights = np.ones_like(spec)
        velRes = CTS.c/(spix/data['wdisp'][alfMask])

        np.savetxt(curdir/'indata'/f"{fit.stem}_restrict.dat",
            np.column_stack((spix, spec, stat, weights, velRes)),
            fmt='%20.10f',
            header=f"{spix[0]*1e-4:.5f} {spix[-1]*1e-4:.5f}")
        
        sStr = ''
        sStr += u'#!/bin/bash -l\n'
        sStr += u'#SBATCH -p cmb\n'
        sStr += f'#SBATCH --job-name="alf_{fit.stem}_restrict"\n'
        sStr += u'#SBATCH -D "/data/phys-gal-dynamics/phys2603/alf"\n'
        sStr += u"#SBATCH --time=01-00:00\n"
        sStr += u'#SBATCH --nodes=1\n'
        sStr += u"#SBATCH --ntasks=16\n"
        sStr += u'#SBATCH --mem-per-cpu=2400\n'
        sStr += u'#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n'
        sStr += u'#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk\n'
        sStr += u'#SBATCH -o "/mnt/extraspace/poci/alf/out.log" '\
            u'# Standard out to galaxy\n'
        sStr += u'#SBATCH -e "/mnt/extraspace/poci/alf/out.log" '\
            u'# Standard err to galaxy\n'
        sStr += u'#SBATCH --open-mode=append\n\n'

        sStr += u'source ${HOME}/.bashrc\n\n'
        for mod in [['gcc', '13.2'], ['openmpi'], ['python', '3.11.4']]:
            sStr += f"module load {'/'.join(mod)}\n"

        sStr += u'export ALF_HOME=/mnt/extraspace/poci/alf/\n'
        sStr += u'### Compile clean version of `alf`\n'
        sStr += u'cd ${ALF_HOME}\n'
        sStr += u'# Run aperture fit\n'
        sStr += u'mpirun --bind-to core --map-by core '\
            f'./bin/alf.exe "{fit.stem}_restrict" 2>&1 | tee -a "{fit.stem}_restrict.log"\n\n'

        sf = io.open(curdir/f"{fit.stem}_restrict.qsys", 'w+', newline='')
        sf.write(sStr)
        sf.flush()
        sf.close()
        print(f"Saved {fit.stem}_restrict.")
    pdb.set_trace()

def readUCMG():
    inpf = curdir/'stacked_fits'
    fits = inpf.glob('*.fits')
    reg = re.compile(r'stacked_([a-zA-Z]+)_([0-9])')
    selections = []
    gals = []
    for fit in fits:
        gal = reg.match(fit.stem)
        if gal:
            gals.append(gal.group(2))
            selections.append(gal.group(1))
    selections = np.unique(selections)
    gals = np.unique(gals)
    props = ['IMF1', 'sigma', 'logage', 'FeH', 'a', 'Na', 'Mg', 'Ti', 'C', 'N']
    res = {key: {sel: np.zeros(len(gals)) for sel in selections} for key in
        props}
    labels = [r'$\Gamma_1$', r'$\sigma$', r'$\log(t\ [Gyr])$', r'$[Fe/H]$',
        r'$[O/H]$', r'$[Na/H]$', r'$[Mg/H]$', r'$[Ti/H]$', r'$[C/H]$',
        r'$[N/H]$']
    for key, label in zip(res.keys(), labels):
        res[key]['label'] = label
    for gal in gals:
        for sel in selections:
            fit = inpf/f"stacked_{sel}_{gal}.fits"
            alf = Alf(curdir/'results'/f"{fit.stem}_restrict",
                mPath=curdir/'results')
            alf.get_total_met()
            alf.normalize_spectra()
            res['IMF1'][sel][int(gal)] = alf.results['IMF1'][0]
            res['a'][sel][int(gal)] = alf.results['a'][0]
            res['Na'][sel][int(gal)] = alf.results['Na'][0]
            res['sigma'][sel][int(gal)] = alf.results['sigma'][0]
            res['logage'][sel][int(gal)] = alf.results['logage'][0]
            res['FeH'][sel][int(gal)] = alf.results['FeH'][0]
            res['Mg'][sel][int(gal)] = alf.results['Mg'][0]
            res['Ti'][sel][int(gal)] = alf.results['Ti'][0]
            res['C'][sel][int(gal)] = alf.results['C'][0]
            res['N'][sel][int(gal)] = alf.results['N'][0]


    sore = res['sigma'][selections[0]].argsort()

    fig, axs = plt.subplots(len(props) // 2, 2, figsize=(15, 2 * (len(props) // 2)), sharex=True)
    axs = axs.flatten()
    for i, prop in enumerate(props):
        for sel in selections:
            scatter = axs[i].scatter(np.arange(len(gals)), res[prop][sel],
                label=sel, s=100)
            axs[i].plot(np.arange(len(gals)), res[prop][sel], lw=0.5,
                color=scatter.get_facecolor()[0])
        axs[i].set_ylabel(res[prop]['label'])
        if i == 0:
            axs[i].legend()
    axs[-1].set_xlabel('Galaxy')
    for ax in axs:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[-1].set_xticks(np.arange(len(gals)))
    axs[-1].set_xticklabels([str(i) for i in np.arange(len(gals))])
    fig.tight_layout()
    fig.savefig(curdir/'stacked_properties_galaxy_restrict.pdf', format='pdf')
    fig, axs = plt.subplots(len(props) // 2, 2, figsize=(15, 2 * (len(props) // 2)), sharex=True)
    axs = axs.flatten()
    for i, prop in enumerate(props):
        for sel in selections:
            scatter = axs[i].scatter(np.arange(len(gals)), res[prop][sel][sore],
                label=sel, s=100)
            axs[i].plot(np.arange(len(gals)), res[prop][sel][sore], lw=0.5,
                color=scatter.get_facecolor()[0])
        axs[i].set_ylabel(res[prop]['label'])
        if i == 0:
            axs[i].legend()
    axs[-1].set_xlabel('Galaxy')
    for ax in axs:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[-1].set_xticks(np.arange(len(gals)))
    axs[-1].set_xticklabels([str(i) for i in np.arange(len(gals))[sore]])
    fig.tight_layout()
    fig.savefig(curdir/'stacked_properties_sigma_restrict.pdf', format='pdf')

    pdb.set_trace()