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
import os
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
from tqdm import tqdm
from functools import partial
from astropy.io import fits as pf
from astropy import units as uts
from scipy.special import gammaln
import itertools

# Custom modules
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
from jampy.jam_axi_proj import jam_axi_proj as jar

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

def aperSpec():
    wfm = next((dDir/'MUSECubes').glob(f"*SNL1_WFM_DATACUBE*.fits"))
    wdu = pf.open(wfm)
    wdd = wdu[1].header
    wnL, wnY, wnX = wdd['NAXIS3'], wdd['NAXIS2'], wdd['NAXIS1']
    wxOrg, wyOrg = GEO.genPix(np.arange(wnX), np.arange(wnY))
    wpixs = np.abs(wdd['CD1_1']) * 60. * 60.
    wLamb = wdd['CRVAL3']+np.arange(wnL)*wdd['CD3_3']
    wCube = np.ma.masked_invalid(wdu[1].data)
    weCube = np.ma.masked_invalid(wdu[2].data)
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
    neCube = np.ma.masked_invalid(ndu[1].data)
    nFlux = np.ma.sum(nCube, axis=0)
    ndu.close()
    nxc, nyc, _, _ = PHT.findCentre(nFlux, 'SNL1_NFM')
    nxp = (nxOrg-nxc)*npixs
    nyp = (nyOrg-nyc)*npixs
    nrp = np.sqrt(nxp**2 + nyp**2)

    wrFlux = wFlux.ravel()
    nrFlux = nFlux.ravel()
    fwCube = wCube.reshape(wnL, -1)
    fnCube = nCube.reshape(nnL, -1)
    fewCube = weCube.reshape(wnL, -1)
    fenCube = neCube.reshape(nnL, -1)

    dWave, dLSF = np.loadtxt(dDir/'MUSE.lsf', unpack=True)
    dLSFFunc = interp1d(dWave, dLSF, 'linear', fill_value='extrapolate')

    r1 = np.where(wrp <= 1.)[0]
    r2 = np.where(nrp <= 1.)[0]
    dwSpec = np.ma.sum(fwCube[:, r1], axis=1)
    dnSpec = np.ma.sum(fnCube[:, r2], axis=1)
    dewSpec = np.ma.squeeze(np.ma.sqrt(np.ma.sum(fewCube[:, r1]**2, axis=1)))
    denSpec = np.ma.squeeze(np.ma.sqrt(np.ma.sum(fenCube[:, r1]**2, axis=1)))
    wRelErr = dewSpec / dwSpec
    nRelErr = denSpec / dnSpec
    dwSpec /= np.ma.median(dwSpec)
    dnSpec /= np.ma.median(dnSpec)
    dewSpec = np.abs(dwSpec)*wRelErr
    denSpec = np.abs(dnSpec)*nRelErr

    dWave, dLSF = np.loadtxt(dDir/'MUSE.lsf', unpack=True)
    dLSFFunc = sint.interp1d(dWave, dLSF, 'linear', fill_value='extrapolate')
    nMuseLSF = dLSFFunc(nLamb)
    nVelRes = CTS.c/(nLamb/nMuseLSF)
    wMuseLSF = dLSFFunc(wLamb)
    wVelRes = CTS.c/(wLamb/wMuseLSF)

    np.savetxt(curdir/'indata'/"SNL1_WFM_1arcs.dat", np.column_stack((wLamb, dwSpec, dewSpec, np.ones_like(wLamb), wVelRes)), fmt='%20.10f', header=f"{wLamb[0]*1e-4:.5f} {wLamb[-1]*1e-4:.5f}")
    np.savetxt(curdir/'indata'/"SNL1_NFM_1arcs.dat", np.column_stack((nLamb, dnSpec, denSpec, np.ones_like(nLamb), nVelRes)), fmt='%20.10f', header=f"{nLamb[0]*1e-4:.5f} {nLamb[-1]*1e-4:.5f}")