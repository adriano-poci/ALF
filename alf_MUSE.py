# -*- coding: utf-8 -*-
"""
    alf_MUSE.py
    Adriano Poci
    Durham University
    2022

    <adriano.poci@durham.ac.uk>

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module prepares the data for input into `alf` for single MUSE pointings

    Author
    ------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:   16 June 2022
v1.1:   Account for possible lack of STATS header in datacube;
        Removed `norm` keyword argument to `display_pixels` to avoid depricated
            clash with `vmin/vmax`. 20 July 2022
v1.2:   Plot and confirm spectral pixel masking before writing files in `aap`.
            27 July 2022
v1.3:   Added `pplots` to `showPlots`. 4 August 2022
v1.4:   Store aperture spectrum and errors with Voronoi binning. 27 September
            2022
v1.5:   Added `contours` kwarg to `afh` to plot flux contours over derived maps.
            29 September 2022
v1.6:   Added `priors` kwarg to `aap`. 13 October 2022
v1.7:   Updated call to `map2img` which now always returns multiple objects.
            12 December 2022
v1.8:   Added extra 100km s^{-1} model smoothening internal to alf in `afh`. 22
            December 2022
v1.9:   Use new `polyPatch`. 1 February 2023
v1.10:  Pull `smin` and `smax` from `kwargs` if provided;
        Use `dcName` for polygon patch files. 21 March 2023
v1.11:  Corrected elemental abundance labels to be relative to Hydrogen. 16 May
            2023
v1.12:  Changed colourmaps to `seaborn.icefire`;
        Fixed bug in max IMF colour value. 13 June 2023
v1.13:  Prettyfied the `input` figure in `showPlots`;
        Corrected label references for the corner plot in `showPlots`. 4 July
            2023
v1.14:  Renamed input properties to `AAP*` to avoid confusion with output `AFH*`
            in `aap`. 5 July 2023
v1.15:  Check for strings in `_mpSpecFromSum`. 26 October 2023
v1.16:  Plot full spectral fit in `showPlots`;
        Allow list of apertures in `showPlots`;
        Check IMF type before plotting corner in `showPlots`. 7 November 2023
v1.17:  Improved full spectral fit figure in `showPlots`;
        Added `dust` treatment. 16 November 2023
v1.18:  Only plot one map if there is only one IMF free parameter in `afh`. 17
            December 2023
v1.19:  Plot [Fe/H] as metallicity, since [Z/H] is only a technical parameter in
            alf --- it determines the isochrone set, but isn't physically the
            metallicity;
        Plot only the abundances with some dynamic range. 18 December 2023
v1.20:  Added `radial` argument to `pplots` to separate the radial profiles of
            the stellar populations;
        Romanicised the metallicity label;
        Added posterior samples to the abundance profiles. 20 December 2023
v1.21:  Explicitly pass consistent value of `fgFrac` to all `find_galaxy` calls.
            6 March 2024
v1.22:  Pass `RZ` to `alfWrite` to set initial guess of the systemic velocity.
            11 March 2024
v1.23:  Return `alf` object from `showPlots` for further analysis. 19 April 2024
v1.24:  Generate custom corner plot with `corner` in `showPlots`. 22 April 2024
v1.25:  Fixed bug where `imft` was being converted to `bool` and therefore
            loosing its integer value. 20 May 2024
v1.26:  Standardised colourmaps for diverging and monotonic maps. 26 July 2024
v1.27:  Allow reading in incomplete ALF runs in `afh`. 14 April 2025
v1.28:  Use `corrected` individual abundances. 5 June 2025
v1.19:  Compute and plot the total metallicity `[Z/H]`. 7 June 2025
v1.20:  Converted to `PowerBin` for binning scheme. 15 October 2025
v1.21:  Replaced `photFilt` and `band` with `filt` kwarg. 22 October 2025
v1.22:  Introduced toggle between binning algorithms. 23 October 2025
v1.23:  Skip M/L calculation is `.bestspec2` file is missing;
        Removed superfluous probes of `outs` in `afh`. 11 December 2025
v1.24:  Ensure `.sum` exists for every `.mcmc` in `afh`. 12 December 2025
v1.25:  Use rendered labels for `corner` in `showPlots`. 17 August 2026
"""
from __future__ import print_function, division

# General modules
import os, re
import traceback, warnings
import sys
import pdb
import time
import json
import signal
import pathlib as plp
from copy import copy
from glob import glob
import shutil as su
import numpy as np
from scipy import ndimage
from scipy.stats.mstats import scoreatpercentile as sssp
from scipy.interpolate import interp1d
from skimage import filters as skilters
from astropy.io import fits as pf
from astropy.stats import sigma_clip as assc
from astropy.cosmology import z_at_value as azav, Planck18 as cosmo
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import seaborn as sns
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import subprocess as sp
import itertools
from inspect import getargvalues as ingav, currentframe as incf
from svo_filters import svo

# Custom modules
from alf.Alf import Alf
import alf.alf_utils as au
from dynamics.SN_Ellipse_Cut import SNRing
from dynamics.IFU.Constants import Constants, UnitStr
from dynamics.IFU.Functions import Plot, Geometric
from dynamics.IFU.Galaxy import Redshift, Photometry, pieceIMF
# from dynamics.IFU import geckos_colourmap as gcc

# Dynamics modules
from mgefit.find_galaxy import find_galaxy
from vorbin.voronoi_2d_binning import voronoi_2d_binning as v2db
from powerbin import PowerBin
from plotbin.display_bins import display_bins as dbi
from plotbin.display_pixels import display_pixels as dpp
from plotbin.symmetrize_velfield import symmetrize_velfield as syvf
from pafit.fit_kinematic_pa import fit_kinematic_pa as fkpa
from ppxf.ppxf_util import log_rebin

curdir = plp.Path(__file__).parent
dDir = au._ddir()
icefire = sns.color_palette('icefire', as_cmap=True)
rocket = sns.color_palette('rocket', as_cmap=True)
rocketr = sns.color_palette('rocket_r', as_cmap=True)

CTS = Constants()
UTS = UnitStr()
POT = Plot()
GEO = Geometric()
PHT = Photometry()

divcmap = 'GECKOSdr'
moncmap = 'inferno'

alfFP = ['chi2', 'velz', 'sigma', 'logage', 'zH', 'FeH', 'a', 'C', 'N', 'Na',
    'Mg', 'Si', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Sr', 'Ba',
    'Eu', 'Teff', 'IMF1', 'IMF2', 'logfy', 'sigma2', 'velz2', 'logm7g',
    'hotteff', 'loghot', 'fy_logage', 'logemline_h', 'logemline_oii',
    'logemline_oiii', 'logemline_sii', 'logemline_ni', 'logemline_nii',
    'logtrans', 'jitter', 'logsky', 'IMF3', 'IMF4', 'h3', 'h4', 'ML_v', 'ML_i',
    'ML_k', 'MW_v', 'MW_i', 'MW_k']
# ORder of ALF free parameters

# ------------------------------------------------------------------------------

def _mpCount( j, gSpec, mpCount ):
    mpCount[j] = np.count_nonzero(np.isnan(gSpec[:,j]))

    return j

# ------------------------------------------------------------------------------

def binNumber(galaxy, SN, full=False, binni=None, dcName=''):
    SN = int(SN)

    if not full:
        tEnd = 'trunc'
    else:
        tEnd = 'full'
    
    gDir = curdir/f"{galaxy}{dcName}"

    pifs = gDir/f"pixels_SN{SN:d}.xz"
    bofs = gDir/f"binning_SN{SN:02d}_{tEnd}.xz"
    sefs = gDir/f"selection_SN{SN:02d}_{tEnd}.xz"

    if isinstance(binni, type(None)):
        PB = au.Load.lzma(bofs)
    else:
        PB = binni
    saur, goods = au.Load.lzma(sefs)
    xpix, ypix, sele, pixs = au.Load.lzma(pifs)
    xbin, ybin = PB['xbin'], PB['ybin']
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    binNum = PB['binNum']

    BNImg, _, _, _ = POT.map2img(xpix, ypix, binNum, pixSize=pixs)

    pf.writeto(gDir/f"binnumber_SN{SN:02d}_{tEnd}.fits",
               BNImg.filled(np.nan), overwrite=True)

# ------------------------------------------------------------------------------

def aap(galaxy='NGC5102', kPath=(dDir/'MUSECubes'), sin=True, targetSN=60,
       minSN=1, full=False, quick=False, kfn=None, dcName='',
       instrument='muse', qProps=dict(timeMax=168, module=[]), smask=[],
       variance=True, priors=True, qsys='slurm', binScheme='voronoi', **kwargs):
    """
    Collates the necessary data to feed into pPXF for NGC 3115
    Args
    ----
        galaxy (str): The name of the galaxy
        kPath (str): the directory containing the reduced data cube
        sin (bool): toggles whether to bin the spectra
        targetSN (int): the target S/N required by the binning algorithm
        minSN (int): the minimum S/N for a spaxel to be included in the binning
        full (bool): toggles whether to fit the entire spectral range of the
            data, or truncate to some pre-defined range
        quick (bool): directive for plotting and printing commands
        kfn (str): the filename of the outputs
        dcName (str): a wildcard string to pass to glob when searching for the
            data-cube. Useful for multiple versions of data-cubes
        instrument (str): the name of the instrument, used to truncate the
            templates on the correct wavelength range
        qprops (dict): provides options for the queuing system being used.
            Options include:
            timeMax (int): the maximum number of hours a job can take
            owner (str): the account under which to run the job
            queue (str): the queue to submit the job to
            module (list:list:str): list of length-2 lists specifying
                ['<module name>', '<module version>'] for each module that
                is required on the system
        smask (list:float): a list of `[smin, smax]` pairs indicating the
            spectral ranges to be masked
        variance (bool): toggles whether the stat table in the cube is the
            variance. Otherwise assumed to be StD
        priors (bool): toggles whether to use the aperture fit to set priors for
            the remaining fits
        binScheme (str): the binning scheme to use. Options are:
            'voronoi': Voronoi binning
            'power': Power binning
    Examples
    --------
    am.aap('ESO484-036', kPath=am.dDir/'GECKOSMaps', full=True, qProps=dict(queue='cmb', timeMax=72, module=[['gcc', '13.2'], ['openmpi'], ['python', '3.11.7']]), smask=[[4700, 4750], [7580, 7700], [8970, 9100]], minSN=1.5, targetSN=100)
    am.aap('NGC3630', kPath=am.dDir/'GECKOSCubes', full=True, qProps=dict(queue='cmb', module=[['gcc', '13.2'], ['openmpi'], ['python', '3.11.4']], sarray=False, NCPU=21), smask=[[4700, 4750], [5575, 5580], [6297, 6302], [6861, 6943], [7580, 7700], [8120, 8201], [8825, 8830], [8900, 9100]], minSN=1.5, targetSN=100, qsys='glamdring')
    """
    targetSN = int(targetSN)

    fgFrac = kwargs.pop('fgFrac', 98.5)

    gDir = curdir/f"{galaxy}{dcName}"
    gDir.mkdir(parents=True, exist_ok=True)
    # (curdir/'results'/galaxy).mkdir(parents=True, exist_ok=True)

    # source code configuration
    varsf = ['fit_indices', 'fit_type', 'fit_hermite', 'imf_type',
        'observed_frame', 'mwimf', 'fit_two_ages', 'nonpimf_alpha', 'extmlpr',
        'nonpimf_regularize', 'use_age_dep_resp_fcns', 'fix_age_dep_resp_fcns',
        'use_z_dep_resp_fcns', 'fix_z_dep_resp_fcns', 'fit_trans', 'atlas_imf',
        'smooth_trans', 'velbroad_simple', 'extmlpr', 'fit_poly', 'maskem',
        'apply_temperrfcn', 'fake_response', 'blueimf_off', 'nstart', 'nend',
        'nlint_max', 'neml', 'npar', 'nage', 'nzmet', 'npowell', 'nage_rfcn',
        'nimf_full', 'nmcut', 'nimfoff', 'nimfnp', 'npolymax', 'poly_dlam',
        'ndat', 'nparsimp', 'nindx', 'nfil',  'nhot', 'imflo', 'imfhi',
        'krpa_imf1', 'krpa_imf2', 'krpa_imf3']
    CNF = dict()
    with open(curdir/'src'/'alf.perm.f90', 'r') as af:
        af90 = af.read()
    with open(curdir/'src'/'alf_vars.f90', 'r') as af:
        vf90 = af.read()
    for key in varsf:
        reg = re.compile(rf"[\,\n]{{1}}.*{key}.?=.*[\,\n]{{1}}")
        val = reg.findall(af90)
        if len(val) > 0:
            val = val[0].strip().split('=')[-1].split('!')[0].strip()
            CNF[key] = val
        val = reg.findall(vf90)
        if len(val) > 0:
            val = val[0].strip().split('=')[-1].split('!')[0].strip()
            if key in CNF.keys():
                assert val == CNF[key], f"Inconsistent settings for {key}:\n"\
                    f"{'': <4s}alf: {CNF[key]}\n"\
                    f"{'': <4s}alf_vars: {val}"
            else:
                CNF[key] = val

    if not full:  # Clip the spectral data if required
        smax = kwargs.get('smax', 7300.)
        tEnd = 'trunc'
    else:
        smax = kwargs.get('smax', 9000.)  # Do star and SSP on the same range
        tEnd = 'full'
    CNF['smax'] = smax
    CNF['full'] = full
    au.Write.lzma(gDir/'config.xz', CNF)

    kinF = gDir/f"AAP_SN{targetSN:02d}_{tEnd}.xz"
    if isinstance(kfn, type(None)):
        kfn = plp.Path(kinF.name)
    else:
        kfn = plp.Path(kfn).name

    # Sort out file existence
    pixels, binning, selection, srn, kines, cubeCube = [False]*6
    pifs = gDir/f"pixels_SN{targetSN:02d}.xz"
    bofs = gDir/f"binning_SN{targetSN:02d}_{tEnd}.xz"
    sefs = gDir/f"selection_SN{targetSN:02d}_{tEnd}.xz"
    snfs = gDir/f"SNR_{tEnd}.xz"
    gfs = curdir.parent/'muse'/'obsData'/f"{galaxy}.xz"
    jfn = dDir/'galaxy-props'/f"{galaxy}.json"

    if dcName != '':
        dcgName = f"*{dcName}*"
    else:
        dcgName = '*'
    try:
        cubeCube = next(kPath.rglob(f"*{galaxy}{dcgName}_DATACUBE*.fits"))
    except StopIteration:
        try:
            cubeCube = next(kPath.rglob(f"*{galaxy}{dcgName}_DATACUBE*.fz"))
        except StopIteration:
            cubeCube = None
    pixels = pifs.exists()
    binning = bofs.exists()
    selection = sefs.exists()
    srn = snfs.exists()
    if not jfn.exists():
        print(f"No galaxy property JSON at {jfn}")
        pdb.set_trace()
    else:
        jgal = au.Load.json(jfn) # load hand-written properties
    if not gfs.exists():
        if not jfn.exists():
            print('No galaxy property dictionary.')
            pdb.set_trace()
        else:
            au.Write.lzma(gfs, jgal)
    if kinF.exists():
        kines = True

    gal = au.Load.lzma(gfs)
    gal.update(jgal)
    au.Write.lzma(gfs, gal)

    print(f"\n\nOptions:\n"+\
        f"{'': <4s}{'Pixel': <10s}: {str(pixels): <5s} ({pifs.name})\n"+\
        f"{'': <4s}{'Binning': <10s}: {str(binning): <5s} ({bofs.name})\n"+\
        f"{'': <4s}{'Selection': <10s}: {str(selection): <5s} "+\
                f"({sefs.name})\n"+\
        f"{'': <4s}{'Kinematics': <10s}: {str(kines): <5s} ({kinF.name})\n"+\
        f"{'': <4s}{'Output': <10s}: {'': <5s} ({kfn.name})\n\n",
    flush=True)


    kfPath = gDir/kfn
    if kfPath.exists():
        output = au.Load.lzma(kfPath)
    else:
        output = dict()

    if not cubeCube:
        raise IOError(f"No spectral data cube for {galaxy}")
    print(f"Reading in data cube...\n{'': <4s}{cubeCube}")
    hdu = pf.open(cubeCube)
    print(hdu.info())
    try:
        dataExt = 'DATA'
        dhdr = hdu[dataExt].header
    except KeyError:
        try:
            dataExt = 1
            dhdr = hdu[dataExt].header
        except IndexError:
            dataExt = 0
            dhdr = hdu[dataExt].header
    try:
        statExt = 'STAT'
        shdr = hdu[statExt].header

    except IndexError:
        try:
            statExt = dataExt+1
            shdr = hdu[statExt].header
        except KeyError:
            statExt = None
            shdr = None
    hData = np.ma.masked_invalid(hdu[dataExt].data)
    print(f"{'': <4s}Data-cube dimensions: {hData.shape}")
    fluxii = np.ma.sum(hData, axis=0)
    print('Done.\n\n')

    pf.writeto(gDir/f"collapsed.fits", fluxii.filled(np.nan),
        overwrite=True)
    fluxi = fluxii.ravel()

    gal = au.Load.lzma(gfs)
    if 'z' in gal.keys():
        RZ = Redshift(redshift=gal['z'])
    elif 'distance' in gal.keys():
        RZ = Redshift(distance=gal['distance'])
    else:
        raise RuntimeError('No distance information.')
    print(RZ)
    vSys = RZ.toVSys()

    if 'mask' in jgal.keys():
        gal['mask'] = jgal['mask']
    au.Write.lzma(gfs, gal)

    print(f"vSys (starting velocity): {vSys:4.4f} km s^{{-1}}")

    nL, nY, nX = dhdr['NAXIS3'], dhdr['NAXIS2'], dhdr['NAXIS1']

    lambA = dhdr['CRVAL3']+np.arange(nL)*dhdr['CD3_3']
    # wavelength in Angstrom
    smin = kwargs.get('smin', np.max([np.min(lambA), 4000.]))

    saur = np.where((lambA >= smin) & (lambA <= smax))[0]
    lPix = lambA[saur]
    lmin, lmax = lPix.min(), lPix.max()
    llen = saur.size
    llen = saur.size

    lMask = np.ma.ones(lPix.size, dtype=bool)
    for pair in smask:
        lMask[(lPix <= pair[1]) & (lPix >= pair[0])] = 0

    if pixels:
        print('Reading pixel grid...')
        xp, yp, sele, pixs = au.Load.lzma(pifs)
        flux = np.compress(sele, fluxi)
        xc, yc, photPA, fcfg, gPlt, fPlt = PHT.findCentre(np.ma.masked_array(
            fluxii, mask=~sele), galaxy, fgFrac=fgFrac)
        print('Done.')
    else:
        print('Generating pixel grid...')

        # Reverse x and y for consistency with FITS kinematics
        xOrgi, yOrgi = GEO.genPix(np.arange(nX), np.arange(nY))
        nXY = len(xOrgi)
        pixs = np.abs(dhdr['CD1_1']) * 60. * 60.

        xMuse = 315 # 3-pixel padding on a side
        yMuse = 315

        print(f"{'': <4s}Selecting appropriate pixels...")
        idim = fluxi.shape
        sele = fluxi > 0

        if 'NGC5102' in galaxy:
            points = [
                [45, 192, 6],
                [11, 125, 5],
                [67, 35, 5],
                [110, 106, 5],
                [88, 270, 5],
                [210, 185, 8],
                [214, 215, 8],
                [298, 303, 10],
            ]
            ellips = []
        elif 'NGC0448' in galaxy:
            points = [
                [255, 128, 11],
                [89, 96, 9],
                [260, 202, 5],
                [254, 204, 5],
                [75, 97, 7],
                [67, 137, 6],
                [39, 117, 6],
                [117, 188, 6],
                [214, 227, 8],
                [199, 223, 6],
                [300, 209, 6],
                [224, 248, 6],
                [209, 243, 5],
                [211, 256, 5],
                [217, 281, 20],
                [108, 64, 5],
                [149, 79, 5],
                [213, 93, 5],
                [15, 71, 6],
                [69, 171, 6],
                [202, 51, 7],
                [241, 259, 7],
                [66, 224, 7],
                [39, 215, 10],
                [286, 277, 10],
                [299, 271, 8],
            ]
            ellips = []
        elif 'NGC2698' in galaxy:
            points = [
                [282, 172, 20],
                [302, 75, 10],
                [231, 153, 10],
                [107, 251, 10],
                [25, 217, 8],
                [283, 307, 8],
                [308, 355, 20],
                [260, 285, 8],
                [290, 373, 8],
                [295, 382, 10],
                [210, 415, 10],
                [205, 183, 8],
                [280, 357, 8],
                [120, 140, 8],
                [142, 120, 8],
                [260, 187, 5],
                [179, 241, 5],
                [27, 215, 5],
                [353, 278, 5],
                [279, 241, 5],
                [245, 372, 5],
                [210, 414, 5],
                [71, 210, 5],
                [206, 85, 8],
                [210, 152, 5],
                [235, 139, 5],
                [267, 317, 5],
                [238, 312, 5],
                [221, 288, 5],
                [200, 291, 5],
                [158, 284, 5],
                [161, 302, 5],
                [234, 139, 5],
                [242, 69, 5],
                [254, 65, 5],
                [354, 309, 8],
                [310, 326, 5],
                [331, 341, 5],
                [279, 367, 5],
                [183, 395, 8],
                [254, 371, 5],
                [281, 405, 5],
                [227, 423, 15],
                [255, 408, 20],
            ]
            ellips = []
        elif 'NGC4365' in galaxy:
            points = [
                [64, 99, 7],
                [232, 6, 8],
                [286, 185, 6],
                [97, 285, 6],
                [235, 263, 7],
                [284, 292, 5],
            ]
            ellips = []
        elif 'NGC4684' in galaxy:
            points = [
                [140, 378, 13],
                [203, 135, 12],
                [273, 148, 6],
                [292, 176, 6],
                [123, 285, 5],
                [125, 225, 5],
            ]
            ellips = []
        elif 'NGC5507' in galaxy:
            points = [
                [349, 106, 21],
                [167, 220, 4],
                [142, 251, 5],
                [126, 244, 4],
                [247, 133, 5],
                [127, 290, 6],
                [141, 296, 5],
                [41, 262, 7],
                [30, 213, 8],
                [20, 256, 5],
                [77, 99, 5],
                [295, 308, 6],
                [262, 336, 6],
                [188, 351, 5],
                [268, 374, 5],
                [228, 197, 5],
                [212, 173, 4],
                [203, 178, 4],
                [127, 220, 6],
                [194, 227, 4],
                [211, 219, 4],
                [167, 242, 4],
                [246, 165, 5],
                [188, 136, 5],
            ]
            ellips = []
        elif 'J0946' in galaxy:
            points = [
                [143, 141, 9],
                [100, 73, 7],
                [41, 90, 27],
                [93, 112, 5],
                [76, 198, 7],
                [98, 147, 5],
                [104, 163, 5],
                [113, 161, 5],
                [117, 140, 5],
                [123, 147, 5],
                [136, 169, 5],
                [125, 165, 5],
                [28, 50, 41],
                [201, 107, 5],
                [215, 188, 16],
                [99, 72, 8],
                [90, 63, 6],
                [112, 97, 3],
                [114, 95, 3],
                [116, 94, 3],
                [120, 93, 2],
                [120, 94, 2],
                [123, 94, 2],
                [126, 96, 2],
                [147, 95, 5],
            ]
            ellips = []
        elif 'J14510239' in galaxy:
            points = [
                [253, 197, 311-253],
                [207, 121, 17],
                [180, 132, 9],
                [139, 160, 6],
                [148, 159, 4],
                [172, 176, 10],
                [174, 216, 15],
                [106, 93, 15],
                [77, 159, 101-77],
            ]
            ellips = []
        elif 'J09120529' in galaxy:
            points = [
                [132, 185, 12],
            ]
            ellips = []
        elif 'J11432962' in galaxy:
            points = [
                [120, 165, 19],
                [179, 158, 5],
                [162, 98, 10],
                [151, 144, 3],
                [153, 143, 3],
            ]
            ellips = []
        elif 'SNL0' in galaxy:
            points = [
                [319, 171, 10],
                [387, 157, 25],
                [355, 138, 25],
                [268, 337, 5],
                [296, 25, 5],
                [231, 161, 8],
                [183, 221, 8],
                [242, 296, 8],
                [243, 300, 8],
                [270, 297, 8],
                [328, 322, 8],
                [176, 272, 8],
                [149, 225, 5],
                [142, 197, 5],
                [115, 247, 5],
                [120, 231, 5],
                [90, 209, 5],
                [268, 337, 20],
                [315, 134, 20],
                [418, 302, 10],
                [414, 292, 10],
                [440, 122, 15],
                [318, 81, 6],
                [389, 104, 6],
                [409, 133, 10],
                [428, 162, 20],
                [90, 210, 10],
                [77, 247, 10],
                [69, 265, 10],
                [54, 347, 10],
                [202, 414, 10],
                [240, 352, 10],
                [295, 342, 10],
                [407, 210, 10],
                [415, 387, 10],
            ]
            ellips = []
        elif 'SNL1' in galaxy and 'WFM' in dcName:
            # GIMP gives reversed y-axis
            points = [
                [256, 21, 10],
            ]
            ellips = [
                # [172, 203, 6, 4, -10.],
                # [165, 193, 5, 4, -10.],
                # [180, 212, 16, 1.3, -20.],
                # [177, 218, 8, 2, -20.],
            ]
        elif 'SNL2' in galaxy:
            # GIMP gives reversed y-axis
            points = [
                [305, nY-224, 10],
                [286, nY-176, 3],
                [117, 114, 7],
                [142, 64, 20],
                [214, 140, 4],
                [285, 131, 5],
                [137, 105, 5],
                [161, 97, 5],
                [94, 78, 5],
                [237, 54, 10],
                [193, 77, 5],
                [200, 115, 4],
                [231, 119, 4],
                [117, 142, 6],
            ]
            ellips = []
        elif 'M87' in galaxy:
            points = []
            ellips = [
                [240, 185, 2, 48, -69.]
            ]
        elif 'NGC3957' in galaxy:
            points = [
                [229, 88, 5],
                [144, 94, 5],
                [120, 144, 8],
                [131, 177, 6],
                [152, 261, 6],
                [158, 332, 10],
                [166, 410, 10],
                [256, 565, 8],
                [250, 57, 30],
                [250, 192, 5],
                [283, 244, 6],
                [277, 343, 6],
                [140, 261, 6],
            ]
            ellips = []
        else:
            points = []
            ellips = []

        for (xj, yj, rj) in points:
            sele &= np.sqrt((xOrgi-xj)**2 + (yOrgi-yj)**2) > rj
        for (xj, yj, rj, ej, aj) in ellips:
            xx = xOrgi-xj
            yy = yOrgi-yj
            AA = np.radians(aj)
            sele &= np.sqrt((xx*np.cos(AA)+yy*np.sin(AA))**2 + \
                ((xx*np.sin(AA)-yy*np.cos(AA))/ej)**2) > rj
        if 'J14510239' in galaxy:
            sele &= np.sqrt((xOrgi-154)**2 + (yOrgi-145)**2) <= 55
        if 'J09120529' in galaxy:
            sele &= np.sqrt((xOrgi-155)**2 + (yOrgi-155)**2) <= 50
        if 'J11432962' in galaxy:
            sele &= np.sqrt((xOrgi-160)**2 + (yOrgi-152)**2) <= 55
        if 'SNL0' in galaxy:
            sele &= np.sqrt((xOrgi-276)**2 + (yOrgi-232)**2) <= 150
        if 'SNL1' in galaxy and 'NFM' in dcName:
            sele &= np.sqrt((xOrgi-177)**2 + (yOrgi-169)**2) <= int(1.25/2./pixs)
        if 'SNL2' in galaxy:
            sele &= np.sqrt((xOrgi-205)**2 + (yOrgi-(125))**2) <= 100

        xc, yc, photPA, fcfg, gPlt, fPlt = PHT.findCentre(
            np.ma.masked_array(fluxii, mask=~sele), galaxy, fgFrac=fgFrac)
        gPlt.savefig(gDir/'fc_flux')
        fPlt.savefig(gDir/'fc_fg')
        if 'UGC00903' in galaxy:
            xc, yc = 249, 77

        if kwargs.pop('dust', False):
            # make colour image
            bfil = svo.Filter('WFPC2.F439W')
            rfil = svo.Filter('WFPC2.F814W')
            bWave = bfil.wave.to('angstrom').value.flatten()
            bTrans = bfil.throughput.flatten()
            bUps = interp1d(bWave, bTrans, fill_value='extrapolate')
            bFilt = bUps(lambA).clip(0.0)
            rWave = rfil.wave.to('angstrom').value.flatten()
            rTrans = rfil.throughput.flatten()
            rUps = interp1d(rWave, rTrans, fill_value='extrapolate')
            rFilt = rUps(lambA).clip(0.0)
            # collapse data cube after applying filter
            if isinstance(hData, type(None)):
                hData = np.ma.masked_invalid(hdu[dataExt].data)
            bImg = np.sum(np.multiply(hData, bFilt[:, np.newaxis, np.newaxis]),
                axis=0)
            rImg = np.sum(np.multiply(hData, rFilt[:, np.newaxis, np.newaxis]),
                axis=0)
            dImg = bImg - rImg # colour image
            # unsharp mask the colour image
            smooth = skilters.gaussian(dImg, 1.5)
            uMask = dImg - smooth
            dust = np.ma.masked_less(uMask.ravel(), 290.)
            dMask = np.ma.getmaskarray(dust)
            dMask[(xOrgi-xc)*pixs > 0.075] = True # mask the non-dust
            dMask[np.sqrt(((xOrgi-xc)*pixs)**2 + ((yOrgi-yc)*pixs)**2) > 1.5
                ] = True
            # plt.clf(); dpp((xOrgi-xc)*pixs, (yOrgi-yc)*pixs, sele & dMask, pixelsize=pixs); plt.savefig('mask'); plt.close('all')
            sele &= dMask # invert the mask

        xOrg = np.compress(sele, xOrgi)
        yOrg = np.compress(sele, yOrgi)
        flux = np.compress(sele, fluxi)
        xp = (xOrg-xc)*pixs
        yp = (yOrg-yc)*pixs

        pfn = dDir.parent/'muse'/'obsData'/f"{galaxy}{dcName}-poly-obs.xz"
        if pfn.is_file():
            aShape = au.Load.lzma(pfn)
            aShape, pPatch = POT.polyPatch(POLYGON=aShape, Xpo=xp, Ypo=yp,
                salpha=0.5, ec=POT.brown, linestyle='--', fill=False, zorder=0,
                lw=0.75)
        else:
            aShape, pPatch = POT.polyPatch(Xpo=xp, Ypo=yp, salpha=0.5,
                ec=POT.brown, linestyle='--', fill=False, zorder=0, lw=0.75)
            au.Write.lzma(pfn, aShape)

        plt.clf()
        dpp(xOrg, yOrg, np.log10(flux), pixelsize=1.0, cmap='prism')
        plt.grid(which='both', axis='both', zorder=10, color='k',
            linewidth=0.4, ls='-')
        plt.gca().set_aspect('equal')
        plt.savefig(gDir/'pixelMask')

        print(f"{'': <4s}Found (xc, yc) = ({xc:3.1f},{yc:3.1f})")

        au.Write.lzma(pifs, [xp, yp, sele, pixs])

        gal['cent'] = [xc, yc]
        au.Write.lzma(gfs, gal)

        plt.clf()
        dpp(xp, yp, np.log10(flux), pixelsize=pixs, cmap='prism')
        plt.gca().add_patch(copy(pPatch))
        plt.gca().set_aspect('equal')
        plt.savefig(gDir/'pixels')
        plt.close('all')
        print('Done.', flush=True)

    # Defined variable here:
    #   xp
    #   yp
    #   sele

    loop = False
    if sin and binning:
        print('Reading binned data...')
        PB = au.Load.lzma(bofs)
        if 'scheme' in PB.keys():
            if PB['scheme'] != binScheme:
                raise RuntimeWarning(f"[Binning] Binning scheme mismatch: "\
                    f"{PB['scheme']} != {binScheme}. Re-running binning.")
                loop = True
        xp = PB['xbin']
        yp = PB['ybin']
        binNum = PB['binNum']
        nPixels = PB['nPixels']
        # scale = PB['scale']
        endSN = PB['endSN']
        gspecs = PB['binSpec']
        stats = PB['binStat']
        # logLam = PB['logLam']

        nbins = xp.size

        aperSpec = PB['aperSpec']

        print('Done.', flush=True)
    elif sin: # binning is desired, but the raw cube needs to be loaded
        loop = True
    if (not sin) or loop:

        print('Generating spectral data...')
        print(f"{'': <4s}Reshaping `gspecs`...")
        # use full length to get the right shape
        if isinstance(hData, type(None)):
            hData = np.ma.masked_invalid(hdu[dataExt].data)
        gspecs = np.compress(sele, hData.reshape(nL, -1), axis=1)
        print(f"{'': <4s}Reshaping `stats`...")
        # use full length to get the right shape
        if statExt:
            stats = np.ma.masked_invalid(np.compress(sele,
                hdu[statExt].data.reshape(nL, -1), axis=1))
        else:
            stats = np.multiply(np.ma.ones(gspecs.shape),
                5./np.log10(gspecs))
        if variance:
            stats = np.ma.sqrt(stats)  # sqrt(var) = 1σ errors

        hdu.close()
        print('Done.')
        # Defined variable here:
        #   gspecs
        #   stats
        gspecs = np.take(gspecs, saur, axis=0)
        stats = np.take(stats, saur, axis=0)

        # notch = [576, 605] # [nm], maximum range for both NFM and WFM
        # nww = np.where((lPix < notch[0]*10) | (lPix > notch[1]*10))[0]
        # the notch contributes NaNs to every spectrum, but isn't of
        #   concern

        if selection:
            print('Reading selections...')
            _saur, goods = au.Load.lzma(sefs)
            print('Done.', flush=True)
        else:
            print('Generating selection...')

            nNaN = np.count_nonzero(np.isnan(gspecs[lMask, :].data), axis=0)
            nNeg = np.count_nonzero(gspecs < 0, axis=0)
            if srn:
                print(f"{'': <4s}Reading S/N...")
                SNR = au.Load.lzma(snfs)
            else:
                print(f"{'': <4s}Computing S/N...")
                signal = np.ma.median(gspecs[lMask, :], axis=0)
                noise = np.abs(np.ma.median(stats[lMask, :], axis=0))
                SNR = np.divide(signal, noise)
                au.Write.lzma(snfs, SNR)
            
            _, _, _, snMask = SNRing(SNR, minSN, xp, yp, flux, pixs, debug=True,
                galaxyPath=gDir, fgFrac=fgFrac)
            # maximum 10% NaN or negative values
            goods = (nNaN < llen/20.) & (nNeg < llen/20.) & (SNR >= minSN) & \
                snMask
            if 'mask' in gal.keys():
                for mk in gal['mask']:
                    X, Y, dia = mk
                    print(f"{'': <4s}Masking ({X:+02d},{Y:+02d}) r={dia:d}")
                    mask = np.where(((xp-X)**2 + (yp-Y)**2) < dia)[0]
                    goods[mask] = False

            au.Write.lzma(sefs, [saur, goods])
            print('Done.', flush=True)
        # Defined variable here:
        #   saur
        #   goods

        print('Applying selections...')
        gspecs = np.compress(goods, gspecs, axis=1)
        stats = np.compress(goods, stats, axis=1)
        xp = np.compress(goods, xp)
        yp = np.compress(goods, yp)
        print('Done.', flush=True)

        print('Plotting...')
        plt.clf()
        dpp(xp, yp, np.ma.sum(gspecs, axis=0), cmap='gist_heat',
            pixelsize=pixs, vmin=1e-1)
        plt.gca().set_aspect('equal')
        plt.savefig(gDir/'flux')
        plt.xlim([xp.min()/5., xp.max()/5.])
        plt.ylim([yp.min()/5., yp.max()/5.])
        plt.axvline(0., lw=0.25)
        plt.axhline(0., lw=0.25)
        plt.savefig(gDir/'fluxCen')
        plt.close('all')
        print('Done.', flush=True)

        if sin:
            try:
                print('Running binning...', flush=True)
                gMed = np.ma.median(gspecs[lMask, :], axis=0)
                sMed = np.ma.median(stats[lMask, :], axis=0)  # median(1σ)
                # one last check
                sMed[np.ma.getmaskarray(sMed)] = np.ma.median(sMed)

                PB = dict()
                if 'voronoi' in binScheme:
                    binNum, xbin, ybin, xbar, ybar, endSN, nPixels, scale = \
                        v2db(xp, yp, gMed, sMed, targetSN, plot=True,
                        quiet=quick, pixelsize=pixs
                    )
                    plt.savefig(gDir/f"v2db_SN{targetSN:02d}")
                    plt.close('all')
                    plt.clf()
                elif 'power' in binScheme:
                    def capacity_spec(index):
                        """Calculates (S/N)^2 for a bin from its pixel indices."""
                        # Standard S/N formula for uncorrelated noise
                        sn = np.sum(gMed[index]) / np.sqrt(np.sum(sMed[index]**2))
                        return sn**2
                    powb = PowerBin(np.column_stack((xp, yp)), capacity_spec,
                        target_capacity=targetSN**2, pixelsize=pixs)
                    plt.clf()
                    powb.plot(capacity_scale='sqrt', ylabel='S/N')
                    plt.savefig(gDir/f"bin2d_SN{targetSN:02d}")
                    plt.close('all')
                    plt.clf()
                    binNum = powb.bin_num
                    endSN = np.sqrt(powb.bin_capacity)
                    nPixels = powb.npix
                    xbin, ybin = powb.xybin.T
                else:
                    raise ValueError(f"Unknown binning scheme: {binScheme}")

                PB['binNum'] = binNum
                PB['xbin'], PB['ybin'] = xbin, ybin
                PB['endSN'] = endSN
                PB['nPixels'] = nPixels
                # PB['scale'] = scale
                PB['lVal'] = lmin
                PB['lN'] = llen
                PB['lDel'] = dhdr['CD3_3']
                PB['photPA'] = photPA
                PB['scheme'] = binScheme

                uniBins = np.unique(binNum)
                nbins = uniBins.size
                binSpec = np.ma.ones([lPix.size, nbins])*np.nan
                binStat = np.ma.ones([lPix.size, nbins])*np.nan
                binSize = np.ma.ones(nbins, dtype=int)
                binFlux = np.ma.ones(nbins)*np.nan
                for obi in range(nbins):
                    wbin = np.nonzero(binNum == obi)[0]
                    bsize = wbin.size
                    selecSpec = np.take(gspecs, wbin, axis=1)
                    selecStat = np.take(stats, wbin, axis=1)

                    binSpec[:, obi] = np.squeeze(
                        np.ma.sum(np.atleast_2d(selecSpec), axis=1))
                    binStat[:, obi] = np.sqrt(np.squeeze(
                        np.ma.sum(np.atleast_2d(selecStat**2), axis=1)))
                        # sum *variances*
                    binSize[obi] = bsize
                    binFlux[obi] = np.ma.sum(binSpec[:, obi])/bsize
                print('Done.', flush=True)
                PB['binSpec'] = binSpec
                PB['binStat'] = binStat
                PB['binFlux'] = binFlux
                PB['binCounts'] = binSize

                apIdx = np.where(np.sqrt(xp**2 + (yp/fcfg.eps)**2) <= ReMaj)[0]
                aperSpec = np.squeeze(np.ma.sum(np.atleast_2d(
                    np.take(gspecs, apIdx, axis=1)), axis=1))
                aperStat = np.sqrt(np.squeeze(np.ma.sum(np.atleast_2d(
                    np.take(stats, apIdx, axis=1)**2), axis=1)))
                PB['aperSpec'] = aperSpec
                PB['aperStat'] = aperStat
                au.Write.lzma(bofs, PB, preset=6)

                binNumber(galaxy, targetSN, full, binni=PB, dcName=dcName)

                print('Done.', flush=True)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                print(f"LINE {exc_traceback.tb_lineno}\n{'': <4s}{exc_type}\n"\
                      f"{'': <4s}{exc_value}")
                pdb.set_trace()

            gspecs = copy(binSpec)
            stats = copy(binStat)
        else:
            endSN = None
    # By this stage, the defined variables should be
    #   gspecs
    #   stats
    #   xp
    #   yp

    print(f"Spectral Range=[{smin: .3f}, {smax: .3f}]", flush=True)
    fig = plt.figure(figsize=plt.figaspect(1./10.))
    ax = fig.gca()
    ax.plot(lPix, PB['aperSpec'], lw=0.4)
    for pair in smask:
        ax.axvspan(pair[0], pair[1], alpha=0.5, facecolor='r', edgecolor=None,
            fill=True)
    fig.savefig(gDir/'apertureSpecMask.pdf')
    au.Write.lzma(gDir/'apertureSpec.figz', fig)
    fig = plt.figure(figsize=plt.figaspect(1./10.))
    ax = fig.gca()
    ax.plot(lPix, PB['binSpec'][:, 0], lw=0.4)
    for pair in smask:
        ax.axvspan(pair[0], pair[1], alpha=0.5, facecolor='r', edgecolor=None,
            fill=True)
    fig.savefig(gDir/'specCentre.pdf')

    output['lVal'] = lmin
    output['lN'] = llen
    output['lDel'] = dhdr['CD3_3']
    output['dhdr'] = dhdr
    output['shdr'] = shdr
    if sin:
        output['binNum'] = binNum
        output['SN'] = endSN
        output['nPixels'] = nPixels
        # output['scale'] = scale
    try:
        au.Write.lzma(gDir/kfn, output)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        print(f"LINE {exc_traceback.tb_lineno}\n{'': <4s}{exc_type}\n"\
              f"{'': <4s}{exc_value}")
        pdb.set_trace()

    del gspecs, stats

    au.prepSpec(galaxy, targetSN, full=full, instrument=instrument,
        wRange=[smin, smax], smask=smask, dcName=dcName)

    au.alfWrite(galaxy, targetSN, nbins, RZ, qProps=qProps, priors=priors,
        dcName=dcName, qsys=qsys)
    plt.close('all')

# ------------------------------------------------------------------------------

def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if s and len(s) > n else (s or "")

def _run_spec_from_sum(aper: int | str, *, galaxy: str, SN: int, dcName: str,
                       exe_dir: str = ".") -> dict:
    """
    Run spec_from_sum.exe on a single target and capture diagnostics.

    Parameters
    ----------
    mfn : str
        Model/file name argument to pass to the executable.
    galaxy : str
        Galaxy name used to resolve the executable path.
    dcName : str
        Additional directory component used to resolve the path.
    exe_dir : str, optional
        Base directory containing the per-galaxy bin/ folder.

    Returns
    -------
    dict
        Structured result with keys:
        - ok : bool
        - returncode : int
        - signal : int or None
        - signal_name : str or None
        - elapsed_s : float
        - stdout, stderr : str (tail)
        - cmd : list[str]
        - hint : str (optional triage hint when failed)

    Raises
    ------
    None
        All exceptions are caught and folded into the result dict.

    Examples
    --------
    >>> _run_spec_from_sum("NGC3630_SN100_0011", galaxy="NGC3630",
    ...                    dcName="", exe_dir="/mnt/extraspace/poci/alf")
    """
    bin_path = f"{exe_dir}/{galaxy}{dcName}/bin/spec_from_sum.exe"
    try:
        suff = f"{aper:04d}"
    except ValueError:
        suff = f"{aper}"
    mfn = f"{galaxy}_SN{SN:02d}_{suff}"
    cmd = [bin_path, mfn]

    # Ensure we don’t inherit silly thread counts from the parent.
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, "1")

    t0 = time.time()
    res = {'ok': True}
    try:
        if not plp.Path(exe_dir, 'results', f"{mfn}.bestspec2").is_file() and \
            plp.Path(exe_dir, 'results', f"{mfn}.bestspec").is_file():
            p = sp.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            dt = time.time() - t0

            sig = -p.returncode if p.returncode < 0 else None
            sig_name = signal.Signals(sig).name if sig else None

            res = {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "signal": sig,
                "signal_name": sig_name,
                "elapsed_s": dt,
                "stdout": _tail(p.stdout),
                "stderr": _tail(p.stderr),
                "cmd": cmd,
            }

        if not res["ok"]:
            # Helpful triage hints
            if sig_name == "SIGKILL":
                res["hint"] = (
                    "Killed by OS (likely OOM or job limit). Try lowering NMP, "
                    "set OMP/MKL/OPENBLAS threads=1, and check `dmesg`/scheduler logs."
                )
            elif sig_name in {"SIGSEGV", "SIGABRT"}:
                res["hint"] = (
                    "Native crash (SEGV/ABRT). Inspect stderr and consider running "
                    "the command directly under gdb/valgrind."
                )
        return res

    except Exception as e:
        return {
            "ok": False,
            "returncode": 999,
            "signal": None,
            "signal_name": None,
            "elapsed_s": time.time() - t0,
            "stdout": "",
            "stderr": repr(e),
            "cmd": cmd,
            "hint": "Python-side exception while launching subprocess.",
        }

# ------------------------------------------------------------------------------

def _mpSpecFromSum(aper, galaxy, SN, dcName=''):
    try:
        suff = f"{aper:04d}"
    except ValueError:
        suff = f"{aper}"
    mfn = f"{galaxy}_SN{SN:02d}_{suff}"
    if not (curdir/'results'/f"{mfn}.bestspec2").is_file() and \
        (curdir/'results'/f"{mfn}.bestspec").is_file():
        # Generate model on longer wavelength range
        try:
            sp.check_call([
                f"{str(curdir)}/{galaxy}{dcName}/bin/spec_from_sum.exe", mfn])
        except:
            # output to log file
            with open(curdir/f"{galaxy}{dcName}/{mfn}.err", 'a') as f:
                f.write(f"Error in spec_from_sum for {mfn}\n")
                f.write(f"{sys.exc_info()[0]}: {sys.exc_info()[1]}\n")

# ------------------------------------------------------------------------------

def makeSpecFromSum(galaxy: str, SN: int = 100, full=True, NMP=1, apers=[],
    dcName=''):
    if not full: # Clip the spectral data if required
        tEnd = 'trunc'
    else:
        tEnd = 'full'

    SN = int(SN)

    bofs = curdir/f"{galaxy}{dcName}"/f"binning_SN{SN:02d}_{tEnd}.xz"
    if bofs.is_file():
        PB = au.Load.lzma(bofs)
    else:
        PB = au.Load.lzma(curdir/f"{galaxy}{dcName}"/\
            f"voronoi_SN{SN:02d}_{tEnd}.xz")
    nSpat = PB['xbin'].size

    if len(apers) < 1:
        apers = np.arange(nSpat)
    else:
        nSpat = len(apers)

    if NMP > 1:
        print(f"{'': <8s}Running {NMP:d} processes")
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=NMP, maxtasksperchild=1) as pool:
            it = pool.imap_unordered(
                partial(_run_spec_from_sum, galaxy=galaxy, SN=SN, dcName=dcName,
                        exe_dir=str(curdir)),
                apers,
                chunksize=1,
            )
            bad = 0
            for res in tqdm(it, desc="specSum", total=nSpat):
                if not res["ok"]:
                    bad += 1
                    print(
                        f"\n[FAIL] {' '.join(map(str, res['cmd']))}\n"
                        f"  returncode={res['returncode']} signal={res['signal_name']}\n"
                        f"  hint: {res.get('hint','(see stderr)')}\n"
                        f"  --- stderr (tail) ---\n{res['stderr']}\n"
                        f"  --- stdout (tail) ---\n{res['stdout']}\n"
                    )
    else:
        for aper in tqdm(apers, desc='specSum', total=nSpat):
            _run_spec_from_sum(aper, galaxy=galaxy, SN=SN, dcName=dcName,
                exe_dir=str(curdir))

# ------------------------------------------------------------------------------

def afh(galaxy='NGC3115', SN=100, full=True, FOV=True, vsys=False,
    pplots=['kin', 'err', 'age', 'metal', 'imf', 'ml', 'abund', 'radial'],
    filt='WFPC2.F814W', contours=False, dcName='', redraw=False, **kwargs):
    """_summary_

    Args:
        galaxy (str, optional): _description_. Defaults to 'NGC3115'.
        SN (int, optional): _description_. Defaults to 100.
        full (bool, optional): _description_. Defaults to True.
        FOV (bool, optional): _description_. Defaults to True.
        vsys (bool, optional): _description_. Defaults to False.
        pplots (list, optional): _description_. Defaults to ['kin', 'err', 'age', 'metal', 'imf', 'ml', 'abund'].
        band (str, optional): _description_. Defaults to 'F814W'.
        NMP (int, optional): _description_. Defaults to 15.
        contours (bool, optional): toggles whether to show isophotal contours
            on output figures
        dcName (str, optional): a suffix to add to the model directory
    Raises:
        RuntimeError: _description_
    Examples
    --------
    am.afh('SNL1', SN=80, band='F814W', photFilt='WFPC2.F814W', vsys=True, FOV=False, full=True, dcName='NFMESOouterError', posterior=True)
    am.afh('NGC3630', SN=100, filt='WFPC2.F814W', vsys=True, FOV=False, full=True, posterior=True)
    """    

    band = filt.split('.')[-1]
    posterior = kwargs.pop('posterior', False)

    if not full: # Clip the spectral data if required
        tEnd = 'trunc'
    else:
        tEnd = 'full'

    mDir = curdir/f"{galaxy}{dcName}"

    pifs = mDir/f"pixels_SN{SN:02d}.xz"
    bofs = mDir/f"binning_SN{SN:02d}_{tEnd}.xz"
    sefs = mDir/f"selection_SN{SN:02d}_{tEnd}.xz"
    afs  = mDir/f"AFH_SN{SN:02d}_{tEnd}.pkl"
    dkfs  = curdir.parent/'dynamics'/'MUSEKinematics'/f"{galaxy}_SN{SN:02d}.xz"
    kfs  = mDir/f"kins_SN{SN:02d}_{tEnd}.xz"
    sffs = mDir/f"pops_SN{SN:02d}_{tEnd}.xz"
    gfs = curdir.parent/'muse'/'obsData'/f"{galaxy}.xz"
    jfn = dDir/'galaxy-props'/f"{galaxy}.json"
    cfn = mDir/'config.xz'
    xpix, ypix, sele, pixs = au.Load.lzma(pifs)
    if bofs.is_file():
        PB = au.Load.lzma(bofs)
    else:
        PB = au.Load.lzma(mDir/f"voronoi_SN{SN:02d}_{tEnd}.xz")
    saur, goods = au.Load.lzma(sefs)
    CFG = au.Load.lzma(cfn)
    interest = ['fit_type', 'imf_type', 'fit_hermite', 'fit_two_ages']
    print('Run Options:\n'+'\n'.join([f"{'': <4s}{key: >20s}: {CFG[key]}" for
        key in interest]))

    young = bool(int(CFG['fit_two_ages']))
    imft = int(CFG['imf_type'])

    binNum = PB['binNum']
    nSpat = PB['xbin'].size

    gal = au.Load.lzma(gfs)
    if 'z' in gal.keys():
        zShift = gal['z']
        RZ = Redshift(redshift=zShift)
    elif 'distance' in gal.keys():
        distance = gal['distance']
        RZ = Redshift(distance=distance)
    else:
        raise RuntimeError('No distance information.')
    print(RZ)

    print(f"Looking for \n{kfs} and \n{sffs}...")
    outs = np.sort([plp.Path(curdir/'results'/\
        f"{galaxy}_SN{SN:02d}_{xi:04d}.mcmc") for xi in range(nSpat)])
    # iterate over every potential aperture and check for existence individually
    if (not kfs.is_file()) or (not sffs.is_file()) or redraw:
        print(f"Looking for {afs}...")
        if (not afs.is_file()) or redraw:
            print('Generating...')
            ALF = dict()
            for j, out in tqdm(enumerate(outs), desc='Reading ALF',
                    total=nSpat):
                if not out.is_file():
                    print(f"Missing {out}...")
                    ALF[f"{j:04d}"] = None
                    continue
                try:
                    alf = Alf(out.parent/out.stem, mPath=out.parent)
                    alf.get_total_met()
                    alf.normalize_spectra()
                    alf.abundance_correct() # convert [X/H] to [X/Fe]
                    # alf.postwidths = np.std(alf.mcmc, axis=0)
                    delattr(alf, 'mcmc')
                    ALF[f"{j:04d}"] = alf
                except:
                    ALF[f"{j:04d}"] = None
            au.Write.pickl(afs, ALF)
        else:
            print(f"Reading {afs}...")
            ALF = au.Load.pickl(afs)
            incomplete = np.array([ap for ap, aper in enumerate(ALF.keys()) if
                ALF[aper] is None])
            if incomplete.size > 0:
                print(f"Found {incomplete.size} incomplete ALF runs.")
                print(outs[incomplete])
                for j, out in tqdm(enumerate(outs[incomplete]),
                        desc='Re-reading ALF', total=len(incomplete)):
                    try:
                        alf = Alf(out.parent/out.stem, mPath=out.parent)
                        alf.get_total_met()
                        alf.normalize_spectra()
                        alf.abundance_correct() # convert [X/H] to [X/Fe]
                        # alf.postwidths = np.std(alf.mcmc, axis=0)
                        delattr(alf, 'mcmc')
                        ALF[f"{j:04d}"] = alf
                    except:
                        ALF[f"{j:04d}"] = None
                au.Write.pickl(afs, ALF)

        # mIdx = ALF['0000'].results['Type'].tolist().index('mean')
        bestKey = 'cl50'
        mIdx = ALF['0000'].results['Type'].tolist().index(bestKey)
        eIdx = ALF['0000'].results['Type'].tolist().index('error')

        KIN = dict()
        KIN['lVal'] = PB['lVal']
        KIN['lN'] = PB['lN']
        KIN['lDel'] = PB['lDel']
        KIN['x'] = PB['xbin']
        KIN['y'] = PB['ybin']
        for j in range(4):
            KIN[f"{j+1}"] = np.ma.ones(nSpat)*np.nan
            KIN[f"{j+1}e"] = np.ma.ones(nSpat)*np.nan
        SFH = dict()
        _popKeys = ['logage', 'zH', 'FeH', 'a', 'C', 'N', 'Na', 'Mg', 'Si', 'K',
            'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Sr', 'Ba', 'Eu',
            'Teff', 'IMF1', 'IMF2', 'logfy', 'logm7g', 'hotteff', 'loghot',
            'fy_logage', 'logemline_h', 'logemline_oii', 'logemline_oiii',
            'logemline_sii', 'logemline_ni', 'logemline_nii', 'IMF3', 'IMF4',
            'ML_v', 'ML_i', 'ML_k', 'MW_v', 'MW_i', 'MW_k']
        popKeys = np.intersect1d(_popKeys, ALF['0000'].labels)
        # clear elements which were not fit
        SFH['age'] = np.ma.ones(nSpat)*np.nan
        SFH['agee'] = np.ma.ones(nSpat)*np.nan
        SFH['zH'] = np.ma.ones(nSpat)*np.nan
        SFH['zHe'] = np.ma.ones(nSpat)*np.nan
        SFH['FeH'] = np.ma.ones(nSpat)*np.nan
        SFH['FeHe'] = np.ma.ones(nSpat)*np.nan
        SFH['yage'] = np.ma.ones(nSpat)*np.nan
        SFH['yagee'] = np.ma.ones(nSpat)*np.nan
        SFH['fyage'] = np.ma.ones(nSpat)*np.nan
        SFH['fyagee'] = np.ma.ones(nSpat)*np.nan
        SFH['abundances'] = dict()
        aLabels = [r'$[{\rm O/H}]$', r'$[{\rm C/H}]$',
            r'$[{\rm N/H}]$', r'$[{\rm Na/H}]$', r'$[{\rm Mg/H}]$',
            r'$[{\rm Si/H}]$', r'$[{\rm K/H}]$', r'$[{\rm Ca/H}]$',
            r'$[{\rm Ti/H}]$', r'$[{\rm V/H}]$', r'$[{\rm Cr/H}]$',
            r'$[{\rm Mn/H}]$', r'$[{\rm Co/H}]$', r'$[{\rm Ni/H}]$',
            r'$[{\rm Cu/H}]$', r'$[{\rm Sr/H}]$', r'$[{\rm Ba/H}]$',
            r'$[{\rm Eu/H}]$']
        aMask = [ki for ki, key in enumerate(np.take(_popKeys,
            np.arange(2, 20)+1)) if (key in ALF['0000'].labels) and
            (np.ptp([ALF[f"{ap:04d}"].results[key][mIdx] for ap, out in
                enumerate(outs) if not isinstance(ALF[f"{ap:04d}"], type(None))
                ]) > 1e-3)]
        aKeys = np.take(np.take(_popKeys, np.arange(2, 20)+1), aMask)
        aLabels = np.take(aLabels, aMask)
        SFH['abundances']['keys'] = aKeys
        SFH['abundances']['labels'] = aLabels
        for ak in aKeys:
            SFH['abundances'][f"{ak}"] = np.ma.ones(nSpat)*np.nan
            SFH['abundances'][f"{ak}e"] = np.ma.ones(nSpat)*np.nan
        SFH['IMF'] = dict()
        for j in range(4):
            SFH['IMF'][f"{j+1}"] = np.ma.ones(nSpat)*np.nan
            SFH['IMF'][f"{j+1}e"] = np.ma.ones(nSpat)*np.nan
        SFH['ML'] = dict()
        SFH['ML'][band] = np.ma.ones(nSpat)*np.nan

        metDict = {'Fe': SFH['FeH'], **{ak: SFH['abundances'][f"{ak}"]
            for ak in aKeys}}
        metDict['O'] = metDict['a']
        metDict.pop('a')
        SFH['metal'] = np.ma.masked_invalid(au.SumMetals(metDict))

        kinKeys = ['velz', 'sigma', 'h3', 'h4', 'velz2', 'sigma2']

        for aper in tqdm(range(nSpat), desc='Apertures', total=nSpat):
            if ALF[f"{aper:04d}"] is None:
                # skip to next aper
                continue
            for ki in range(4):
                KIN[f"{ki+1}" ][aper] = \
                    ALF[f"{aper:04d}"].results[kinKeys[ki]][mIdx]
                KIN[f"{ki+1}e"][aper] = \
                    ALF[f"{aper:04d}"].results[kinKeys[ki]][eIdx]
                SFH['IMF'][f"{ki+1}"][aper] = \
                    ALF[f"{aper:04d}"].results[f"IMF{ki+1}"][mIdx]
                SFH['IMF'][f"{ki+1}e"][aper] = \
                    ALF[f"{aper:04d}"].results[f"IMF{ki+1}"][eIdx]
            if int(CFG['imf_type']) == 0:
                SFH['IMF']['2'][aper] = SFH['IMF']['1'][aper]
                SFH['IMF']['2e'][aper] = SFH['IMF']['1e'][aper]
            for ak in aKeys:
                SFH['abundances'][f"{ak}"][aper] = \
                    ALF[f"{aper:04d}"].xFe[ak][bestKey] +\
                    ALF[f"{aper:04d}"].results['FeH'][mIdx]
                # use the `corrected` xFe abundances, but convert them back to
                # [X/H] by adding the FeH abundance
                SFH['abundances'][f"{ak}e"][aper] = \
                    ALF[f"{aper:04d}"].results[ak][eIdx]
            SFH['age'][aper] = \
                10.0**(ALF[f"{aper:04d}"].results['logage'][mIdx])
            SFH['agee'][aper] = \
                10.0**(ALF[f"{aper:04d}"].results['logage'][eIdx])
            SFH['yage'][aper] = \
                10.0**(ALF[f"{aper:04d}"].results['fy_logage'][mIdx])
            SFH['yagee'][aper] = \
                10.0**(ALF[f"{aper:04d}"].results['fy_logage'][eIdx])
            SFH['fyage'][aper] = \
                10.0**(ALF[f"{aper:04d}"].results['logfy'][mIdx])
            SFH['fyagee'][aper] = \
                10.0**(ALF[f"{aper:04d}"].results['logfy'][eIdx])
            SFH['zH'][aper] = ALF[f"{aper:04d}"].results['zH'][mIdx]
            SFH['zHe'][aper] = ALF[f"{aper:04d}"].results['zH'][eIdx]
            SFH['FeH'][aper] = ALF[f"{aper:04d}"].results['FeH'][mIdx]
            SFH['FeHe'][aper] = ALF[f"{aper:04d}"].results['FeH'][eIdx]
            # MLa = au.getM2L('solar',
            #     ALF[f"{aper:04d}"].results['logage'][mIdx], SFH['zH'][aper],
            #     SFH['IMF']['1'][aper], SFH['IMF']['2'][aper], 2.3, RZ=RZ,
            #     band=band, **kwargs)
            bs2 = plp.Path(curdir/'results'/\
                f"{galaxy}_SN{SN:02d}_{aper:04d}.bestspec2")
            if bs2.is_file():
                MLa = au.getM2L(f"{galaxy}_SN{SN:02d}_{aper:04d}",
                    ALF[f"{aper:04d}"].results['logage'][mIdx], SFH['zH'][aper],
                    SFH['IMF']['1'][aper], SFH['IMF']['2'][aper], 2.3, RZ=RZ,
                    filt=filt, **kwargs)
                SFH['ML'][band][aper] = MLa

        KIN['2'] = np.sqrt(KIN['2']**2 + 100.**2) # add model broadening
        au.Write.lzma(kfs, KIN)
        au.Write.lzma(sffs, SFH)
        su.copy2(kfs,
            dDir/'MUSEKinematics'/str(kfs.name).replace('kins', galaxy))
    else:
        KIN = au.Load.lzma(kfs)
        SFH = au.Load.lzma(sffs)


    gfs = curdir.parent/'muse'/'obsData'/f"{galaxy}.xz"
    gal = au.Load.lzma(gfs)

    aKeys = SFH['abundances']['keys']

    if contours:
        with pf.open(mDir/f"collapsed.fits") as cdu:
            fluxii = cdu[0].data
        flux = np.compress(goods, np.compress(sele, fluxii.ravel()))
        flevels = np.ma.max(flux)*10**(-0.4*np.arange(0, 14, 0.5)[::-1])
    xbin, ybin = KIN['x'], KIN['y']
    if vsys:
        print('Determining systemic velocity...')
        if 'FCC170' in galaxy:
            vMask = ((xbin < 0) & (xbin > -5) & (ybin < -55)) |\
                (np.sqrt((xbin-37)**2 + (ybin--17)**2) < 10.)
        else:
            vMask = np.zeros_like(xbin, dtype=bool)
        circ = np.sqrt(xbin**2 + ybin**2)
        ww = np.where(circ < np.min([5., circ.max()/2.])) # for FOV smaller
            # than 5''
        mVel = np.ma.masked_invalid(np.ma.masked_array(KIN['1'], vMask))
        _vSys = np.ma.median(mVel[ww])
        vMask = np.ma.getmaskarray(mVel)
        mVel = mVel[~vMask]
        plt.clf()
        angBest, angErr, vSys = fkpa(xbin[~vMask], ybin[~vMask], mVel-_vSys,
            quiet=True, plot=True, nsteps=int((360*2)+1))
        plt.savefig(mDir/f"fitPA_SN{SN:02d}")
        plt.close('all')
        vSys += _vSys
        gal['vSys'] = vSys
        if angErr:
            gal['PA'] = 90.+PB['photPA']
        else: gal['PA'] = 90.-angBest
        PA = gal['PA']
        au.Write.lzma(gfs, gal)
        print(f"{'': <4s}kinPA: {90.-angBest: 4.4} +/- {angErr: 4.4}")
        print(f"{'': <4s}phtPA: {PB['photPA']: 4.4}")
        print(f"Systemic velocity determined to be {vSys:4.4f} km s^{{-1}}")
    else:
        if 'vSys' in gal.keys():
            vSys = gal['vSys']  # systemic velocity estimate
        else:
            if 'z' in gal.keys():
                vSys = np.log(gal['z']+1)*CTS.c
        print(f"Systemic velocity read in as {vSys:4.4f} km s^{{-1}}")
        PA = gal['PA']
    KIN['1'] -= vSys
    # Plots
    xbix, ybix = GEO.rotate2D(xpix, ypix, PA)
    pfn = dDir.parent/'muse'/'obsData'/f"{galaxy}{dcName}-poly-rot.xz"
    if pfn.is_file():
        aShape = au.Load.lzma(pfn)
        aShape, pPatch = POT.polyPatch(POLYGON=aShape, Xpo=xbix, Ypo=ybix,
            salpha=0.5, ec=POT.brown, linestyle='--', fill=False, zorder=0,
            lw=0.75)
    else:
        aShape, pPatch = POT.polyPatch(Xpo=xbix, Ypo=ybix, salpha=0.5,
            ec=POT.brown, linestyle='--', fill=False, zorder=0, lw=0.75)
        au.Write.lzma(pfn, aShape)
    xmin, xmax = np.amin(xbix), np.amax(xbix)
    ymin, ymax = np.amin(ybix), np.amax(ybix)
    xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    xbix = np.compress(goods, xbix)
    ybix = np.compress(goods, ybix)
    if not FOV:
        xmin, xmax = np.amin(xbix), np.amax(xbix)
        ymin, ymax = np.amin(ybix), np.amax(ybix)
        xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels

    nMom = 2
    if bool(int(CFG['fit_hermite'])):
        nMom = 4
    if xLen < yLen:
        cDim = np.ceil(np.sqrt(nMom)).astype(int)
        rema = nMom % cDim
        rDim = np.floor((nMom-rema)/cDim).astype(int)
    else:
        cDim = np.floor(np.sqrt(nMom)).astype(int)
        rema = nMom % cDim
        rDim = np.ceil((nMom-rema)/cDim).astype(int)

    pren = 2
    aspect = (rDim*yLen)/(cDim*xLen)

    vmin, vmax = POT.sigClip(KIN['1'], 'V', clipBins=0.05)
    dmin, dmax = POT.sigClip(KIN['2'], r'σ', clipBins=0.05)
    vmax = np.ceil(np.max([np.abs(vmin), vmax])/5)*5
    vmin = -vmax
    dmin = np.floor(dmin/20)*20
    dmax = np.ceil(dmax/10)*10
    lims = [[vmin, vmax], [dmin, dmax]]
    mome = ['V', r'\sigma']
    units = [fr"[{UTS.kms1}]", fr"[{UTS.kms1}]"]
    for j in range(nMom-2):
        lims += [[-0.3, 0.3]]
        mome += [fr"h{j+3:d}"]
        units += ['']

    arcRA = r'$x\ [{\rm arcsec}]$'
    pcRA = fr"$x\ [{UTS.kpace}]$"
    arcDec = r'$y\ [{\rm arcsec}]$'
    pcDec = fr"$y\ [{UTS.kpace}]$"
    pc = RZ.getPC()
    akpc = pc * 1e-3

    pKeys = np.unique(np.append(aKeys, ['logage', 'FeH', 'IMF1',
        'IMF2', 'zH']))
    if imft == 0:
        pKeys = np.delete(pKeys, np.where(pKeys=='IMF2')[0])
    
    eps = 1e-3
    IMF1 = SFH['IMF']['1'].astype(np.float64, copy=True)
    m1 = (IMF1 == 1)
    if m1.any():
        IMF1[m1] += (np.random.random(m1.sum()) - 0.5) * eps
    imin, imax = POT.sigClip(IMF1, 'IMF', clipBins=0.025)
    IMF2 = SFH['IMF']['2'].astype(np.float64, copy=True)
    m2 = (IMF2 == 1)
    if m2.any():
        IMF2[m2] += (np.random.random(m2.sum()) - 0.5) * eps
    imfs = [pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.0),
        slopes=(x1, x2, 2.3)) for (x1, x2) in zip(IMF1, IMF2)]
    xiTop = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=0.5)[0], imfs)))
    xiBot = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=1.0)[0], imfs)))
    xi = xiTop/xiBot
    ximin, ximax = POT.sigClip(xi, 'IMF', clipBins=0.025)

    regex = re.compile(rf"^{galaxy}_SN{SN:02d}_[0-9]+.mcmc$")
    outs = [dp for dp in plp.Path(curdir/'results').iterdir() if
        regex.match(dp.name)]

    if 'kin' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        axs, kaxs = [], []
        for mm in tqdm(range(nMom)):
            if (mm+1) == 2:
                vcmap = moncmap
                ytc = 'w'
            else:
                vcmap = divcmap
                ytc = 'k'

            ax = fig.add_subplot(gs[mm])
            lmi, lma = lims[mm]
            lab = r'\ '.join([ql for ql in [mome[mm], units[mm]] if ql != ''])

            img = dpp(xpix, ypix, (KIN[f"{mm+1:d}"][binNum]), pixelsize=pixs,
                vmin=lmi, vmax=lma, angle=PA, cmap=vcmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[mm], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)

            miText = POT.prec(pren, lmi)
            maText = POT.prec(pren, lma)
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = ax.text(5e-2, 1-1e-2, rf"$\mathbf{{{lab}}}$", va='top',
                ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color=ytc, transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            axs += [ax]
            kaxs += [_]
        for ax, pax in zip(axs, kaxs):
            ax.tick_params('both', which='major', width=0.75, length=5)
            # ax.tick_params('both', length=4, which='minor')
            pax.tick_params(bottom=False, top=True, left=False, right=True,
                labelright=True, labeltop=True, labelbottom=False,
                labelleft=False, width=0.75, length=5)
            pax.xaxis.set_label_position('top')
            pax.yaxis.set_label_position('right')
            pax.xaxis.set_ticks_position('top')
            pax.yaxis.set_ticks_position('right')
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if not pax.get_subplotspec().is_first_row():
                pax.set_xticklabels([])
            pax.set_yticklabels([])
            ax.minorticks_off()
            pax.minorticks_off()

        fig.savefig(mDir/f"kinematics_4_SN{SN:02d}")
        plt.close('all')

    if 'err' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        axs, kaxs = [], []

        evmin, evmax = POT.sigClip(KIN['1e'], 'v_error', clipBins=0.05)
        edmin, edmax = POT.sigClip(KIN['2e'], 'd_error', clipBins=0.05)
        evmin = np.floor(evmin/20)*20
        evmax = np.ceil(evmax/5)*5
        edmin = np.floor(edmin/20)*20
        edmax = np.ceil(edmax/10)*10
        elims = [[evmin, evmax], [edmin, edmax]]
        for _m in range(2, nMom):
            elims += [[0., 0.3]]

        print('Plotting moments...')
        for mm in tqdm(range(nMom)):

            ax = fig.add_subplot(gs[mm])
            emin, emax = elims[mm]
            lab = r'\ '.join([ql for ql in [fr"\delta({mome[mm]})",
                units[mm]] if ql != ''])

            img = dpp(xpix, ypix, (KIN[f"{mm+1:d}e"][binNum]), pixelsize=pixs,
                vmin=emin, vmax=emax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[mm], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)

            miText = POT.prec(pren, emin)
            maText = POT.prec(pren, emax)
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = ax.text(5e-2, 1-1e-2, rf"$\mathbf{{{lab}}}$", va='top',
                ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='k', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            axs += [ax]
            kaxs += [_]
        for ax, pax in zip(axs, kaxs):
            ax.tick_params('both', which='major', width=0.75, length=5)
            # ax.tick_params('both', length=4, which='minor')
            pax.tick_params(bottom=False, top=True, left=False, right=True,
                labelright=True, labeltop=True, labelbottom=False,
                labelleft=False, width=0.75, length=5)
            pax.xaxis.set_label_position('top')
            pax.yaxis.set_label_position('right')
            pax.xaxis.set_ticks_position('top')
            pax.yaxis.set_ticks_position('right')
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if not pax.get_subplotspec().is_first_row():
                pax.set_xticklabels([])
            pax.set_yticklabels([])
            ax.minorticks_off()
            pax.minorticks_off()

        fig.savefig(mDir/f"kinematicErrors_4_SN{SN:02d}")
        plt.close('all')

    if 'age' in pplots:
        mwage = np.ma.average(np.column_stack((SFH['age'], SFH['yage'])),
            weights=np.column_stack((1-SFH['fyage'], SFH['fyage'])), axis=1)
        mmin, mmax = POT.sigClip(mwage, 'mwage', clipBins=0.05)
        amin, amax = POT.sigClip(SFH['age'], 'age', clipBins=0.05)
        jmin, jmax = POT.sigClip(SFH['yage'], 'yage', clipBins=0.05)
        fmin, fmax = POT.sigClip(SFH['fyage'], 'fyage', clipBins=0.05)

        if young:
            gs = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.0)
            mainAge = mwage
            cbmid = True
        else:
            gs = gridspec.GridSpec(1, 1, hspace=0.0, wspace=0.0)
            mainAge = SFH['age']
            cbmid = False
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        axs, kaxs = [], []

        ax = fig.add_subplot(gs[0])
        img = dpp(xpix, ypix, mainAge[binNum], pixelsize=pixs, vmin=mmin,
            vmax=mmax, angle=PA, cmap=moncmap)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        _ = fig.add_subplot(gs[0], adjustable='datalim')
        _.set_xlim(np.array(ax.get_xlim()) * akpc)
        _.set_ylim(np.array(ax.get_ylim()) * akpc)
        _.set_aspect('equal', adjustable='box')
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)
        if not ax.get_subplotspec().is_last_row():
            ax.set_xticklabels([])
        if not ax.get_subplotspec().is_first_col():
            ax.set_yticklabels([])
        miText = POT.prec(pren, mmin)
        maText = POT.prec(pren, mmax)
        cax = POT.attachAxis(ax, 'right', 0.075, mid=cbmid)
        cb = plt.colorbar(img, cax=cax)
        lT = ax.text(1e-2, 1-1e-2, rf"$\mathbf{{t\ [{UTS.gyr}]}}$", va='top',
            ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='w', transform=cax.transAxes)
        cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='k', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)
        axs += [ax]
        kaxs += [_]

        if young:
            ax = fig.add_subplot(gs[1])
            img = dpp(xpix, ypix, SFH['age'][binNum], pixelsize=pixs, vmin=amin,
                vmax=amax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[1], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, amin)
            maText = POT.prec(pren, amax)
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = ax.text(5e-2, 1-1e-2, rf"$\mathbf{{t\ [{UTS.gyr}]}}$", va='top',
                ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='k', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)
            axs += [ax]
            kaxs += [_]

            ax = fig.add_subplot(gs[2])
            img = dpp(xpix, ypix, SFH['yage'][binNum], pixelsize=pixs,
                vmin=jmin, vmax=jmax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[2], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, jmin)
            maText = POT.prec(pren, jmax)
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = ax.text(1e-2, 1-1e-2, rf"$\mathbf{{t_y\ [{UTS.gyr}]}}$",
                va='top', ha='left', color=POT.pgreen, transform=ax.transAxes,
                zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='k', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)
            axs += [ax]
            kaxs += [_]

            ax = fig.add_subplot(gs[3])
            img = dpp(xpix, ypix, SFH['fyage'][binNum], pixelsize=pixs,
                vmin=fmin, vmax=fmax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[3], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            lT = ax.text(5e-2, 1-1e-2, r'$\mathbf{f_y}$', va='top', ha='left',
                color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, fmin)
            maText = POT.prec(pren, fmax)
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            # lT = cax.text(0.5, 0.5, r"$f_y$", va='center', ha='center',
            #     rotation=270, color=POT.pgreen, transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='k', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)
            axs += [ax]
            kaxs += [_]

        for ax, pax in zip(axs, kaxs):
            ax.tick_params('both', which='major', width=0.75, length=5)
            # ax.tick_params('both', length=4, which='minor')
            pax.tick_params(bottom=False, top=True, left=False, right=True,
                labelright=True, labeltop=True, labelbottom=False,
                labelleft=False, width=0.75, length=5)
            pax.xaxis.set_label_position('top')
            pax.yaxis.set_label_position('right')
            pax.xaxis.set_ticks_position('top')
            pax.yaxis.set_ticks_position('right')
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if not pax.get_subplotspec().is_first_row():
                pax.set_xticklabels([])
            pax.set_yticklabels([])
            ax.minorticks_off()
            pax.minorticks_off()

        fig.savefig(mDir/f"afh_age_SN{SN:02d}")
        plt.close('all')

    if 'metal' in pplots:
        amin, amax = POT.sigClip(SFH['FeH'], 'metal', clipBins=0.05)
        gs = gridspec.GridSpec(1, 1, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        ax = fig.add_subplot(gs[0])
        img = dpp(xpix, ypix, SFH['FeH'][binNum], pixelsize=pixs,
            vmin=amin, vmax=amax, angle=PA, cmap=moncmap)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        _ = fig.add_subplot(gs[0], adjustable='datalim')
        _.set_xlim(np.array(ax.get_xlim()) * akpc)
        _.set_ylim(np.array(ax.get_ylim()) * akpc)
        _.set_aspect('equal', adjustable='box')
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        miText = POT.prec(pren, amin)
        maText = POT.prec(pren, amax)
        cax = POT.attachAxis(ax, 'right', 0.075)
        cb = plt.colorbar(img, cax=cax)
        lT = ax.text(1e-2, 1-1e-2, r'$\mathbf{[\mathrm{Fe/H}]}$', va='top',
            ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='w', transform=cax.transAxes)
        cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='k', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        ax.tick_params('both', which='major', width=0.75, length=5)
        # ax.tick_params('both', length=4, which='minor')
        _.tick_params(bottom=False, top=True, left=False, right=True,
            labelright=True, labeltop=True, labelbottom=False,
            labelleft=False, width=0.75, length=5)
        _.xaxis.set_label_position('top')
        _.yaxis.set_label_position('right')
        _.xaxis.set_ticks_position('top')
        _.yaxis.set_ticks_position('right')
        if not ax.get_subplotspec().is_last_row():
            ax.set_xticklabels([])
        if not ax.get_subplotspec().is_first_col():
            ax.set_yticklabels([])
        if not _.get_subplotspec().is_first_row():
            _.set_xticklabels([])
        _.set_yticklabels([])
        ax.minorticks_off()
        _.minorticks_off()

        fig.savefig(mDir/f"afh_metal_SN{SN:02d}")
        plt.close('all')


        amin, amax = POT.sigClip(SFH['metal'], 'metal', clipBins=0.05)
        gs = gridspec.GridSpec(1, 1, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        ax = fig.add_subplot(gs[0])
        img = dpp(xpix, ypix, SFH['metal'][binNum], pixelsize=pixs,
            vmin=amin, vmax=amax, angle=PA, cmap=moncmap)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        _ = fig.add_subplot(gs[0], adjustable='datalim')
        _.set_xlim(np.array(ax.get_xlim()) * akpc)
        _.set_ylim(np.array(ax.get_ylim()) * akpc)
        _.set_aspect('equal', adjustable='box')
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        miText = POT.prec(pren, amin)
        maText = POT.prec(pren, amax)
        cax = POT.attachAxis(ax, 'right', 0.075)
        cb = plt.colorbar(img, cax=cax)
        lT = ax.text(1e-2, 1-1e-2, r'$\mathbf{[\mathrm{Z/H}]}$', va='top',
            ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='w', transform=cax.transAxes)
        cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='k', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        ax.tick_params('both', which='major', width=0.75, length=5)
        # ax.tick_params('both', length=4, which='minor')
        _.tick_params(bottom=False, top=True, left=False, right=True,
            labelright=True, labeltop=True, labelbottom=False,
            labelleft=False, width=0.75, length=5)
        _.xaxis.set_label_position('top')
        _.yaxis.set_label_position('right')
        _.xaxis.set_ticks_position('top')
        _.yaxis.set_ticks_position('right')
        if not ax.get_subplotspec().is_last_row():
            ax.set_xticklabels([])
        if not ax.get_subplotspec().is_first_col():
            ax.set_yticklabels([])
        if not _.get_subplotspec().is_first_row():
            _.set_xticklabels([])
        _.set_yticklabels([])
        ax.minorticks_off()
        _.minorticks_off()

        fig.savefig(mDir/f"afh_totalmetal_SN{SN:02d}")
        plt.close('all')

    if 'imf' in pplots:

        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        if imft == 1 or imft == 3:

            i2min, i2max = POT.sigClip(IMF2, 'IMF', clipBins=0.025)
            imin = np.min((imin, i2min))
            imax = np.max((imax, i2max))
            gs = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.0)

            ax = fig.add_subplot(gs[0])
            lT = ax.text(1e-2, 1-1e-2, r'$\mathbf{\alpha_1}$', va='top',
                ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
            cbmid = True
        else:
            gs = gridspec.GridSpec(1, 1, hspace=0.0, wspace=0.0)
            ax = fig.add_subplot(gs[0])
            cbmid = False

        img = dpp(xpix, ypix, IMF1[binNum], pixelsize=pixs,
            vmin=imin, vmax=imax, angle=PA, cmap=moncmap)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        _ = fig.add_subplot(gs[0], adjustable='datalim')
        _.set_xlim(np.array(ax.get_xlim()) * akpc)
        _.set_ylim(np.array(ax.get_ylim()) * akpc)
        _.set_aspect('equal', adjustable='box')
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        axs, kaxs = [], []

        miText = POT.prec(pren, imin)
        maText = POT.prec(pren, imax)
        cax = POT.attachAxis(ax, 'right', 0.075, mid=cbmid)
        cb = plt.colorbar(img, cax=cax)
        # lT = cax.text(0.5, 0.5, r'$\alpha$', va='center',
        #     ha='center', rotation=270, color=POT.pgreen,
        #     transform=cax.transAxes)
        # lT.set_path_effects(
        #     [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='w', transform=cax.transAxes)
        cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='k', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)
        axs += [ax]
        kaxs += [_]

        if imft == 1 or imft == 3:
            ax = fig.add_subplot(gs[1])
            img = dpp(xpix, ypix, IMF2[binNum], pixelsize=pixs,
                vmin=imin, vmax=imax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[1], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            lT = ax.text(5e-2, 1-1e-2, r'$\mathbf{\alpha_2}$', va='top',
                ha='left', color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            axs += [ax]
            kaxs += [_]
            
            ax = fig.add_subplot(gs[2])
            img = dpp(xpix, ypix, xi[binNum], pixelsize=pixs,
                vmin=ximin, vmax=ximax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[2], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            lT = ax.text(1e-2, 1-1e-2, r'$\mathbf{\xi}$', va='top', ha='left',
                color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)

            miText = POT.prec(pren, ximin)
            maText = POT.prec(pren, ximax)
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            # lT = cax.text(0.5, 0.5, r'$\mathbf{\xi}$', va='center',
            #     ha='center', rotation=270, color=POT.pgreen,
            #     transform=cax.transAxes)
            # lT.set_path_effects(
            #     [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='k', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)
            axs += [ax]
            kaxs += [_]

        for ax, pax in zip(axs, kaxs):
            ax.tick_params('both', which='major', width=0.75, length=5)
            # ax.tick_params('both', length=4, which='minor')
            pax.tick_params(bottom=False, top=True, left=False, right=True,
                labelright=True, labeltop=True, labelbottom=False,
                labelleft=False, width=0.75, length=5)
            pax.xaxis.set_label_position('top')
            pax.yaxis.set_label_position('right')
            pax.xaxis.set_ticks_position('top')
            pax.yaxis.set_ticks_position('right')
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if not pax.get_subplotspec().is_first_row():
                pax.set_xticklabels([])
            pax.set_yticklabels([])
            ax.minorticks_off()
            pax.minorticks_off()
        
        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        fig.savefig(mDir/f"afh_IMF_SN{SN:02d}")


        da = []
        sa = []
        # Histogram the slopes and the uncertainties
        if imft == 1 or imft == 3:
            gs = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.0)
            fig = plt.figure(figsize=plt.figaspect(1.0))
        else:
            gs = gridspec.GridSpec(1, 2, hspace=0.0, wspace=0.0)
            fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(gs[0])
        ax.set_yticks([])
        ax.hist(IMF1, bins=20, histtype='stepfilled', color='k', lw=2)
        lT = ax.text(1-1e-2, 1-1e-2, r"$\mathbf{\alpha_1}$"'\n'\
            rf"$\mathbf{{\sigma = {np.std(IMF1):.3f}}}$", va='top', ha='right',
            color=POT.pgreen, transform=ax.transAxes, zorder=200)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        da += [ax]
        ax = fig.add_subplot(gs[1])
        ax.set_yticks([])
        ax.hist(SFH['IMF']['1e'], bins=20, histtype='stepfilled', color='k',
            lw=2)
        lT = ax.text(1-1e-2, 1-1e-2, r"$\mathbf{\sigma_{\alpha_1}}$", va='top',
            ha='right', color=POT.pgreen, transform=ax.transAxes, zorder=200)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        sa += [ax]
        if imft == 1 or imft == 3:
            ax = fig.add_subplot(gs[2])
            ax.set_yticks([])
            ax.hist(IMF2, bins=20, histtype='stepfilled', color='k', lw=2)
            lT = ax.text(1e-2, 1-1e-2, r"$\mathbf{\alpha_2}$"'\n'\
            rf"$\mathbf{{\sigma = {np.std(IMF2):.3f}}}$", va='top', ha='left',
                color=POT.pgreen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            da += [ax]
            ax = fig.add_subplot(gs[3])
            ax.set_yticks([])
            ax.hist(SFH['IMF']['2e'], bins=20, histtype='stepfilled', color='k',
                lw=2)
            lT = ax.text(1-1e-2, 1-1e-2, r"$\mathbf{\sigma_{\alpha_2}}$",
                va='top', ha='right', color=POT.pgreen, transform=ax.transAxes,
                zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            sa += [ax]
        dxmin, dxmax = [np.min([dax.get_xlim()[0] for dax in da]),
            np.max([dax.get_xlim()[1] for dax in da])]
        for dax in da:
            dax.set_xlim(dxmin, dxmax)
            dax.tick_params('both', which='major', width=0.75, length=5)
            # dax.tick_params('both', length=4, which='minor')
            if not dax.get_subplotspec().is_last_row():
                dax.set_xticklabels([])
        sxmin, sxmax = [np.min([sax.get_xlim()[0] for sax in sa]),
            np.max([sax.get_xlim()[1] for sax in sa])]
        for sax in sa:
            sax.set_xlim(sxmin, sxmax)
            sax.tick_params('both', which='major', width=0.75, length=5)
            # sax.tick_params('both', length=4, which='minor')
            if not sax.get_subplotspec().is_last_row():
                sax.set_xticklabels([])
        fig.savefig(mDir/f"afh_IMFhist_SN{SN:02d}")

        plt.close('all')

    if 'ml' in pplots:

        amin, amax = POT.sigClip(SFH['ML'][band], f'ML_{band}', clipBins=0.08)
        gs = gridspec.GridSpec(1, 1, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        ax = fig.add_subplot(gs[0])
        img = dpp(xpix, ypix, SFH['ML'][band][binNum], pixelsize=pixs,
            vmin=amin, vmax=amax, angle=PA, cmap=moncmap)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        _ = fig.add_subplot(gs[0], adjustable='datalim')
        _.set_xlim(np.array(ax.get_xlim()) * akpc)
        _.set_ylim(np.array(ax.get_ylim()) * akpc)
        _.set_aspect('equal', adjustable='box')
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        miText = POT.prec(pren, amin)
        maText = POT.prec(pren, amax)
        cax = POT.attachAxis(ax, 'right', 0.075)
        cb = plt.colorbar(img, cax=cax)
        lT = ax.text(1e-2, 1-1e-2,
            rf"$\mathbf{{M/L_{{{band}}}\ [{UTS.msun}/{UTS.lsun}]}}$",
            va='top', ha='left', color=POT.pgreen, transform=ax.transAxes,
            zorder=200)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='w', transform=cax.transAxes)
        cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='k', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        ax.tick_params('both', which='major', width=0.75, length=5)
        # ax.tick_params('both', length=4, which='minor')
        _.tick_params(bottom=False, top=True, left=False, right=True,
            labelright=True, labeltop=True, labelbottom=False,
            labelleft=False, width=0.75, length=5)
        _.xaxis.set_label_position('top')
        _.yaxis.set_label_position('right')
        _.xaxis.set_ticks_position('top')
        _.yaxis.set_ticks_position('right')
        if not ax.get_subplotspec().is_last_row():
            ax.set_xticklabels([])
        if not ax.get_subplotspec().is_first_col():
            ax.set_yticklabels([])
        if not _.get_subplotspec().is_first_row():
            _.set_xticklabels([])
        _.set_yticklabels([])
        ax.minorticks_off()
        _.minorticks_off()

        fig.savefig(mDir/f"afh_ML{band}_SN{SN:02d}")
        plt.close('all')

    if 'abund' in pplots:
        nAbund = len(aKeys)
        dim = np.ceil(np.sqrt(nAbund)).astype(int)
        rema = nAbund % dim
        lo = np.floor((nAbund - rema) / dim).astype(int)
        if lo * dim < nAbund:
            lo += 1
        if (lo*dim) % nAbund > dim/2:
            dim += 1 # wider rather than taller
            lo -= 1

        gs = gridspec.GridSpec(lo, dim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(lo*yLen/(dim*xLen)) * nAbund/9.5)

        arcB = fig.add_subplot(gs[:], adjustable='box')
        arcB.set_frame_on(False)
        arcB.set_xticks([])
        arcB.set_yticks([])
        arcB.set_xlabel(arcRA, labelpad=20)
        arcB.set_ylabel(arcDec, rotation=90, labelpad=30)

        pcB = fig.add_subplot(gs[:], adjustable='datalim')
        pcB.tick_params(bottom='off', top='on', left='off', right='on',
            labelright='on', labeltop='on', labelbottom='off', labelleft='off')
        pcB.xaxis.set_label_position('top')
        pcB.yaxis.set_label_position('right')
        pcB.set_frame_on(False)
        pcB.set_xticks([])
        pcB.set_yticks([])
        pcB.set_xlabel(pcRA, labelpad=30)
        # pcB.set_ylabel(pcDec, rotation=270, labelpad=40)

        axs, kaxs = [], []

        for ai, key in enumerate(aKeys):
            abund = SFH['abundances'][key]
            label = SFH['abundances']['labels'][ai]

            amin, amax = POT.sigClip(abund, key, clipBins=0.05)
            ax = fig.add_subplot(gs[ai])
            img = dpp(xpix, ypix, abund[binNum], pixelsize=pixs,
                vmin=amin, vmax=amax, angle=PA, cmap=moncmap)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            _ = fig.add_subplot(gs[ai], adjustable='datalim')
            _.set_xlim(np.array(ax.get_xlim()) * akpc)
            _.set_ylim(np.array(ax.get_ylim()) * akpc)
            _.set_aspect('equal', adjustable='box')
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, amin)
            maText = POT.prec(pren, amax)
            lT = ax.text(6e-2, 1-1e-2, rf"$\mathbf{{{label.strip('$')}}}$",
                va='top', ha='left', color=POT.pgreen, transform=ax.transAxes,
                zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax = POT.attachAxis(ax, 'right', 0.075, mid=True)
            cb = plt.colorbar(img, cax=cax)
            cax.text(0.43, 5e-3, miText, va='bottom', ha='center',
                rotation=270, color='w', transform=cax.transAxes)
            cax.text(0.43, 1.-5e-3, maText, va='top', ha='center',
                rotation=270, color='k', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)
            axs += [ax]
            kaxs += [_]
        for ax, pax in zip(axs, kaxs):
            ax.tick_params('both', which='major', width=0.75, length=5)
            # ax.tick_params('both', length=4, which='minor')
            pax.tick_params(bottom=False, top=True, left=False, right=True,
                labelright=True, labeltop=True, labelbottom=False,
                labelleft=False, width=0.75, length=5)
            pax.xaxis.set_label_position('top')
            pax.yaxis.set_label_position('right')
            pax.xaxis.set_ticks_position('top')
            pax.yaxis.set_ticks_position('right')
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            if not pax.get_subplotspec().is_first_row():
                pax.set_xticklabels([])
            pax.set_yticklabels([])
            ax.minorticks_off()
            pax.minorticks_off()

        fig.savefig(mDir/f"afh_elements_SN{SN:02d}")
        plt.close('all')

    if 'radial' in pplots:
        if posterior:
            posfn = mDir/f"afh_elements_posteriors_SN{SN:02d}.xz"
            nPost = 500
            if not posfn.is_file():
                pKeys = np.unique(np.append(aKeys, ['logage', 'FeH', 'IMF1',
                    'IMF2', 'zH']))
                if imft == 0:
                    pKeys = np.delete(pKeys, np.where(pKeys=='IMF2')[0])
                maps = dict()
                for ai, key in enumerate(pKeys):
                    maps[key] = np.ma.ones((nSpat, nPost))*np.nan
                maps['ML'] = np.ma.ones((nSpat, nPost))*np.nan
                for j, out in tqdm(enumerate(outs), total=len(outs),
                    desc='Generating posterior samples'):
                    alf = Alf(out.parent/out.stem, mPath=out.parent)
                    alf.get_total_met()
                    alf.normalize_spectra()
                    # alf.abundance_correct() # convert [X/H] to [X/Fe]
                    for ai, key in enumerate(pKeys):
                        aidx = alfFP.index(key)
                        maps[key][j, :] = np.random.choice(alf.mcmc[:, aidx],
                            nPost, replace=False)
                    # maps['ML'][j, :] = au.getM2L(
                    #   f"{galaxy}_SN{SN:02d}_{j:04d}",
                    #     np.random.choice(alf.mcmc[:, alfFP.index('logage')],
                    #         nPost, replace=False),
                    #     np.random.choice(alf.mcmc[:, alfFP.index('zH')],
                    #         nPost, replace=False),
                    #     np.random.choice(alf.mcmc[:, alfFP.index('IMF1')],
                    #         nPost, replace=False),
                    #     np.random.choice(alf.mcmc[:, alfFP.index('IMF2')],
                    #         nPost, replace=False),
                    #     np.repeat(2.3, nPost), RZ=RZ, band=band, **kwargs)
                    if out.with_suffix('.bestspec2').is_file():
                        if imft == 0:
                            maps['ML'][j, :] = au.getM2L(
                                f"{galaxy}_SN{SN:02d}_{j:04d}",
                                maps['logage'][j, :], maps['zH'][j, :],
                                maps['IMF1'][j, :], maps['IMF1'][j, :],
                                np.repeat(2.3, nPost), RZ=RZ, filt=filt,
                                    **kwargs)
                        else:
                            maps['ML'][j, :] = au.getM2L(
                                f"{galaxy}_SN{SN:02d}_{j:04d}",
                                maps['logage'][j, :], maps['zH'][j, :],
                                maps['IMF1'][j, :], maps['IMF2'][j, :],
                                np.repeat(2.3, nPost), RZ=RZ, filt=filt,
                                    **kwargs)
                    # use these random samples, don't re-sample.
                au.Write.lzma(posfn, maps)
            else:
                maps = au.Load.lzma(posfn)
        nrad = 12
        regex = re.compile(rf'^{galaxy}*.mge$')
        mgefile = [dp for dp in (dDir.parent/'muse'/'obsData').iterdir()
            if regex.match(dp.name) if dp.is_file() and '-mass' not in dp.name]
        if len(mgefile) == 0:
            eps = 0.75
        else:
            sMGE = au.Load.mge(mgefile[0])
            eps = sMGE.epsE
        rade = np.sqrt(xbin**2 + (ybin/eps)**2) # elliptical radius
        rore = np.argsort(rade)
        # rade = np.ma.masked_invalid(np.log10(rade[rore]))
        rade = np.ma.masked_invalid(rade[rore])
        medBins = np.linspace(*POT.sigClip(np.append(1e-3, rade), 'radius', 0.1), nrad+1)
        delta = medBins[1:] - medBins[:-1]
        idx = np.digitize(rade, medBins[:-1])
        pBins = medBins[1:] - delta/2
        symbs = ['X', 'p', '^', '<', '>', '8', 's', 'D', 'P', '*', 'h', 'H',
            '+', 'x', 'o', 'v', r'$\ast$', '.', 'd', r'$\ddagger$', r'$\scurel$',
            r'$\maltese$', r'$\flat$', r'$\natural$']+ list(np.arange(4)+4) + \
            ['1', '3']
        colos = plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['#0b89d5',
            '#DE3163', '#DFFF00', '#FF00FF', '#ff9966', '#520e25', '#000000',
            '#b03516', '#a8896e', '#ff9900', '#21301c', '#e67e22', '#f5b7b1',
            '#d6eaf8', '#0b5345', '#4a235a']
        # dcols = np.tile(plt.rcParams['axes.prop_cycle'].by_key()[
            # 'color'][::-1], 3)[:len(symbs)]
        # colmar = [pair for pair in zip(symbs, dcols)]
        nrows = 6
        HIGH = False
        if (len(aKeys) > 9) or (np.any(np.asarray(list([np.mean(SFH['abundances'][key])] for key in aKeys)) > 0.6)):
            nrows += 2
        if np.any(np.asarray(list([np.mean(SFH['abundances'][key])] for key in aKeys)) > 0.6):
            HIGH = True
        if imft == 1 or imft == 3:
            nrows += 1
        main = plt.figure(figsize=plt.figaspect(nrows/4.)*1.15)
        gs = gridspec.GridSpec(nrows, 1, hspace=0.0, wspace=0.0)
        axi = 0
        ai = 0

        ax = main.add_subplot(gs[axi])
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['age'][rore][idx==k]) for k in
            np.arange(nrad)+1])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                linewidth=0.5, edgecolors='k', s=70, zorder=50,
                label=rf"$t\ [{UTS.gyr}]$")
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['logage'][:, jp][rore][
                    idx==k]) for k in np.arange(nrad)+1])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], 10.0**pamed[amask], alpha=0.1, lw=0.2,
                    c=col, zorder=0)
        else:
            aerr = np.array([np.ma.std(SFH['age'][rore][idx==k]) for k in
                range(nrad)])/2.
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, mew=0.75, mec='k', ecolor=col, ms=12,
                label=rf"$t\ [{UTS.gyr}]$", zorder=50)
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.set_xlim(right=medBins[idx.max()])
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(rf"$t\ [{UTS.gyr}]$")
        kpAx = ax.twiny()
        # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
        kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3)
        # kpAx.set_xlabel(fr"$\log_{{10}}\left(r\ [{UTS.kpace}]\right)$")
        kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        axi += 1

        ax = main.add_subplot(gs[axi])
        ai += 1
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['FeH'][rore][idx==k]) for k in
            np.arange(nrad)+1])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                linewidth=0.5, edgecolors='k', s=70, zorder=50,
                label=r'$[\mathrm{Fe/H}]$')
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['FeH'][:, jp][rore][idx==k])
                    for k in np.arange(nrad)+1])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, mew=0.75, mec='k', ecolor=col, ms=12,
                label=r'$[\mathrm{Fe/H}]$', zorder=50)
            aerr = np.array([np.ma.std(SFH['FeH'][rore][idx==k]) for k in
                np.arange(nrad)+1])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.set_xlim(right=medBins[idx.max()])
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(r'$[\mathrm{Fe/H}]$')
        kpAx = ax.twiny()
        # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
        kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3)
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        kpAx.set_xticklabels([])
        axi += 1

        ax = main.add_subplot(gs[axi:axi+2])
        for aj, key in enumerate(aKeys[:9]):
            ai += 1
            abund = SFH['abundances'][key][rore]
            label = SFH['abundances']['labels'][aj]
            mkr = symbs[ai]
            col = colos[aj]
            amed = np.array([np.ma.median(abund[rore][idx==k])
                for k in np.arange(nrad)+1])
            amed = np.ma.masked_invalid(amed)
            amask = ~np.ma.getmaskarray(amed)
            if posterior:
                ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
                ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
                ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                    label=label, linewidth=0.5, edgecolors='k', s=70,
                    zorder=len(aKeys)-aj+50)
                for jp in range(nPost):
                    pamed = np.array([np.ma.median(maps[key][:, jp][rore][
                        idx==k]) for k in np.arange(nrad)+1])
                    pamed = np.ma.masked_invalid(pamed)
                    ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                        c=col, zorder=0)
            else:
                ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                    marker=mkr, mfc=col, label=label, mew=0.75, mec='k',
                    ecolor=col, ms=12, zorder=len(aKeys)-aj)
                aerr = np.array([np.ma.std(abund[rore][idx==k]) for k in
                    np.arange(nrad)+1])/2.
                ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                    amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.legend(ncol=4, loc='upper center')
        ax.set_xlim(right=medBins[idx.max()])
        ax.set_ylim(top=np.max(ax.get_ylim())+np.ptp(ax.get_ylim())*0.4)
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(r'${\rm Abundance}\ [{\rm dex}]$')
        kpAx = ax.twiny()
        # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
        kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() * 1e-3)
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        kpAx.set_xticklabels([])
        axi += 2

        if (len(aKeys) > 9):
            ax = main.add_subplot(gs[axi:axi+2])
            for aj, key in enumerate(aKeys[9:]):
                ai += 1
                abund = SFH['abundances'][key][rore]
                label = SFH['abundances']['labels'][aj+9]
                mkr = symbs[ai]
                col = colos[aj]
                amed = np.array([np.ma.median(abund[rore][idx==k])
                    for k in np.arange(nrad)+1])
                amed = np.ma.masked_invalid(amed)
                amask = ~np.ma.getmaskarray(amed)
                if posterior:
                    ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
                    ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
                    ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                        label=label, linewidth=0.5, edgecolors='k', s=70,
                        zorder=len(aKeys)-aj+50)
                    for jp in range(nPost):
                        pamed = np.array([np.ma.median(maps[key][:, jp][rore][
                            idx==k]) for k in np.arange(nrad)+1])
                        pamed = np.ma.masked_invalid(pamed)
                        ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                            c=col, zorder=0)
                else:
                    ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                        marker=mkr, mfc=col, label=label, mew=0.75, mec='k',
                        ecolor=col, ms=12, zorder=len(aKeys)-aj)
                    aerr = np.array([np.ma.std(abund[rore][idx==k]) for k in
                        np.arange(nrad)+1])/2.
                    ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                        amed[amask]-aerr[amask], alpha=0.1, color=col)
            ax.legend(ncol=4, loc='upper center')
            ax.set_xlim(right=medBins[idx.max()])
            ax.set_ylim(top=np.max(ax.get_ylim())+np.ptp(ax.get_ylim())*0.4)
            # add 50% to top for legend
            # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
            ax.set_ylabel(r'${\rm Abundance}\ [{\rm dex}]$')
            kpAx = ax.twiny()
            # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() * 1e-3)
            # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
            kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
                top=True)
            kpAx.xaxis.set_label_position('top')
            ax.set_xticklabels([])
            kpAx.set_xticklabels([])
            axi += 2

        ax = main.add_subplot(gs[axi])
        ai += 1
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['IMF']['1'][rore][idx==k]) for k in
            np.arange(nrad)+1])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                label=r'$\alpha_1$', linewidth=0.5, edgecolors='k', s=70,
                zorder=50)
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['IMF1'][:, jp][rore][
                    idx==k]) for k in np.arange(nrad)+1])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, label=r'$\alpha_1$', mew=0.75, mec='k',
                ecolor=col, ms=12, zorder=50)
            aerr = np.array([np.ma.std(SFH['IMF']['1'][rore][idx==k]) for k in
                np.arange(nrad)+1])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        if imft == 1 or imft == 3:
            ai += 1
            mkr = symbs[ai]
            col = colos[ai]
            amed = np.array([np.ma.median(SFH['IMF']['2'][rore][idx==k])
                for k in np.arange(nrad)+1])
            amed = np.ma.masked_invalid(amed)
            amask = ~np.ma.getmaskarray(amed)
            if posterior:
                ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
                ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
                ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                    label=r'$\alpha_2$', linewidth=0.5, edgecolors='k', s=70,
                    zorder=50)
                for jp in range(nPost):
                    pamed = np.array([np.ma.median(maps['IMF2'][:, jp][rore][
                        idx==k]) for k in np.arange(nrad)+1])
                    pamed = np.ma.masked_invalid(pamed)
                    ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                        c=col, zorder=0)
            else:
                ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                    marker=mkr, mfc=col, label=r'$\alpha_2$', mew=0.75, mec='k',
                    ecolor=col, ms=12, zorder=50)
                aerr = np.array([np.ma.std(SFH['IMF']['2'][rore][idx==k])
                    for k in np.arange(nrad)+1])/2.
                ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                    amed[amask]-aerr[amask], alpha=0.1, color=col)
            
            ax.legend(ncols=2)
        ax.set_xlim(right=medBins[idx.max()])
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(r'$\alpha$')
        kpAx = ax.twiny()
        # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
        kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3)
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        kpAx.set_xticklabels([])
        axi += 1

        if imft == 1 or imft == 3:
            ax = main.add_subplot(gs[axi])
            ai += 1
            mkr = symbs[ai]
            col = colos[ai]
            amed = np.array([np.ma.median(xi[rore][idx==k])
                for k in np.arange(nrad)+1])
            amed = np.ma.masked_invalid(amed)
            amask = ~np.ma.getmaskarray(amed)
            if posterior:
                ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
                ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
                ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                    label=r'$\xi$', linewidth=0.5, edgecolors='k', s=70,
                    zorder=50)
                eps = 1e-3
                a1 = maps['IMF1'].astype(np.float64, copy=True)
                m1 = (a1 == 1)
                if m1.any():
                    a1[m1] += (np.random.random(m1.sum()) - 0.5) * eps
                a2 = maps['IMF2'].astype(np.float64, copy=True)
                m2 = (a2 == 1)
                if m2.any():
                    a2[m2] += (np.random.random(m2.sum()) - 0.5) * eps
                for jp in tqdm(range(nPost), desc=u'ξ chains', total=nPost):
                    imfs = [pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.0),
                        slopes=(x1, x2, 2.3)) for x1, x2 in zip(
                            a1[:, jp], a2[:, jp])]
                    xii = np.array(list(map(lambda imf: imf.integrate(
                        mlow=0.2, mhigh=0.5)[0], imfs))) / np.array(list(map(
                        lambda imf: imf.integrate(
                            mlow=0.2, mhigh=1.0)[0], imfs)))
                    pamed = np.array([np.ma.median(xii[rore][idx==k])
                        for k in np.arange(nrad)+1])
                    pamed = np.ma.masked_invalid(pamed)
                    ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                        c=col, zorder=0)
            else:
                ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                    marker=mkr, mfc=col, label=r'$\alpha_2$', mew=0.75, mec='k',
                    ecolor=col, ms=12, zorder=50)
                aerr = np.array([np.ma.std(xi[rore][idx==k])
                    for k in np.arange(nrad)+1])/2.
                ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                    amed[amask]-aerr[amask], alpha=0.1, color=col)
            ax.set_xlim(right=medBins[idx.max()])
            # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
            ax.set_ylabel(r'$\xi$')
            kpAx = ax.twiny()
            # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() *
                1e-3)
            # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
            kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
                top=True)
            kpAx.xaxis.set_label_position('top')
            ax.set_xticklabels([])
            kpAx.set_xticklabels([])
            axi += 1
        
        ax = main.add_subplot(gs[axi])
        ai += 1
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['ML'][band][rore][idx==k]) for k in
            np.arange(nrad)+1])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                linewidth=0.5, edgecolors='k', s=70, zorder=50,
                label=rf"$M/L_{{{band}}}\ [{UTS.msun}/{UTS.lsun}]$")
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['ML'][:, jp][rore][idx==k])
                    for k in np.arange(nrad)+1])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], pamed[amask], alpha=0.1, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, mew=0.75, mec='k', ecolor=col, ms=12,
                label=rf"$M/L_{{{band}}}\ [{UTS.msun}/{UTS.lsun}]$",
                zorder=50)
            aerr = np.array([np.ma.std(SFH['ML'][band][rore][idx==k]) for k in
                np.arange(nrad)+1])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.set_xlim(right=medBins[idx.max()])
        # ax.set_xlabel(r"$\log_{10}\left(R\ [{\rm arcsec}]\right)$")
        ax.set_xlabel(r'$R\ [{\rm arcsec}]$')
        ax.set_ylabel(rf'$M/L_{{{band}}}$')
        kpAx = ax.twiny()
        # kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
        kpAx.set_xlim(np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3)
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        kpAx.set_xticklabels([])

        main.savefig(mDir/f"afh_elements_radial_SN{SN:02d}.png", format='png')
        plt.close('all')

    # plt.clf(); vmin,vmax=POT.sigClip(maps['FeH'][:, 300], 'a'); cnt=dpp(xpix, ypix, maps['FeH'][:, 300][binNum], pixelsize=pixs, angle=PA, vmin=-0.14, vmax=0.18, cmap=icefire); plt.colorbar(cnt); plt.savefig('../alf/map')
    # maps['ML'][j, :] = au.getM2L(f"{galaxy}_SN{SN:02d}_{j:04d}",maps['logage'][j, :], maps['zH'][j, :],maps['IMF1'][j, :], maps['IMF1'][j, :],np.repeat(2.3, nPost), RZ=RZ, band=band, **kwargs)

# ------------------------------------------------------------------------------

def showPlots(galaxy, apers, SN=100, clabels=None, pplots=['spec', 'corn'],
    dcName=''):
    """
    Show plots for the given galaxy and aperture.

    Parameters
    ----------
    galaxy : str
        Name of the galaxy.
    apers : list
        List of apertures to plot.
    SN : int, optional
        Signal-to-noise ratio. Default is 100.
    clabels : list, optional
        List of labels for the colorbar. Default is None.
    pplots : list, optional
        List of plots to generate. Default is ['spec', 'corn'].
    dcName : str, optional
        Name of the data cube. Default is ''.

    Examples
    --------
    >>> am.showPlots('NGC4365', np.arange(2300)[::100], SN=100, clabels=['velz', 'sigma', 'h3', 'h4', 'logage', 'FeH', 'IMF1', 'Na', 'a', 'Ti', 'C', 'N', 'Mg', 'Ca'])
    """

    mDir = curdir/f"{galaxy}{dcName}"
    cfn = mDir/'config.xz'
    CFG = au.Load.lzma(cfn)

    if isinstance(clabels, type(None)):
        clabels = ['velz', 'sigma', 'h3', 'h4', 'logage', 'zH', 'IMF1', 'IMF2',
            'FeH', 'Na', 'a', 'Ti', 'C', 'N', 'Mg', 'Ca']
    if int(CFG['imf_type']) == 0 and 'IMF2' in clabels:
        clabels.pop(clabels.index('IMF2'))

    for aper in np.atleast_1d(apers):
        if 'aperture' in str(aper):
            astr = aper
        else:
            astr = f"{aper:04d}"

        ofn = curdir/'results'/f"{galaxy}_SN{SN:02d}_{astr}.mcmc"
        ifn = curdir/'indata'/f"{galaxy}_SN{SN:02d}_{astr}.dat"
        alf = Alf(ofn.parent/ofn.stem, mPath=ofn.parent)
        alf.get_total_met()
        alf.normalize_spectra()
        # alf.abundance_correct() # convert [X/H] to [X/Fe]
        # alf.get_corrected_abundance_posterior()
        waves, tPix, spec, err, weights, vel = au.readSpec(ifn)

        if 'input' in pplots:
            print('Plotting input spectrum...')
            fig = plt.figure(figsize=plt.figaspect(1./10.))
            ax = fig.gca()
            for wpair in waves:
                ww = np.where((tPix >= wpair[0]*1e4) & (tPix <= wpair[1]*1e4))[0]
                ax.plot(tPix[ww], spec[ww], lw=0.4, c='r')
            ax.fill_between(tPix, weights*spec.max(), alpha=0.2, facecolor='k',
                zorder=0)
            ax.set_ylim(top=(spec*weights).max()*1.1)
            fig.savefig(mDir/f"input_{astr}")
        if 'spec' in pplots:
            print('Plotting spectral fit...')
            alf.plot_model(mDir/f"specFit_{astr}.pdf")

            mwave, model, sinp, merr, _, mres = np.loadtxt(ofn.parent/\
                f"{ofn.stem}.bestspec", unpack=True)
            fig = plt.figure(figsize=plt.figaspect(3.0/10.))
            gs = gridspec.GridSpec(2, 1, hspace=0, wspace=0)
            ax = fig.add_subplot(gs[0, 0])
            for wpair in waves:
                ww = np.where((tPix >= wpair[0]*1e4) & (tPix <= wpair[1]*1e4)
                    )[0]
                ax.plot(tPix[ww], spec[ww], lw=0.4, c='k')
            ax.plot(mwave, model, lw=0.4, c='r')
            ax.fill_between(tPix, (1.0-weights)*spec.max(), alpha=0.2,
                facecolor='k', zorder=0)
            ax.set_ylim(bottom=(spec*weights).min()*1.05,
                top=(spec*weights).max()*1.05)
            ax.set_xlim(min(mwave.min(), tPix.min())-20.,
                max(mwave.max(), tPix.max())+20.)
            ax.set_xticklabels([])
            ax.set_ylabel('Flux')

            ax = fig.add_subplot(gs[1, 0])
            mask = (tPix >= (mwave.min()-1.)) & (tPix <= (mwave.max()+1.))
            temp = tPix[mask][np.ma.getmaskarray(np.ma.masked_less(
                weights[mask], 0.5))]
            mwm = np.array([np.argmin(np.abs(tempi-mwave)) for tempi in temp])
            newm = np.zeros_like(mwave)
            newm[mwm] = 1
            residual = np.ma.masked_array((sinp-model)/sinp*100., mask=newm)
            ax.scatter(mwave, residual, marker='^', c='g', s=2)
            ax.axhline(0.0, lw=0.5, ls='--', c='grey')
            ax.fill_between(tPix, y1=-0.05*(1-weights), y2=0.05*(1-weights),
                alpha=0.2, facecolor='k', zorder=0)
            ax.set_xlabel(rf"Wavelength $[{UTS.angst}]$")
            ax.set_ylabel(r'Residual $[\%]$')
            ax.set_xlim(min(mwave.min(), tPix.min())-20.,
                max(mwave.max(), tPix.max())+20.)
            ax.set_ylim((-5, 5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            fig.savefig(mDir/f"spec_{astr}.pdf")
        if 'corn' in pplots:
            print('Plotting corner...')
            from corner import corner
            lidx = np.array([np.where(np.isin(alf.labels, clab))[0] for clab in
                clabels]).ravel()
            # ensure the order of the labels matches the order of the data
            # columns
            plabels = [au.labelDict.get(label, label) for label in
                alf.labels[lidx].tolist()]
            fig = plt.figure(figsize=plt.figaspect(1.)*1.6)
            fig = corner(alf.mcmc[:, lidx],
                labels=plabels, smooth=0.8, plot_contours=False, labelpad=0.5,
                max_n_ticks=2, plot_datapoints=False, plot_density=True,
                fig=fig, pcolor_kwargs=dict(cmap=rocket))
            ndim = len(lidx)
            axes = np.array(fig.axes).reshape((ndim, ndim))
            for ax in axes[-1, :]:
                ax.xaxis.label.set_rotation(45)
                ax.xaxis.label.set_ha('right')
                ax.xaxis.label.set_va('top')
            fig.savefig(mDir/f"corner_{astr}")
        if 'post' in pplots:
            print('Plotting posteriors...')
            alf.plot_posterior(mDir/f"posterior_{astr}")
        if 'trace' in pplots:
            print('Plotting traces...')
            alf.plot_traces(mDir/f"traces_{astr}.pdf")
        plt.close('all')
    
    return alf

# ------------------------------------------------------------------------------

def kinShow(galaxy, SN, nMom=6, vsys=True, debug=False, full=False,
    pplots=['kin', 'err', 'hist', 'symm'], FOV=True, fit='star'):
    """
    This function delegates to the correct plotting function based on `fit`
    Args
    ----
        galaxy (str): the name of the galaxy
        SN (int): the S/N used to creates the bins on which the kinematics are
            extracted
        nMom (int): the number of Gauss-Hermite moments that were extracted
        vsys (bool): toggles whether to recompute and store the systemic
            velocity
        debug (bool): toggles whether to enter debugging for poorly-fitted
            bins
        full (bool): toggles whether the data was fitted to the `full' spectral
            range
        pplots (list): a list of identifiers to determine which plots to
            produce
        FOV (bool): toggles whether to plot to full spectroscopic FOV
        fit (str):
            'star': fit with MILES stellar templates
            'mstar': fit with MILES stellar templates and multiple LOSVD
                components
    """
    if fit == 'star':
        kpFunc = _kinShow
    elif fit == 'mstar':
        kpFunc = multiKinShow

    kpFunc(galaxy, SN, nMom=nMom, vsys=vsys, debug=debug, full=full,
        pplots=pplots, FOV=FOV)

#------------------------------------------------------------------------------

def _kinShow(galaxy, SN, nMom=6, vsys=True, debug=False, full=False,
    pplots=['kin', 'err', 'hist', 'symm'], FOV=True):
    """
    This function plots the kinematics extracted in pPXF.
    Args
    ----
        galaxy (str): the name of the galaxy
        SN (int): the S/N used to creates the bins on which the kinematics are
            extracted
        nMom (int): the number of Gauss-Hermite moments that were extracted
        vsys (bool): toggles whether to recompute and store the systemic
            velocity
        debug (bool): toggles whether to enter debugging for poorly-fitted
            bins
        full (bool): toggles whether the data was fitted to the `full' spectral
            range
        pplots (list): a list of identifiers to determine which plots to
            produce
        FOV (bool): toggles whether to plot to full spectroscopic FOV
    """

    SN = int(SN)

    if not full:
        tEnd = 'trunc'
    else:
        tEnd = 'full'
    kfs = curdir/galaxy/f"kinematics_SN{SN:02d}.xz"
    pifs = curdir/galaxy/f"pixels_SN{SN:d}.xz"
    bofs = curdir/galaxy/f"binning_SN{SN:02d}_{tEnd}.xz"
    sefs = curdir/galaxy/f"selection_SN{SN:02d}_{tEnd}.xz"
    bfn = kfs.name
    basefn = kfs.stem
    baseName = curdir/galaxy/'mpData'/basefn/('{:07d}_'+f"{basefn}.jl")

    PB = au.Load.lzma(bofs)
    try:
        cubeFlux = PB['binFlux']/PB['binCounts']
    except KeyError:
        binSpec = PB['binSpec']
        nPixels = PB['nPixels']
        cubeFlux = np.ma.sum(binSpec, axis=0)/nPixels
        del binSpec, nPixels

    VO = au.Load.lzma(kfs)
    binNum = VO['binNum']
    if debug:
        mask = (VO['chi2'] < 20)
        pwn = mask[binNum]  # from bins to pixels
        bads = np.unique(binNum[~pwn])
        print(bads)
        xbin = PB['xbin']
        ybin = PB['ybin']
        xbar = PB['xbar']
        ybar = PB['ybar']
        scale = PB['scale']
        endSN = PB['endSN']
        binStat = PB['binStat']
        tLPix = np.arange(PB['lVal'], PB['lVal'] +
                          (PB['lN']*PB['lDel']), PB['lDel'])
        for jk in bads:
            spp = au.Load.jobl(baseName.format(jk))
            spectrum = binSpec[:, jk]
            spectrum /= np.ma.median(spectrum)
            plt.clf()
            plt.plot(tLPix, spectrum)
            plt.savefig(curdir/galaxy/f"dbg_{jk:07d}")
        pdb.set_trace()

    xbin, ybin = VO['x'], VO['y']
    xpix, ypix, sele, pixs = au.Load.lzma(pifs)
    saur, goods = au.Load.lzma(sefs)

    gfs = curdir.parent/'muse'/'obsData'/f"{galaxy}.xz"
    gal = au.Load.lzma(gfs)

    if vsys:
        print('Determining systemic velocity...')
        if 'FCC170' in galaxy:
            vMask = ((xbin < 0) & (xbin > -5) & (ybin < -55)) |\
                (np.sqrt((xbin-37)**2 + (ybin--17)**2) < 10.)
        else:
            vMask = np.zeros_like(xbin, dtype=bool)
        circ = np.sqrt(xbin**2 + ybin**2)
        ww = np.where(circ < np.min([10., circ.max()/2.])) # for FOV smaller
        # than 10''
        mVel = np.ma.masked_invalid(np.ma.masked_array(VO['1'], vMask))
        _vSys = np.ma.median(mVel[ww])
        vMask = np.ma.getmaskarray(mVel)
        mVel = mVel[~vMask]
        plt.clf()
        angBest, angErr, vSys = fkpa(xbin[~vMask], ybin[~vMask], mVel-_vSys,
            quiet=True, plot=True, nsteps=int((360*2)+1))
        plt.savefig(curdir/galaxy/f"fitPA_SN{SN:02d}")
        plt.close('all')
        vSys += _vSys
        if angErr > 10.0:
            angBest = PB['photPA']
            vSys = _vSys
        gal['vSys'] = vSys
        gal['PA'] = 90.-angBest
        au.Write.lzma(gfs, gal)
        print(f"{'': <4s}PA: {angBest: 4.4} +/- {angErr: 4.4}")
        print(
            f"Systemic velocity determined to be {vSys:4.4f} km s^{{-1}}")
    else:
        if 'vSys' in gal.keys():
            vSys = gal['vSys']  # systemic velocity estimate
        else:
            if 'z' in gal.keys():
                vSys = np.log(gal['z']+1)*CTS.c
        print(f"Systemic velocity read in as {vSys:4.4f} km s^{{-1}}")
    VO['1'] -= vSys
    PA = gal['PA']

    gal['sigmaE'] = VO['aperture']['2']
    au.Write.lzma(gfs, gal)

    xbix, ybix = GEO.rotate2D(xpix, ypix, PA)
    pfn = dDir.parent/'muse'/'obsData'/f"{galaxy}{dcName}-poly-rot.xz"
    if pfn.is_file():
        aShape = au.Load.lzma(pfn)
        aShape, pPatch = POT.polyPatch(POLYGON=aShape, Xpo=xbix, Ypo=ybix,
            salpha=0.5, ec=POT.brown, linestyle='--', fill=False, zorder=0,
            lw=0.75)
    else:
        aShape, pPatch = POT.polyPatch(Xpo=xbix, Ypo=ybix, salpha=0.5,
            ec=POT.brown, linestyle='--', fill=False, zorder=0, lw=0.75)
        au.Write.lzma(pfn, aShape)
    xmin, xmax = np.amin(xbix), np.amax(xbix)
    ymin, ymax = np.amin(ybix), np.amax(ybix)
    xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    xbix = np.compress(goods, xbix)
    ybix = np.compress(goods, ybix)
    if not FOV:
        xmin, xmax = np.amin(xbix), np.amax(xbix)
        ymin, ymax = np.amin(ybix), np.amax(ybix)
        xLen, yLen = np.ptp(xbix), np.ptp(ybix) # unmasked pixels

    if xLen < yLen:
        cDim = np.ceil(np.sqrt(nMom)).astype(int)
        rema = nMom % cDim
        rDim = np.floor((nMom-rema)/cDim).astype(int)
    else:
        cDim = np.floor(np.sqrt(nMom)).astype(int)
        rema = nMom % cDim
        rDim = np.ceil((nMom-rema)/cDim).astype(int)

    pren = 2
    # add 20% to width for colourbars and labels
    aspect = (rDim*yLen)/((cDim*xLen)+(cDim))

    assert xbin.size == ybin.size, 'Size inconsistencies.'
    vmin, vmax = POT.sigClip(VO['1'], 'velocity', clipBins=0.05)
    dmin, dmax = POT.sigClip(VO['2'], 'velocity dispersion', clipBins=0.05)
    vmax = np.ceil(np.max([np.abs(vmin), vmax])/5)*5
    vmin = -vmax
    dmin = np.floor(dmin/20)*20
    dmax = np.ceil(dmax/10)*10
    lims = [[vmin, vmax], [dmin, dmax]]
    # labels = [fr"$V\ [{UTS.kms1}]$", fr"$\sigma\ [{UTS.kms1}]$"]
    mome = ['V', r'\sigma']
    units = [fr"[{UTS.kms1}]", fr"[{UTS.kms1}]"]
    for j in range(nMom-2):
        lims += [[-0.2, 0.2]]
        mome += [fr"h{j+3:d}"]
        units += ['']

    if 'kin' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)
        # double the size equally

        print('Plotting moments...')
        # 0.5 mag/arcsec^2 steps
        levels = np.ma.max(cubeFlux) * 10**(-0.4*np.arange(0, 10, 1.)[::-1])
        for mm in tqdm(range(nMom)):
            ax = fig.add_subplot(gs[mm])
            lmi, lma = lims[mm]
            lab = r'\ '.join([ql for ql in [mome[mm], units[mm]] if ql != ''])

            img = dpp(xpix, ypix, (VO[f"{mm+1:d}"][binNum]), pixelsize=pixs,
                      vmin=lmi, vmax=lma, angle=PA)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))

            miText = POT.prec(pren, lmi)
            maText = POT.prec(pren, lma)
            cax = POT.attachAxis(ax, 'right', 0.05)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"${lab}$", va='center',
                ha='center', rotation=270, color=POT.pgreen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='white', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])

        print()
        if rDim > 1 or cDim > 1:
            BIG = fig.add_subplot(gs[:])
            BIG.set_frame_on(False)
            BIG.set_xticks([])
            BIG.set_yticks([])
            BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
            BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)
        else:
            ax.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=7)
            ax.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=7)
        fig.savefig(curdir/galaxy/f"kinematics_{nMom:d}_SN{SN:02d}")
        plt.close('all')

    if 'err' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)

        evmin, evmax = POT.sigClip(VO['1e'], 'v_error', clipBins=0.05)
        edmin, edmax = POT.sigClip(VO['2e'], 'd_error', clipBins=0.05)
        evmin = np.floor(evmin/20)*20
        evmax = np.ceil(evmax/5)*5
        edmin = np.floor(edmin/20)*20
        edmax = np.ceil(edmax/10)*10
        elims = [[evmin, evmax], [edmin, edmax]]
        for _m in range(2, nMom):
            elims += [[0., 0.2]]

        print('Plotting moments...')
        for mm in tqdm(range(nMom)):

            ax = fig.add_subplot(gs[mm])
            emin, emax = elims[mm]
            lab = r'\ '.join([ql for ql in [fr"\delta({mome[mm]})",
                units[mm]] if ql != ''])

            img = dpp(xpix, ypix, (VO[f"{mm+1:d}e"][binNum]), pixelsize=pixs,
                vmin=emin, vmax=emax, angle=PA)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))

            miText = POT.prec(pren, emin)
            maText = POT.prec(pren, emax)

            cax = POT.attachAxis(ax, 'right', 0.05)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"${lab}$", va='center', ha='center',
                rotation=270, color=POT.pgreen, transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='white', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])

        print()
        if rDim > 1 or cDim > 1:
            BIG = fig.add_subplot(gs[:])
            BIG.set_frame_on(False)
            BIG.set_xticks([])
            BIG.set_yticks([])
            BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
            BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)
        else:
            ax.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=7)
            ax.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=7)
        fig.savefig(curdir/galaxy/f"kinematicErrors_{nMom:d}_SN{SN:02d}")
        plt.close('all')

    if 'hist' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.16, wspace=0.)
        fig = plt.figure(figsize=plt.figaspect(rDim/float(cDim))*1.5)

        print('Properties:')

        for mm in range(nMom):
            ax = fig.add_subplot(gs[mm])
            lmi, lma = lims[mm]
            lab = mome[mm]

            emom = np.ma.masked_invalid(VO[f"{mm+1:d}e"])
            mMean = np.ma.mean(emom)
            stdErr = np.ma.std(emom)
            print(f"{'': <4s}{mm+1:d}\n{'': <8s}{'Mean': <10s}: {mMean:4.4}"\
                f"\n{'': <8s}{'StD': <10s}: {stdErr:4.4}")

            ax.hist(emom, histtype='stepfilled', lw=0.8, ec='blue', fc='none')
            ax.axvline(mMean, lw=0.8, c='r', label=r'$\mu$')
            ax.axvspan(mMean-3.*stdErr, mMean+3.*stdErr, alpha=0.4,
                fc='grey', ec='none', label=r'$\pm 3\sigma$')
            ax.axvline(emom.min(), c='k', lw=0.7, ls='--', label='Min/Max')
            ax.axvline(emom.max(), c='k', lw=0.7, ls='--', label='Min/Max')
            if 'floors' in gal.keys():
                ax.hist(emom.clip(min=gal['floors'][mm]), histtype='stepfilled',
                    lw=0.8, ec='green', fc='none', label='Clipped')
            ax.set_xlim(left=0)
            ax.legend(loc=1)
            ax.set_aspect(1./ax.get_data_ratio())
            ax.set_xlabel(fr"${lab}$")
            ax.set_yticks([])
        fig.savefig(curdir/galaxy/f"kinematicErrorHists_{nMom:d}_SN{SN:02d}")

    if 'symm' in pplots:
        if xLen < yLen:
            cDim = np.ceil(np.sqrt(nMom)).astype(int)
            rema = nMom % cDim
            rDim = np.floor((nMom-rema)/cDim).astype(int)
        else:
            cDim = np.floor(np.sqrt(nMom)).astype(int)
            rema = nMom % cDim
            rDim = np.ceil((nMom-rema)/cDim).astype(int)

        pren = 2
        aspect = (rDim*yLen)/((cDim*xLen)+(cDim))
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)

        for mm in tqdm(range(nMom)):
            ax = fig.add_subplot(gs[mm])
            lab = r'\ '.join([ql for ql in [fr"\Delta({mome[mm]})",
                units[mm]] if ql != ''])

            symmed = syvf(xbin, ybin, VO[f"{mm+1:d}"], sym=mm%2+3, pa=PA)
                # point-symmetry: is 3 for (V, h3, h5) and 4 for (sigma, h4, h6)
            delSymm = VO[f"{mm+1:d}"]-symmed
            smax = np.max(np.abs(POT.sigClip(delSymm, 'symm'+mome[mm],
                clipBins=0.02)))
            smin = -smax

            img = dpp(xpix, ypix, delSymm[binNum], pixelsize=pixs,
                angle=PA, vmin=smin, vmax=smax)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))

            miText = POT.prec(pren, smin)
            maText = POT.prec(pren, smax)
            cax = POT.attachAxis(ax, 'right', 0.05)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"${lab}$", va='center',
                ha='center', rotation=270, color=POT.pgreen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.45, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='white', transform=cax.transAxes)
            cax.text(0.45, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])

        print()
        if rDim > 1 or cDim > 1:
            BIG = fig.add_subplot(gs[:])
            BIG.set_frame_on(False)
            BIG.set_xticks([])
            BIG.set_yticks([])
            BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
            BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)
        else:
            ax.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=7)
            ax.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=7)
        fig.savefig(curdir/galaxy/f"symmDiff_{nMom:d}_SN{SN:02d}")
        plt.close('all')

# ------------------------------------------------------------------------------

def plotStackedSpectra(galaxy, apers, SN=100, dcName='',
    spec_sep=1.0, resid_height=0.15, pair_gap=0.25,
    save_name='spec_stacked.pdf'):
    """
    Plot multiple spectral fits and residuals vertically on one axis.

    Each aperture is represented by a spectral fit with its residuals
    immediately below it. Successive spectrum/residual pairs are vertically
    offset to produce a single stacked spectral plot.

    Parameters
    ----------
    galaxy : str
        Name of the galaxy.
    apers : array_like
        Apertures to plot.
    SN : int, optional
        Signal-to-noise ratio. Default is 100.
    dcName : str, optional
        Name appended to the galaxy directory. Default is ''.
    spec_sep : float, optional
        Vertical range assigned to each normalized spectrum. Default is 1.0.
    resid_height : float, optional
        Vertical range corresponding to +/- 5 percent residuals. Default is
        0.15.
    pair_gap : float, optional
        Vertical gap between successive spectrum/residual pairs. Default is
        0.25.
    save_name : str, optional
        Output filename. Default is 'spec_stacked.pdf'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the stacked spectra.
    ax : matplotlib.axes.Axes
        Axes containing the stacked spectra.

    Raises
    ------
    OSError
        If an input spectrum or best-fit spectrum cannot be read.

    Examples
    --------
    >>> fig, ax = plotStackedSpectra(
    ...     'NGC4365',
    ...     np.arange(2300)[::100],
    ...     SN=100,
    ... )
    """
    mDir = curdir / f"{galaxy}{dcName}"

    apers = np.atleast_1d(apers)[::-1]

    pair_height = spec_sep + resid_height + pair_gap

    fig_height = max(6.0, 0.8 * len(apers))
    fig, ax = plt.subplots(
        figsize=(12, fig_height),
        constrained_layout=True,
    )

    xmin = np.inf
    xmax = -np.inf

    for ii, aper in enumerate(apers):
        if 'aperture' in str(aper):
            astr = aper
        else:
            astr = f"{aper:04d}"

        ofn = (
            curdir / 'results' /
            f"{galaxy}_SN{SN:02d}_{astr}.mcmc"
        )
        ifn = (
            curdir / 'indata' /
            f"{galaxy}_SN{SN:02d}_{astr}.dat"
        )

        waves, tPix, spec, err, weights, vel = au.readSpec(ifn)

        mwave, model, sinp, merr, _, mres = np.loadtxt(
            ofn.parent / f"{ofn.stem}.bestspec",
            unpack=True,
        )

        # -------------------------------------------------------------
        # Establish the vertical locations for this pair.
        #
        # The spectrum occupies roughly:
        #
        #     spec_base --> spec_base + spec_sep
        #
        # while the residuals sit immediately below spec_base.
        # -------------------------------------------------------------
        pair_base = ii * pair_height

        resid_base = pair_base
        spec_base = pair_base + resid_height

        # -------------------------------------------------------------
        # Normalize the spectrum locally so all apertures have a
        # comparable visual height.
        # -------------------------------------------------------------
        good = weights > 0.5

        if np.any(good):
            smin = np.nanpercentile(spec[good], 1.0)
            smax = np.nanpercentile(spec[good], 99.0)
        else:
            smin = np.nanmin(spec)
            smax = np.nanmax(spec)

        scale = smax - smin

        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        spec_plot = (
            spec_base +
            spec_sep * (spec - smin) / scale
        )
        model_plot = (
            spec_base +
            spec_sep * (model - smin) / scale
        )

        # -------------------------------------------------------------
        # Input spectrum.
        # -------------------------------------------------------------
        for wpair in waves:
            ww = np.where(
                (tPix >= wpair[0] * 1e4) &
                (tPix <= wpair[1] * 1e4)
            )[0]

            ax.plot(
                tPix[ww],
                spec_plot[ww],
                lw=0.45,
                c='k',
                zorder=2,
            )

        # Best-fitting model.
        ax.plot(
            mwave,
            model_plot,
            lw=0.6,
            c='r',
            zorder=3,
        )

        # -------------------------------------------------------------
        # Construct the residual mask exactly as in showPlots().
        # -------------------------------------------------------------
        mask = (
            (tPix >= (mwave.min() - 1.0)) &
            (tPix <= (mwave.max() + 1.0))
        )

        temp = tPix[mask][
            np.ma.getmaskarray(
                np.ma.masked_less(weights[mask], 0.5)
            )
        ]

        mwm = np.array(
            [
                np.argmin(np.abs(tempi - mwave))
                for tempi in temp
            ],
            dtype=int,
        )

        newm = np.zeros_like(mwave, dtype=bool)
        newm[mwm] = True

        residual = np.ma.masked_array(
            (sinp - model) / sinp * 100.0,
            mask=newm,
        )

        # Map +/- 5 percent onto +/- resid_height / 2.
        residual_plot = (
            resid_base +
            residual * (0.5 * resid_height / 5.0)
        )

        ax.scatter(
            mwave,
            residual_plot,
            marker='.',
            c='g',
            s=3,
            linewidths=0,
            zorder=2,
        )

        # Zero-residual reference line.
        ax.axhline(
            resid_base,
            lw=0.4,
            ls='--',
            c='0.6',
            zorder=1,
        )

        # -------------------------------------------------------------
        # Shade masked spectral regions.
        # -------------------------------------------------------------
        bad = weights < 0.5

        if np.any(bad):
            ax.fill_between(
                tPix,
                resid_base - 0.5 * resid_height,
                spec_base + spec_sep,
                where=bad,
                alpha=0.08,
                facecolor='k',
                linewidth=0,
                zorder=0,
            )

        # Aperture label.
        ax.text(
            mwave.min()+10.0,
            spec_base + 0.75 * spec_sep,
            str(astr),
            ha='left',
            va='center',
            fontsize=8,
        )

        xmin = min(xmin, mwave.min(), tPix.min())
        xmax = max(xmax, mwave.max(), tPix.max())

    # -----------------------------------------------------------------
    # Final formatting.
    # -----------------------------------------------------------------
    ax.set_xlim(xmin - 20.0, xmax + 20.0)

    ymin = -0.6 * resid_height
    ymax = (
        (len(apers) - 1) * pair_height +
        resid_height +
        spec_sep +
        0.1 * spec_sep
    )

    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(rf"Wavelength $[{UTS.angst}]$")

    # Absolute y values no longer have physical meaning after stacking.
    ax.set_ylabel('Normalized flux + offset')
    ax.set_yticks([])

    # Clean presentation.
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', left=False)

    fig.savefig(
        mDir / save_name,
        bbox_inches='tight',
    )

    return fig, ax

# ------------------------------------------------------------------------------
