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
"""
from __future__ import print_function, division

# General modules
import os, re
import traceback, warnings
import json
import sys
import pdb
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

# Dynamics modules
from mgefit.find_galaxy import find_galaxy
from vorbin.voronoi_2d_binning import voronoi_2d_binning as v2db
from plotbin.display_bins import display_bins as dbi
from plotbin.display_pixels import display_pixels as dpp
from plotbin.symmetrize_velfield import symmetrize_velfield as syvf
from pafit.fit_kinematic_pa import fit_kinematic_pa as fkpa
from ppxf.ppxf_util import log_rebin

curdir = plp.Path(__file__).parent
dDir = au._ddir()
icefire = sns.color_palette("icefire", as_cmap=True)
rocket = sns.color_palette("rocket", as_cmap=True)
rocketr = sns.color_palette("rocket_r", as_cmap=True)

CTS = Constants()
UTS = UnitStr()
POT = Plot()
GEO = Geometric()
PHT = Photometry()

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

def vorBinNumber(galaxy, SN, full=False, voro=None, dcName=''):
    SN = int(SN)

    if not full:
        tEnd = 'trunc'
    else:
        tEnd = 'full'
    
    gDir = curdir/f"{galaxy}{dcName}"

    pifs = gDir/f"pixels_SN{SN:d}.xz"
    vofs = gDir/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    sefs = gDir/f"selection_SN{SN:02d}_{tEnd}.xz"

    if isinstance(voro, type(None)):
        VB = au.Load.lzma(vofs)
    else:
        VB = voro
    saur, goods = au.Load.lzma(sefs)
    xpix, ypix, sele, pixs = au.Load.lzma(pifs)
    xbin, ybin = VB['xbin'], VB['ybin']
    xpix = np.compress(goods, xpix)
    ypix = np.compress(goods, ypix)
    binNum = VB['binNum']

    BNImg, _, _, _ = POT.map2img(xpix, ypix, binNum, pixSize=pixs)

    pf.writeto(gDir/f"binnumber_SN{SN:02d}_{tEnd}.fits",
               BNImg.filled(np.nan), overwrite=True)

# ------------------------------------------------------------------------------

def aap(galaxy='NGC5102', kPath=(dDir/'MUSECubes'), vbin=True, targetSN=60,
       minSN=1, full=False, quick=False, kfn=None, dcName='',
       instrument='muse', qProps=dict(timeMax=168, module=[]), smask=[],
       variance=True, priors=True, **kwargs):
    """
    Collates the necessary data to feed into pPXF for NGC 3115
    Args
    ----
        galaxy (str): The name of the galaxy
        kPath (str): the directory containing the reduced data cube
        vbin (bool): toggles whether to Voronoi-bin the spectra
        targetSN (int): the target S/N required by the Voronoi tesselation
            algorithm
        minSN (int): the minimum S/N for a spaxel to be included in the Voronoi
            binning
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
    """
    targetSN = int(targetSN)

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
    pixels, voronoi, selection, srn, kines, cubeCube = [False]*6
    pifs = gDir/f"pixels_SN{targetSN:02d}.xz"
    vofs = gDir/f"voronoi_SN{targetSN:02d}_{tEnd}.xz"
    sefs = gDir/f"selection_SN{targetSN:02d}_{tEnd}.xz"
    snfs = gDir/f"SNR_{tEnd}.xz"
    gfs = curdir.parent/'muse'/'obsData'/f"{galaxy}.xz"
    jfn = dDir/'galaxy-props'/f"{galaxy}.json"

    if dcName != '':
        dcgName = f"*{dcName}*"
    else:
        dcgName = '*'
    try:
        cubeCube = next(kPath.glob(f"*{galaxy}{dcgName}_DATACUBE*.fits"))
    except StopIteration:
        try:
            cubeCube = next(kPath.glob(f"*{galaxy}{dcgName}_DATACUBE*.fz"))
        except StopIteration:
            cubeCube = None
    if pifs.exists():
        pixels = True
    if vofs.exists():
        voronoi = True
    if sefs.exists():
        selection = True
    if snfs.exists():
        srn = True
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
        f"{'': <4s}{'Voronoi': <10s}: {str(voronoi): <5s} ({vofs.name})\n"+\
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
        dataExt = 1 # new F3D format
        dhdr = hdu[dataExt].header
    except IndexError:
        dataExt = 0
        dhdr = hdu[dataExt].header
    try:
        statExt = dataExt+1
        shdr = hdu[statExt].header
    except IndexError:
        try:
            statExt = 'STAT'
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
            fluxii, mask=~sele), galaxy)
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
            points = []
            ellips = [
                [172, 203, 6, 4, -10.],
                [165, 193, 5, 4, -10.],
                [180, 212, 16, 1.3, -20.],
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
            np.ma.masked_array(fluxii, mask=~sele), galaxy)

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
    if vbin and voronoi:
        print('Reading Voronoi-binned data...')
        VB = au.Load.lzma(vofs)
        xp = VB['xbin']
        yp = VB['ybin']
        binNum = VB['binNum']
        nPixels = VB['nPixels']
        # scale = VB['scale']
        endSN = VB['endSN']
        gspecs = VB['binSpec']
        stats = VB['binStat']
        # logLam = VB['logLam']

        nbins = xp.size

        print('Done.', flush=True)
    elif vbin: # Voronoi-binning is desired, but the raw cube needs to be loaded
        loop = True
    if (not vbin) or loop:

        print('Generating spectral data...')
        print(f"{'': <4s}Reshaping `gspecs`...")
        # use full length to get the right shape
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
            snEps, snPA, snRad, snMask = SNRing(
                SNR, minSN, xp, yp, flux, pixs, debug=True, galaxyPath=gDir)
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

        if vbin:
            try:
                print('Running Voronoi tesselation binning...', flush=True)
                gMed = np.ma.median(gspecs[lMask, :], axis=0)
                sMed = np.ma.median(stats[lMask, :], axis=0)  # median(1σ)
                sMed[np.ma.getmaskarray(sMed)] = np.ma.median(sMed)
                # one last check
                binNum, xpin, ypin, xbar, ybar, endSN, nPixels, scale = v2db(
                    xp, yp, gMed, sMed, targetSN, plot=True, quiet=quick,
                    pixelsize=pixs)
                plt.savefig(gDir/f"v2db_SN{targetSN:02d}")
                plt.close('all')
                plt.clf()
                VB = dict()
                VB['binNum'] = binNum
                VB['xbin'] = xpin
                VB['ybin'] = ypin
                VB['xbar'] = xbar
                VB['ybar'] = ybar
                VB['endSN'] = endSN
                VB['nPixels'] = nPixels
                # VB['scale'] = scale
                VB['lVal'] = lmin
                VB['lN'] = llen
                VB['lDel'] = dhdr['CD3_3']
                VB['photPA'] = photPA

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
                VB['binSpec'] = binSpec
                VB['binStat'] = binStat
                VB['binFlux'] = binFlux
                VB['binCounts'] = binSize
                if 'rEMaj' in gal.keys():
                    ReMaj = gal['rEMaj']
                    # Give the effective radius in arcseconds
                elif 'rE' in gal.keys():
                    ReMaj = gal['rE']
                else:
                    warnings.warn("Using default R_e = 10''", RuntimeWarning)
                    ReMaj = 10.0
                apIdx = np.where(np.sqrt(xp**2 + (yp/fcfg.eps)**2) <= ReMaj)[0]
                VB['aperSpec'] = np.squeeze(np.ma.sum(np.atleast_2d(
                    np.take(gspecs, apIdx, axis=1)), axis=1))
                VB['aperStat'] = np.sqrt(np.squeeze(np.ma.sum(np.atleast_2d(
                    np.take(stats, apIdx, axis=1)**2), axis=1)))
                au.Write.lzma(vofs, VB, preset=6)

                vorBinNumber(galaxy, targetSN, full, voro=VB, dcName=dcName)

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
    ax.plot(lPix, VB['aperSpec'], lw=0.4)
    for pair in smask:
        ax.axvspan(pair[0], pair[1], alpha=0.5, facecolor='r', edgecolor=None,
            fill=True)
    fig.savefig(gDir/'apertureSpecMask.pdf')
    au.Write.lzma(gDir/'apertureSpec.figz', fig)

    output['lVal'] = lmin
    output['lN'] = llen
    output['lDel'] = dhdr['CD3_3']
    output['dhdr'] = dhdr
    output['shdr'] = shdr
    if vbin:
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

    au.alfWrite(galaxy, targetSN, nbins, qProps=qProps, priors=priors,
        dcName=dcName)
    plt.close('all')

# ------------------------------------------------------------------------------

def _mpSpecFromSum(aper, galaxy, SN, dcName=''):
    try:
        tail = f"{aper:04d}"
    except ValueError:
        tail = f"{aper}"
    mfn = f"{galaxy}_SN{SN:02d}_{tail}"
    if not (curdir/'results'/f"{mfn}.bestspec2").is_file() and \
        (curdir/'results'/f"{mfn}.bestspec").is_file():
        # Generate model on longer wavelength range
        sp.check_call([f"{str(curdir)}/{galaxy}{dcName}/bin/spec_from_sum.exe",
            mfn])

# ------------------------------------------------------------------------------

def makeSpecFromSum(galaxy='NGC3115', SN=100, full=True, NMP=1, apers=[],
    dcName=''):
    if not full: # Clip the spectral data if required
        tEnd = 'trunc'
    else:
        tEnd = 'full'

    vofs = curdir/f"{galaxy}{dcName}"/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    VO = au.Load.lzma(vofs)
    nSpat = VO['xbin'].size

    if len(apers) < 1:
        apers = np.arange(nSpat)
    else:
        nSpat = len(apers)

    if NMP > 1:
        print(f"{'': <8s}Running {NMP:d} processes")
        with mp.Pool(processes=NMP) as pool:
            it = pool.imap_unordered(partial(_mpSpecFromSum, galaxy=galaxy,
                SN=SN, dcName=dcName), apers)
            for j in tqdm(it, desc='specSum', total=nSpat):
                pass
    else:
        for aper in tqdm(apers, desc='specSum', total=nSpat):
            _mpSpecFromSum(aper, galaxy, SN, dcName)

# ------------------------------------------------------------------------------

def afh(galaxy='NGC3115', SN=100, full=True, FOV=True, vsys=False,
    pplots=['kin', 'err', 'age', 'metal', 'imf', 'ml', 'abund', 'radial'],
    band='F814W', NMP=15, contours=False, dcName='', **kwargs):
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
        am.afh('SNL1', SN=80, full=True, FOV=False, band='F814W',
            photFilt='WFPC2.F814W', dcName='NFMESOouter')
    """    

    posterior = kwargs.pop('posterior', False)

    if not full: # Clip the spectral data if required
        tEnd = 'trunc'
    else:
        tEnd = 'full'

    mDir = curdir/f"{galaxy}{dcName}"

    pifs = mDir/f"pixels_SN{SN:02d}.xz"
    vofs = mDir/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    sefs = mDir/f"selection_SN{SN:02d}_{tEnd}.xz"
    afs  = mDir/f"AFH_SN{SN:02d}_{tEnd}.pkl"
    dkfs  = curdir.parent/'dynamics'/'MUSEKinematics'/f"{galaxy}_SN{SN:02d}.xz"
    kfs  = mDir/f"kins_SN{SN:02d}_{tEnd}.xz"
    sffs = mDir/f"pops_SN{SN:02d}_{tEnd}.xz"
    gfs = curdir.parent/'muse'/'obsData'/f"{galaxy}.xz"
    jfn = dDir/'galaxy-props'/f"{galaxy}.json"
    cfn = mDir/'config.xz'
    xpix, ypix, sele, pixs = au.Load.lzma(pifs)
    VO = au.Load.lzma(vofs)
    saur, goods = au.Load.lzma(sefs)
    CFG = au.Load.lzma(cfn)
    interest = ['fit_type', 'imf_type', 'fit_hermite', 'fit_two_ages']
    print('Run Options:\n'+'\n'.join([f"{'': <4s}{key: >20s}: {CFG[key]}" for
        key in interest]))

    young = bool(int(CFG['fit_two_ages']))
    imft = bool(int(CFG['imf_type']))

    binNum = VO['binNum']
    nSpat = VO['xbin'].size

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
    if (not kfs.is_file()) or (not sffs.is_file()):
        print(f"Looking for {afs}...")
        if not afs.is_file():
            print('Generating...')
            outs = np.sort([xi for xi in plp.Path(curdir/'results').glob(
                f"{galaxy}_SN{SN:02d}_*.mcmc")])[:nSpat] # omit 'aperture'
            ALF = dict()
            for j, out in tqdm(enumerate(outs), desc='Reading ALF',
                    total=nSpat):
                alf = Alf(out.parent/out.stem, mPath=out.parent)
                alf.get_total_met()
                alf.normalize_spectra()
                alf.abundance_correct()
                delattr(alf, 'mcmc')
                ALF[f"{j:04d}"] = alf

            au.Write.pickl(afs, ALF)
        else:
            print(f"Reading {afs}...")
            ALF = au.Load.pickl(afs)

        # mIdx = ALF['0000'].results['Type'].tolist().index('mean')
        mIdx = ALF['0000'].results['Type'].tolist().index('chi2')
        eIdx = ALF['0000'].results['Type'].tolist().index('error')

        KIN = dict()
        KIN['lVal'] = VO['lVal']
        KIN['lN'] = VO['lN']
        KIN['lDel'] = VO['lDel']
        KIN['x'] = VO['xbin']
        KIN['y'] = VO['ybin']
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
        aLabels = [r'$[\alpha{\rm /Fe}]$', r'$[{\rm C/H}]$',
            r'$[{\rm N/H}]$', r'$[{\rm Na/H}]$', r'$[{\rm Mg/H}]$',
            r'$[{\rm Si/H}]$', r'$[{\rm K/H}]$', r'$[{\rm Ca/H}]$',
            r'$[{\rm Ti/H}]$', r'$[{\rm V/H}]$', r'$[{\rm Cr/H}]$',
            r'$[{\rm Mn/H}]$', r'$[{\rm Co/H}]$', r'$[{\rm Ni/H}]$',
            r'$[{\rm Cu/H}]$', r'$[{\rm Sr/H}]$', r'$[{\rm Ba/H}]$',
            r'$[{\rm Eu/H}]$']
        aMask = [ki for ki, key in enumerate(np.take(_popKeys,
            np.arange(2, 20)+1)) if (key in ALF['0000'].labels) and
            (np.ptp([ALF[f"{aper:04d}"].results[key][mIdx] for aper in
                range(nSpat)]) > 1e-3)]
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

        kinKeys = ['velz', 'sigma', 'h3', 'h4', 'velz2', 'sigma2']

        for aper in tqdm(range(nSpat), desc='Apertures', total=nSpat):
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
                    ALF[f"{aper:04d}"].results[ak][mIdx]
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
            MLa = au.getM2L(f"{galaxy}_SN{SN:02d}_{aper:04d}",
                ALF[f"{aper:04d}"].results['logage'][mIdx], SFH['zH'][aper],
                SFH['IMF']['1'][aper], SFH['IMF']['2'][aper], 2.3, RZ=RZ,
                band=band, **kwargs)
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
        fluxii = pf.open(mDir/f"collapsed.fits")[0].data
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
            gal['PA'] = 90.+VO['photPA']
        else: gal['PA'] = 90.-angBest
        PA = gal['PA']
        au.Write.lzma(gfs, gal)
        print(f"{'': <4s}kinPA: {90.-angBest: 4.4} +/- {angErr: 4.4}")
        print(f"{'': <4s}phtPA: {VO['photPA']: 4.4}")
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
    # add 20% to width for colourbars and labels
    aspect = (rDim*yLen)/((cDim*xLen)+(cDim))

    vmin, vmax = POT.sigClip(KIN['1'], 'V', clipBins=0.05)
    dmin, dmax = POT.sigClip(KIN['2'], r'σ', clipBins=0.05)
    vmax = np.ceil(np.max([np.abs(vmin), vmax])/5)*5
    vmin = -vmax
    dmin = np.floor(dmin/20)*20
    dmax = np.ceil(dmax/10)*10
    lims = [[vmin, vmax], [dmin, dmax]]
    # labels = [fr"$V\ [{UTS.kms1}]$", fr"$\sigma\ [{UTS.kms1}]$"]
    mome = ['V', r'\sigma']
    units = [fr"[{UTS.kms1}]", fr"[{UTS.kms1}]"]
    for j in range(nMom-2):
        lims += [[-0.3, 0.3]]
        mome += [fr"h{j+3:d}"]
        units += ['']

    if 'kin' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)
        for mm in tqdm(range(nMom)):
            ax = fig.add_subplot(gs[mm])
            lmi, lma = lims[mm]
            lab = r'\ '.join([ql for ql in [mome[mm], units[mm]] if ql != ''])

            img = dpp(xpix, ypix, (KIN[f"{mm+1:d}"][binNum]), pixelsize=pixs,
                vmin=lmi, vmax=lma, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)

            miText = POT.prec(pren, lmi)
            maText = POT.prec(pren, lma)
            cax = POT.attachAxis(ax, 'right', 0.05)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"${lab}$", va='center',
                ha='center', rotation=270, color=POT.lmagen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
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
        fig.savefig(mDir/f"kinematics_4_SN{SN:02d}")
        plt.close('all')

    if 'err' in pplots:
        gs = gridspec.GridSpec(rDim, cDim, hspace=0.0, wspace=0.0)
        fig = plt.figure(figsize=plt.figaspect(aspect)*1.5)

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
                vmin=emin, vmax=emax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)

            miText = POT.prec(pren, emin)
            maText = POT.prec(pren, emax)

            cax = POT.attachAxis(ax, 'right', 0.05)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"${lab}$", va='center', ha='center',
                rotation=270, color=POT.lmagen, transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
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
        else:
            gs = gridspec.GridSpec(1, 1, hspace=0.0, wspace=0.0)
            mainAge = SFH['age']
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        ax = fig.add_subplot(gs[0])
        img = dpp(xpix, ypix, mainAge[binNum], pixelsize=pixs, vmin=mmin,
            vmax=mmax, angle=PA, cmap=icefire)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)
        if not ax.get_subplotspec().is_last_row():
            ax.set_xticklabels([])
        if not ax.get_subplotspec().is_first_col():
            ax.set_yticklabels([])
        miText = POT.prec(pren, mmin)
        maText = POT.prec(pren, mmax)
        cax = POT.attachAxis(ax, 'right', 0.05, mid=True)
        cb = plt.colorbar(img, cax=cax)
        lT = cax.text(0.5, 0.5, rf"$\langle t\rangle\ [{UTS.gyr}]$",
            va='center', ha='center', rotation=270, color=POT.lmagen,
            transform=cax.transAxes)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        if young:
            ax = fig.add_subplot(gs[1])
            img = dpp(xpix, ypix, SFH['age'][binNum], pixelsize=pixs, vmin=amin,
                vmax=amax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, amin)
            maText = POT.prec(pren, amax)
            cax = POT.attachAxis(ax, 'right', 0.05, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"$t\ [{UTS.gyr}]$", va='center',
                ha='center', rotation=270, color=POT.lmagen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            ax = fig.add_subplot(gs[2])
            img = dpp(xpix, ypix, SFH['yage'][binNum], pixelsize=pixs,
                vmin=jmin, vmax=jmax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, jmin)
            maText = POT.prec(pren, jmax)
            cax = POT.attachAxis(ax, 'right', 0.05, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"$t_y\ [{UTS.gyr}]$", va='center',
                ha='center', rotation=270, color=POT.lmagen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            ax = fig.add_subplot(gs[3])
            img = dpp(xpix, ypix, SFH['fyage'][binNum], pixelsize=pixs,
                vmin=fmin, vmax=fmax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, fmin)
            maText = POT.prec(pren, fmax)
            cax = POT.attachAxis(ax, 'right', 0.05, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, fr"$f_y$", va='center', ha='center',
                rotation=270, color=POT.lmagen, transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

        BIG = fig.add_subplot(gs[:])
        BIG.set_frame_on(False)
        BIG.set_xticks([])
        BIG.set_yticks([])
        BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
        BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)

        fig.savefig(mDir/f"afh_age_SN{SN:02d}")
        plt.close('all')

    if 'metal' in pplots:
        amin, amax = POT.sigClip(SFH['FeH'], 'metal',
            clipBins=0.05)
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))
        ax = fig.gca()
        img = dpp(xpix, ypix, SFH['FeH'][binNum], pixelsize=pixs,
            vmin=amin, vmax=amax, angle=PA, cmap=icefire)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        miText = POT.prec(pren, amin)
        maText = POT.prec(pren, amax)
        cax = POT.attachAxis(ax, 'right', 0.05)
        cb = plt.colorbar(img, cax=cax)
        lT = cax.text(0.5, 0.5, r'$[\mathrm{Fe/H}]$', va='center',
            ha='center', rotation=270, color=POT.lmagen,
            transform=cax.transAxes)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        ax.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=7)
        ax.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=7)

        fig.savefig(mDir/f"afh_metal_SN{SN:02d}")
        plt.close('all')

    if 'imf' in pplots:

        IMF1 = np.ma.masked_equal(SFH['IMF']['1'], 1)
        im1 = np.ma.getmaskarray(IMF1)
        IMF1[im1] = IMF1.data[im1] + ((np.random.ranf()-0.5)*1e-3)
        imin, imax = POT.sigClip(IMF1, 'IMF', clipBins=0.025)

        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))

        if imft == 1 or imft == 3:
            IMF2 = np.ma.masked_equal(SFH['IMF']['2'], 1)
            im2 = np.ma.getmaskarray(IMF2)
            IMF2[im2] = IMF2.data[im2] + ((np.random.ranf()-0.5)*1e-3)
            imfs = [pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.0),
                slopes=(x1, x2, 2.3)) for (x1, x2) in zip(IMF1, IMF2)]
            xiTop = np.array(list(map(lambda imf: imf.integrate(
                mlow=0.2, mhigh=0.5)[0], imfs)))
            xiBot = np.array(list(map(lambda imf: imf.integrate(
                mlow=0.2, mhigh=1.0)[0], imfs)))
            xi = xiTop/xiBot

            i2min, i2max = POT.sigClip(IMF2, 'IMF', clipBins=0.025)
            imin = np.min((imin, i2min))
            imax = np.max((imax, i2max))
            amin, amax = POT.sigClip(xi, 'IMF', clipBins=0.025)
            gs = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.0)

            ax = fig.add_subplot(gs[0])
            lT = ax.text(1e-3, 1-1e-3, r'$\alpha_1$', va='top', ha='left',
            color=POT.lmagen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
        else:
            ax = fig.gca()
            ax.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=7)
            ax.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=7)
        img = dpp(xpix, ypix, IMF1[binNum], pixelsize=pixs,
            vmin=imin, vmax=imax, angle=PA, cmap=icefire)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        ax.set_xticklabels([])
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        miText = POT.prec(pren, imin)
        maText = POT.prec(pren, imax)
        cax = POT.attachAxis(ax, 'right', 0.05, mid=True)
        cb = plt.colorbar(img, cax=cax)
        lT = cax.text(0.5, 0.5, r'$\alpha$', va='center',
            ha='center', rotation=270, color=POT.lmagen,
            transform=cax.transAxes)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        if imft == 1 or imft == 3:
            ax = fig.add_subplot(gs[1])
            img = dpp(xpix, ypix, IMF2[binNum], pixelsize=pixs,
                vmin=imin, vmax=imax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            ax.set_yticklabels([])
            lT = ax.text(5e-2, 1-1e-3, r'$\alpha_2$', va='top', ha='left',
                color=POT.lmagen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            
            ax = fig.add_subplot(gs[2])
            img = dpp(xpix, ypix, xi[binNum], pixelsize=pixs,
                vmin=amin, vmax=amax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            lT = ax.text(1e-3, 1-1e-3, r'$\xi$', va='top', ha='left',
                color=POT.lmagen, transform=ax.transAxes, zorder=200)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.75, foreground='k')])
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)

            miText = POT.prec(pren, amin)
            maText = POT.prec(pren, amax)
            cax = POT.attachAxis(ax, 'right', 0.05, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, r'$\xi$', va='center',
                ha='center', rotation=270, color=POT.lmagen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

            BIG = fig.add_subplot(gs[:])
            BIG.set_frame_on(False)
            BIG.set_xticks([])
            BIG.set_yticks([])
            BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
            BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)

        fig.savefig(mDir/f"afh_IMF_SN{SN:02d}")
        plt.close('all')

    if 'ml' in pplots:

        amin, amax = POT.sigClip(SFH['ML'][band], f'ML_{band}', clipBins=0.05)
        fig = plt.figure(figsize=plt.figaspect(yLen/xLen))
        ax = fig.gca()
        img = dpp(xpix, ypix, SFH['ML'][band][binNum], pixelsize=pixs,
            vmin=amin, vmax=amax, angle=PA, cmap=icefire)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.add_patch(copy(pPatch))
        if contours:
            ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                levels=flevels)

        miText = POT.prec(pren, amin)
        maText = POT.prec(pren, amax)
        cax = POT.attachAxis(ax, 'right', 0.05)
        cb = plt.colorbar(img, cax=cax)
        lT = cax.text(0.5, 0.5, rf"$M/L_{{{band}}}\ [{UTS.msun}/{UTS.lsun}]$",
            va='center', ha='center', rotation=270, color=POT.lmagen,
            transform=cax.transAxes)
        lT.set_path_effects(
            [PathEffects.withStroke(linewidth=1.5, foreground='k')])
        cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
            rotation=270, color='black', transform=cax.transAxes)
        cb.set_ticks([])
        cax.set_zorder(100)

        ax.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=7)
        ax.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=7)

        fig.savefig(mDir/f"afh_ML{band}_SN{SN:02d}")
        plt.close('all')

    if 'abund' in pplots:
        nAbund = len(aKeys)
        dim = np.ceil(np.sqrt(nAbund)).astype(int)
        rema = nAbund % dim
        lo = np.floor((nAbund - rema) / dim).astype(int)
        if lo * dim < nAbund:
            lo += 1

        gs = gridspec.GridSpec(lo, dim, hspace=0.01, wspace=0.01)
        main = plt.figure(figsize=plt.figaspect(lo*yLen / (dim*xLen)) * 2.)

        for ai, key in enumerate(aKeys):
            abund = SFH['abundances'][key]
            label = SFH['abundances']['labels'][ai]

            amin, amax = POT.sigClip(abund, key, clipBins=0.05)
            ax = main.add_subplot(gs[ai])
            img = dpp(xpix, ypix, abund[binNum], pixelsize=pixs,
                vmin=amin, vmax=amax, angle=PA, cmap=icefire)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.add_patch(copy(pPatch))
            if contours:
                ax.tricontour(xbix, ybix, flux, colors='k', linewidths=0.3,
                    levels=flevels)
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticklabels([])
            if not ax.get_subplotspec().is_first_col():
                ax.set_yticklabels([])
            miText = POT.prec(pren, amin)
            maText = POT.prec(pren, amax)
            cax = POT.attachAxis(ax, 'right', 0.1, mid=True)
            cb = plt.colorbar(img, cax=cax)
            lT = cax.text(0.5, 0.5, label, va='center', ha='center',
                rotation=270, color=POT.lmagen, transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
                rotation=270, color='black', transform=cax.transAxes)
            cb.set_ticks([])
            cax.set_zorder(100)

        BIG = main.add_subplot(gs[:])
        BIG.set_frame_on(False)
        BIG.set_xticks([])
        BIG.set_yticks([])
        BIG.set_xlabel(r'$x\ [{\rm arcsec}]$', labelpad=25)
        BIG.set_ylabel(r'$y\ [{\rm arcsec}]$', labelpad=25)

        main.savefig(mDir/f"afh_elements_SN{SN:02d}")
        plt.close('all')

    if 'radial' in pplots:
        if posterior:
            posfn = mDir/f"afh_elements_posteriors_SN{SN:02d}.xz"
            nPost = 100
            if not posfn.is_file():
                outs = np.sort([xi for xi in plp.Path(curdir/'results').glob(
                    f"{galaxy}_SN{SN:02d}_*.mcmc")])[:nSpat] # omit 'aperture'
                pKeys = np.append(aKeys, ['logage', 'FeH', 'IMF1'])
                if imft == 1 or imft == 3:
                    pKeys = np.append(pKeys, 'IMF2')
                maps = dict()
                for ai, key in enumerate(pKeys):
                    maps[key] = np.ma.ones((nSpat, nPost))*np.nan
                maps['ML'] = np.ma.ones((nSpat, nPost))*np.nan
                for j, out in tqdm(enumerate(outs), total=nSpat,
                    desc='Generating posterior samples'):
                    alf = Alf(out.parent/out.stem, mPath=out.parent)
                    alf.get_total_met()
                    alf.normalize_spectra()
                    alf.abundance_correct()
                    for ai, key in enumerate(pKeys):
                        aidx = alfFP.index(key)
                        maps[key][j, :] = np.random.choice(alf.mcmc[:, aidx],
                            nPost, replace=False)
                    maps['ML'][j, :] = au.getM2L(f"{galaxy}_SN{SN:02d}_{j:04d}",
                        np.random.choice(alf.mcmc[:, alfFP.index('logage')],
                            nPost, replace=False),
                        np.random.choice(alf.mcmc[:, alfFP.index('zH')],
                            nPost, replace=False),
                        np.random.choice(alf.mcmc[:, alfFP.index('IMF1')],
                            nPost, replace=False),
                        np.random.choice(alf.mcmc[:, alfFP.index('IMF2')],
                            nPost, replace=False),
                        np.repeat(2.3, nPost), RZ=RZ, band=band, **kwargs)
                au.Write.lzma(posfn, maps)
            else:
                maps = au.Load.lzma(posfn)
        nrad = 12
        eps = 1.-gal['sMGE'].epsE
        rade = np.sqrt(xbin**2 + (ybin/eps)**2)
        rore = np.argsort(rade)
        rade = np.ma.masked_invalid(np.log10(rade[rore]))
        medBins = np.linspace(*POT.sigClip(rade, 'radius', 0.1), nrad+1)
        delta = medBins[1:] - medBins[:-1]
        idx = np.digitize(rade, medBins[1:])
        pBins = medBins[1:] - delta/2
        symbs = ['X', 'p', '^', '<', '>', '8', 's', 'D', 'P', '*', 'h', 'H',
            '+', 'x', 'o', 'v', '1', '2', '3', '4']
        colos = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # dcols = np.tile(plt.rcParams['axes.prop_cycle'].by_key()[
            # 'color'][::-1], 3)[:len(symbs)]
        # colmar = [pair for pair in zip(symbs, dcols)]
        main = plt.figure(figsize=plt.figaspect(1.5)*1.15)
        gs = gridspec.GridSpec(6, 1, hspace=0.0, wspace=0.0)
        ai = 0

        ax = main.add_subplot(gs[0])
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['age'][idx==k]) for k in
            range(nrad)])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                linewidth=0.75, edgecolors='k', s=70, zorder=50,
                label=rf"$\langle t\rangle\ [{UTS.gyr}]$")
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['logage'][:, jp][idx==k])
                    for k in range(nrad)])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], 10.0**pamed[amask], alpha=0.2, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, mew=0.75, mec='k', ecolor=col, ms=12,
                label=rf"$\langle t\rangle\ [{UTS.gyr}]$",
                zorder=50)
            aerr = np.array([np.ma.std(SFH['age'][idx==k]) for k in
                range(nrad)])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.set_xlim(right=rade.max()*1.15)
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(rf"$\langle t\rangle\ [{UTS.gyr}]$")
        kpAx = ax.twiny()
        kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3))
        kpAx.set_xlabel(fr"$\log_{{10}}(r\ [{UTS.kpace}])$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])

        ax = main.add_subplot(gs[1])
        ai += 1
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['FeH'][idx==k]) for k in
            range(nrad)])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                linewidth=0.75, edgecolors='k', s=70, zorder=50,
                label=r'$[\mathrm{Fe/H}]$')
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['FeH'][:, jp][idx==k])
                    for k in range(nrad)])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], pamed[amask], alpha=0.2, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, mew=0.75, mec='k', ecolor=col, ms=12,
                label=r'$[\mathrm{Fe/H}]$',
                zorder=50)
            aerr = np.array([np.ma.std(SFH['FeH'][idx==k]) for k in
                range(nrad)])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.set_xlim(right=rade.max()*1.15)
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(r'$[\mathrm{Fe/H}]$')
        kpAx = ax.twiny()
        kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3))
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        kpAx.set_xticklabels([])

        ax = main.add_subplot(gs[2:4])
        for aj, key in enumerate(aKeys):
            ai += 1
            abund = SFH['abundances'][key][rore]
            label = SFH['abundances']['labels'][aj]
            mkr = symbs[ai]
            col = colos[aj]
            amed = np.array([np.ma.median(abund[idx==k]) for k in range(nrad)])
            amed = np.ma.masked_invalid(amed)
            amask = ~np.ma.getmaskarray(amed)
            if posterior:
                ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
                ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
                ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                    label=label, linewidth=0.75, edgecolors='k', s=70,
                    zorder=len(aKeys)-aj+50)
                for jp in range(nPost):
                    pamed = np.array([np.ma.median(maps[key][:, jp][idx==k]) for
                        k in range(nrad)])
                    pamed = np.ma.masked_invalid(pamed)
                    ax.plot(pBins[amask], pamed[amask], alpha=0.2, lw=0.2,
                        c=col, zorder=0)
            else:
                ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                    marker=mkr, mfc=col, label=label, mew=0.75, mec='k',
                    ecolor=col, ms=12, zorder=len(aKeys)-aj)
                aerr = np.array([np.ma.std(abund[idx==k]) for k in
                    range(nrad)])/2.
                ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                    amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.legend(ncol=4)
        ax.set_xlim(right=rade.max()*1.15)
        ax.set_ylim(top=np.max(ax.get_ylim())*1.5)
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(r'${\rm Abundance}\ [{\rm dex}]$')
        kpAx = ax.twiny()
        kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3))
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        kpAx.set_xticklabels([])

        ax = main.add_subplot(gs[4])
        ai += 1
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['IMF']['1'][idx==k]) for k in
            range(nrad)])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                label=r'$\alpha_1$', linewidth=0.75, edgecolors='k', s=70,
                zorder=50)
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['IMF1'][:, jp][idx==k]) for
                    k in range(nrad)])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], pamed[amask], alpha=0.2, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, label=r'$\alpha_1$', mew=0.75, mec='k',
                ecolor=col, ms=12, zorder=50)
            aerr = np.array([np.ma.std(SFH['IMF']['1'][idx==k]) for k in
                range(nrad)])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        if imft == 1 or imft == 3:
            ai += 1
            mkr, col = colmar[ai]
            amed = np.array([np.ma.median(SFH['IMF']['2'][idx==k]) for k in
                range(nrad)])
            amed = np.ma.masked_invalid(amed)
            amask = ~np.ma.getmaskarray(amed)
            if posterior:
                ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
                ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
                ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                    label=r'$\alpha_2$', linewidth=0.75, edgecolors='k', s=70,
                    zorder=50)
                for jp in range(nPost):
                    pamed = np.array([np.ma.median(maps['IMF2'][:, jp][idx==k])
                        for k in range(nrad)])
                    pamed = np.ma.masked_invalid(pamed)
                    ax.plot(pBins[amask], pamed[amask], alpha=0.2, lw=0.2,
                        c=col, zorder=0)
            else:
                ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                    marker=mkr, mfc=col, label=r'$\alpha_2$', mew=0.75, mec='k',
                    ecolor=col, ms=12, zorder=50)
                aerr = np.array([np.ma.std(SFH['IMF']['2'][idx==k]) for k in
                    range(nrad)])/2.
                ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                    amed[amask]-aerr[amask], alpha=0.1, color=col)
            ax.legend(ncols=2)
        ax.set_xlim(right=rade.max()*1.15)
        # ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(r'$\alpha$')
        kpAx = ax.twiny()
        kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3))
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        ax.set_xticklabels([])
        kpAx.set_xticklabels([])
        
        ax = main.add_subplot(gs[5])
        ai += 1
        mkr = symbs[ai]
        col = 'r'
        amed = np.array([np.ma.median(SFH['ML'][band][idx==k]) for k in
            range(nrad)])
        amed = np.ma.masked_invalid(amed)
        amask = ~np.ma.getmaskarray(amed)
        if posterior:
            ax.plot(pBins[amask], amed[amask], lw=1.0, c=col, zorder=10)
            ax.plot(pBins[amask], amed[amask], lw=2.0, c='k', zorder=2)
            ax.scatter(pBins[amask], amed[amask], marker=mkr, c=col,
                linewidth=0.75, edgecolors='k', s=70, zorder=50,
                label=rf"$M/L_{{{band}}}\ [{UTS.msun}/{UTS.lsun}]$")
            for jp in range(nPost):
                pamed = np.array([np.ma.median(maps['ML'][:, jp][idx==k]) for
                    k in range(nrad)])
                pamed = np.ma.masked_invalid(pamed)
                ax.plot(pBins[amask], pamed[amask], alpha=0.2, lw=0.2,
                    c=col, zorder=0)
        else:
            ax.errorbar(pBins[amask], amed[amask], yerr=aerr[amask],
                marker=mkr, mfc=col, mew=0.75, mec='k', ecolor=col, ms=12,
                label=rf"$M/L_{{{band}}}\ [{UTS.msun}/{UTS.lsun}]$",
                zorder=50)
            aerr = np.array([np.ma.std(SFH['ML'][band][idx==k]) for k in
                range(nrad)])/2.
            ax.fill_between(pBins[amask], amed[amask]+aerr[amask],
                amed[amask]-aerr[amask], alpha=0.1, color=col)
        ax.set_xlim(right=rade.max()*1.15)
        ax.set_xlabel(r'$\log_{10}(R\ [{\rm arcsec}]$)')
        ax.set_ylabel(rf'$M/L_{{{band}}}$')
        kpAx = ax.twiny()
        kpAx.set_xlim(np.log10(10.0**np.array(ax.get_xlim()) * RZ.getPC() *
            1e-3))
        # kpAx.set_xlabel(fr"$r\ [{UTS.kpace}]$")
        kpAx.tick_params(labelbottom=False, labeltop=True, bottom=False,
            top=True)
        kpAx.xaxis.set_label_position('top')
        kpAx.set_xticklabels([])

        main.savefig(mDir/f"afh_elements_radial_SN{SN:02d}.png", format='png')
        plt.close('all')

# ------------------------------------------------------------------------------

def showPlots(galaxy, apers, SN=100, full=True, clabels=None,
    pplots=['input', 'spec', 'corn', 'post', 'trace'], dcName=''):
    frame = incf()
    funcArgs, _, _, funcValues = ingav(frame)
    pplots = funcValues['pplots']

    mDir = curdir/f"{galaxy}{dcName}"
    cfn = mDir/'config.xz'
    CFG = au.Load.lzma(cfn)

    if isinstance(clabels, type(None)):
        clabels = ['velz', 'sigma', 'h3', 'h4', 'logage', 'zH', 'IMF1', 'IMF2',]
    if int(CFG['imf_type']) == 0:
        clabels.pop(clabels.index('IMF2'))

    for aper in np.atleast_1d(apers):
        if 'aperture' in str(aper):
            astr = 'aperture'
        else:
            astr = f"{aper:04d}"

        if not full: # Clip the spectral data if required
            tEnd = 'trunc'
        else:
            tEnd = 'full'
        
        gDir = curdir/f"{galaxy}{dcName}"

        ofn = curdir/'results'/f"{galaxy}_SN{SN:02d}_{astr}.mcmc"
        ifn = curdir/'indata'/f"{galaxy}_SN{SN:02d}_{astr}.dat"
        alf = Alf(ofn.parent/ofn.stem, mPath=ofn.parent)
        alf.get_total_met()
        alf.normalize_spectra()
        alf.abundance_correct()
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
            fig.savefig(gDir/f"input_{astr}")
        if 'spec' in pplots:
            print('Plotting spectral fit...')
            alf.plot_model(gDir/f"specFit_{astr}.pdf")

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
            fig.savefig(gDir/f"spec_{astr}.pdf")
        if 'corn' in pplots:
            print('Plotting corner...')
            alf.plot_corner(gDir/f"corner_{astr}", clabels)
        if 'post' in pplots:
            print('Plotting posteriors...')
            alf.plot_posterior(gDir/f"posterior_{astr}")
        if 'trace' in pplots:
            print('Plotting traces...')
            alf.plot_traces(gDir/f"traces_{astr}.pdf")
        plt.close('all')

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
    vofs = curdir/galaxy/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    sefs = curdir/galaxy/f"selection_SN{SN:02d}_{tEnd}.xz"
    bfn = kfs.name
    basefn = kfs.stem
    baseName = curdir/galaxy/'mpData'/basefn/('{:07d}_'+f"{basefn}.jl")

    frame = incf()
    funcArgs, _, _, funcValues = ingav(frame)
    pplots = funcValues['pplots']

    VB = au.Load.lzma(vofs)
    try:
        cubeFlux = VB['binFlux']/VB['binCounts']
    except KeyError:
        binSpec = VB['binSpec']
        nPixels = VB['nPixels']
        cubeFlux = np.ma.sum(binSpec, axis=0)/nPixels
        del binSpec, nPixels

    VO = au.Load.lzma(kfs)
    binNum = VO['binNum']
    if debug:
        mask = (VO['chi2'] < 20)
        pwn = mask[binNum]  # from bins to pixels
        bads = np.unique(binNum[~pwn])
        print(bads)
        xbin = VB['xbin']
        ybin = VB['ybin']
        xbar = VB['xbar']
        ybar = VB['ybar']
        scale = VB['scale']
        endSN = VB['endSN']
        binStat = VB['binStat']
        tLPix = np.arange(VB['lVal'], VB['lVal'] +
                          (VB['lN']*VB['lDel']), VB['lDel'])
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
            angBest = VB['photPA']
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
                ha='center', rotation=270, color=POT.lmagen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='white', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
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
                rotation=270, color=POT.lmagen, transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='white', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
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
                ha='center', rotation=270, color=POT.lmagen,
                transform=cax.transAxes)
            lT.set_path_effects(
                [PathEffects.withStroke(linewidth=1.5, foreground='k')])
            cax.text(0.5, 1e-3, miText, va='bottom', ha='center',
                rotation=270, color='white', transform=cax.transAxes)
            cax.text(0.5, 1.-1e-3, maText, va='top', ha='center',
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
