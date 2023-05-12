# -*- coding: utf-8 -*-
r"""
    alf_utils.py
    Adriano Poci
    Durham University
    2022

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module contains many functions that are common useful in many
        diffierent applications. They are placed here to remove repeated code
        in other scripts.

    Authors
    -------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:   7 June 2022
v1.1:   Added `smask` keyword to `prepSpec` to mask specific regions of the
            spectra. 23 June 2022
v1.2:   Added `getMass` and `getM2L`. 19 July 2022
v1.3:   Clean old queue files before writing new ones. 7 September 2022
v1.4:   Added aperture fitting and `sed` replacements scripts to `alfWrite`;
        Generate local copies of executables for each galaxy. 28 September 2022
v1.5:   Added `priors` kwarg to `alfWrite`. 13 October 2022
v1.6:   Break the spectra scripts into smaller chunks. 26 October 2022
v1.7:   Fixed bug in `getM2L` by using `RZ.shiftFnu` to compute the model
            magnitude as well. 6 December 2022
v1.8:   Run `make clean` before `make` in `alfWrite`. 8 March 2023
"""
from __future__ import print_function, division

# Core modules
import os, io
import warnings
import traceback
import sys
import pdb
import re
import pathlib as plp
import numpy as np
import shutil as sh
from glob import glob
from copy import copy
from astropy import units as uts
import astropy.io.fits as pf
from astropy.table import Table, unique as atuni
import multiprocessing as mp
from scipy.special import erf as sserf
from scipy.interpolate import interp1d
from functools import partial
from tqdm import tqdm
import subprocess as sp
from inspect import getargvalues as ingav, currentframe as incf
from svo_filters import svo

from alf.scripts import read_alf as ra

# Custom modules
from dynamics.IFU.Galaxy import Mge, Schwarzschild
from dynamics.IFU.FileIO import Load, Write, Read
from dynamics.IFU.Constants import Constants, Units
from cythonModules import C_utils as Cu
from spectres import spectres

from plotbin.sauron_colormap import register_sauron_colormap as srsc

curdir = plp.Path(__file__).parent
# ------------------------------------------------------------------------------

def _ddir():
    """
    This function finds the absolute path to `dynamics` from the system path
    """
    for diir in [x for x in sys.path if plp.Path(x, 'dynamics').is_dir()]:
        dDir = diir
    del diir

    return plp.Path(dDir, 'dynamics')

# ------------------------------------------------------------------------------
dDir = _ddir()

fortMaxInt = 2147483647  # the maximum 32-bit integer in Fortran
CTS = Constants()

# ------------------------------------------------------------------------------

def prepSpec(galaxy, SN, instrument='MUSE', wRange=[4000, 10000], full=True,
    smask=[], dcName=''):

    if not full:  # Clip the spectral data if required
        tEnd = 'trunc'
    else:
        tEnd = 'full'
    
    gDir = curdir/f"{galaxy}{dcName}"

    VO = Load.lzma(gDir/f"voronoi_SN{SN:02d}_{tEnd}.xz")
    tPix = VO['lVal']+np.arange(VO['lN'])*VO['lDel']
    wRange = np.clip(wRange, VO['lVal'], np.max(tPix))
    # wRange = np.insert(wRange, 1,
        # [x for xs in smask for x in xs]).reshape(-1, 2)
    dWave, dLSF = np.loadtxt(dDir/f"{instrument.upper()}.lsf", unpack=True)
    dLSFFunc = interp1d(dWave, dLSF, 'linear', fill_value='extrapolate')
    museLSF = dLSFFunc(tPix)

    binSpec = np.ma.masked_invalid(VO['binSpec'])
    binStat = np.ma.masked_invalid(VO['binStat'])

    velRes = CTS.c/(tPix/museLSF) # lambda/DeltaLambda = c/DeltaVel = R

    relErr = binStat / binSpec
    binSpec /= np.ma.median(binSpec, axis=0)
    binStat = np.ma.abs(binSpec)*relErr

    lDel = np.min(np.diff(tPix))

    weights = np.ones_like(tPix)
    if len(smask) > 0:
        for pair in smask:
            mask = (tPix >= (pair[0]-lDel)) & (tPix <= (pair[1]+lDel))
            weights[mask] = 0.0

    for binn in tqdm(range(binSpec.shape[-1]), desc='Storing Spectra',
        total=binSpec.shape[-1]):
        np.savetxt(curdir/'indata'/f"{galaxy}_SN{SN:02d}_{binn:04d}.dat",
            np.column_stack((tPix, binSpec[:, binn], binStat[:, binn],
            weights, velRes)), fmt='%20.10f',
            header=f"{wRange[0]*1e-4:.5f} {wRange[1]*1e-4:.5f}")
    np.savetxt(curdir/'indata'/f"{galaxy}_SN{SN:02d}_aperture.dat",
        np.column_stack((tPix, VO['aperSpec'], VO['aperStat'],
        weights, velRes)), fmt='%20.10f',
        header=f"{wRange[0]*1e-4:.5f} {wRange[1]*1e-4:.5f}")

# ------------------------------------------------------------------------------

def readSpec(afn):
    tPix, spec, err, weights, vel = np.loadtxt(afn, unpack=True)
    with open(afn, 'r') as sfn:
        header = [line for line in sfn.readlines() if line.startswith('#')]
    waves = np.array([head.lstrip('#').strip().split() for head in header],
        dtype=float)
    return waves, tPix, spec, err, weights, vel

# ------------------------------------------------------------------------------

def alfRead():
    outs = np.sort([xi for xi in plp.Path('results').glob('ngc4365_*.mcmc')])
    nSpat = len(outs)
    apers = np.array([int(out.stem.split('_')[-1]) for out in outs])
    sore = np.argsort(apers)
    outs = outs[sore]
    imfs = []
    for j, out in tqdm(enumerate(outs), total=nSpat):
        alf = ra.Alf(out.parent/out.stem)
        imf = pieceIMF(massCuts=(0.08, 0.5, 1.0, 100.0),
            slopes=(alf.results['IMF1'].data[0], alf.results['IMF2'].data[0],
            2.3))
        imfs += [imf]
    xiTop = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=0.5), imfs)))
    xiBot = np.array(list(map(lambda imf: imf.integrate(
        mlow=0.2, mhigh=1.0), imfs)))

    VO = Load.lzma(dDir.parent/'pxf'/'NGC4365'/'voronoi_SN100_full.xz')
    INF = Load.lzma(dDir.parent/'muse'/'tri_models'/'fin4365'/'infil.xz')
    pdb.set_trace()

# ------------------------------------------------------------------------------

def alfWrite(galaxy, SN, nbins, hours=48, qProps=dict(timeMax=168, module=[]),
    priors=True, dcName=''):

    gDir = curdir/f"{galaxy}{dcName}"

    hours = np.ceil(hours).astype(int)
    hours = np.min((hours, qProps['timeMax']))
    if hours >= 24:
        days = np.floor(hours/24).astype(int)
        thours = hours - int(days*24)
        timeStr = f"{days:d}-{thours:02d}:00:00"
    else:
        timeStr = f"0-{hours:02d}:00:00"
    if 'queue' in qProps.keys():
        if 'cosma' in qProps['queue']:
            timeStr = f"0-{hours:d}:00:00"

    nSteps = 330
    nScripts = np.ceil(nbins/nSteps).astype(int)

    remain = nbins

    for fil in gDir.glob('alf*.qsys'):
        fil.unlink(missing_ok=False)

    for ss in range(nScripts):
        ws = '' # whitespace
        nl = r'\n' # newline
        sStr = ''
        sStr += u'#!/bin/bash -l\n'

        add = nSteps*ss
        top = nSteps
        if remain-nSteps < 0:
            top = copy(remain)
        remain -= nSteps

        if 'owner' in qProps.keys():
            sStr += f"#SBATCH -A {str(qProps['owner'])}\n"
        if 'queue' in qProps.keys():
            sStr += f"#SBATCH -p {str(qProps['queue'])}\n"
        sStr += f'#SBATCH --job-name="alf_{galaxy}_SN{SN:02d}"\n'
        sStr += f'#SBATCH -D "{str(gDir)}"\n'
        sStr += f"#SBATCH --time={timeStr}\n"
        sStr += u'#SBATCH --ntasks=1\n'
        # sStr += u'#SBATCH -N 1\n'
        sStr += u'#SBATCH --cpus-per-task=16\n'
        sStr += u'#SBATCH --mem-per-cpu=3000\n'
        sStr += f'#SBATCH --array=0-{top}\n'
        sStr += u'#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n'
        sStr += u'#SBATCH --mail-user=adriano.poci@durham.ac.uk\n'
        sStr += f'#SBATCH -o "{str(gDir)}/out.log" '\
            u'# Standard out to galaxy\n'
        sStr += f'#SBATCH -e "{str(gDir)}/out.log" '\
            u'# Standard err to galaxy\n'
        sStr += f'#SBATCH --open-mode=append\n\n'

        sStr += u'source ${HOME}/.bashrc\n\n'
        for mod in qProps['module']:
            sStr += f"module load {'/'.join(mod)}\n"
        sStr += f'export ALF_HOME={curdir}{plp.os.sep}\n\n'
        sStr += u'cd ${ALF_HOME}\n'
        if add > 0:
            sStr += u'declare idx=$(printf %04d $((${SLURM_ARRAY_TASK_ID} + '\
                f"{add})))\n"
        else:
            sStr += u'declare idx=$(printf %04d ${SLURM_ARRAY_TASK_ID})\n'
        sStr += u'mpirun --oversubscribe -np ${SLURM_CPUS_PER_TASK} '\
            f'./{galaxy}{dcName}/bin/alf.exe "{galaxy}_SN{SN:02d}_${{idx}}" '\
            f'2>&1 | tee -a "{galaxy}{dcName}/out_${{idx}}.log"\n'

        sf = io.open(gDir/f"alf{ss:02d}.qsys", 'w+', newline='')
        sf.write(sStr)
        sf.flush()
        sf.close()
    
    ws = '' # whitespace
    nl = r'\n' # newline
    sStr = ''
    sStr += u'#!/bin/bash -l\n'

    if 'owner' in qProps.keys():
        sStr += f"#SBATCH -A {str(qProps['owner'])}\n"
    if 'queue' in qProps.keys():
        sStr += f"#SBATCH -p {str(qProps['queue'])}\n"
    sStr += f'#SBATCH --job-name="alf_{galaxy}_SN{SN:02d}_aperture"\n'
    sStr += f'#SBATCH -D "{str(gDir)}"\n'
    sStr += f"#SBATCH --time={timeStr}\n"
    sStr += u'#SBATCH --ntasks=1\n'
    # sStr += u'#SBATCH -N 1\n'
    sStr += u'#SBATCH --cpus-per-task=16\n'
    sStr += u'#SBATCH --mem-per-cpu=3000\n'
    sStr += u'#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n'
    sStr += u'#SBATCH --mail-user=adriano.poci@durham.ac.uk\n'
    sStr += f'#SBATCH -o "{str(gDir)}/out.log" '\
        u'# Standard out to galaxy\n'
    sStr += f'#SBATCH -e "{str(gDir)}/out.log" '\
        u'# Standard err to galaxy\n'
    sStr += f'#SBATCH --open-mode=append\n\n'

    sStr += u'source ${HOME}/.bashrc\n\n'
    for mod in qProps['module']:
        sStr += f"module load {'/'.join(mod)}\n"

    sStr += f'export ALF_HOME={curdir}{plp.os.sep}\n\n'
    sStr += u'### Compile clean version of `alf`\n'
    sStr += u'cd ${ALF_HOME}src\n'
    sStr += u'cp alf.f90.perm alf.f90\n'
    sStr += u'# Remove prior placeholders on velz\n'
    sStr += u'sed -i "/prlo%velz = -999./d" alf.f90\n'
    sStr += u'sed -i "/prhi%velz = 999./d" alf.f90\n'
    sStr += u'make clean && make all && make clean\n'
    sStr += u'cd ${ALF_HOME}\n'
    sStr += u'# Run aperture fit\n'
    sStr += u'mpirun --oversubscribe -np ${SLURM_CPUS_PER_TASK} '\
        f'./bin/alf.exe "{galaxy}_SN{SN:02d}_aperture" 2>&1 | tee -a '\
        f'"{galaxy}{dcName}/out_aperture.log"\n\n'
    sStr += '# Read in the aperture fit\n'
    sStr += u"Ipy='ipython --pylab --pprint --autoindent'\n"
    sStr += f"galax='{galaxy}'\n"
    sStr += f"SN={SN:d}\n"
    sStr += u'pythonOutput=$($Ipy alf_aperRead.py -- -g "$galax" -sn "$SN")\n'
    sStr += f'echo "$pythonOutput" 2>&1 | tee -a '\
        f'"{galaxy}{dcName}/out_aperture.log"\n'
    if priors:
        sStr += u'# Temporary variable for the last line of the Python output\n'
        sStr += u'readarray -t tmp <<< $(echo "$pythonOutput" | tail -n1)\n'
        sStr += u'# Transform into bash array\n'
        sStr += u"IFS=',' read -ra aperKin <<< "'"$tmp"\n'
        sStr += u'echo "${aperKin[*]}" 2>&1 | tee -a '\
            f'"{galaxy}{dcName}/out_aperture.log"\n\n'
        sStr += u'### Compile modified velocity priors\n'
        sStr += u'cd src\n'
        sStr += u'cp alf.f90.perm alf.f90\n'
        sStr += u'# `bc` arithmetic to define the lower and upper velocity bounds\n'
        sStr += u'newVLo=$(bc -l <<< "(${aperKin[0]} - ${aperKin[1]}) - '\
            u'5.0 * (${aperKin[2]} + ${aperKin[3]})")\n'
        sStr += u'newVHi=$(bc -l <<< "(${aperKin[0]} + ${aperKin[1]}) + '\
            u'5.0 * (${aperKin[2]} + ${aperKin[3]})")\n'
        sStr += u'sed -i "s/prlo%velz = -999./prlo%velz = ${newVLo}/g" alf.f90\n'
        sStr += u'sed -i "s/prhi%velz = 999./prhi%velz = ${newVHi}/g" alf.f90\n'
        sStr += u'# Replace the placeholder value in `sed` script\n'
        sStr += u'sed -i "s/velz = 999/velz = ${aperKin[0]}/g" '\
            f"${{ALF_HOME}}{galaxy}{dcName}/alf_replace.sed\n"
        sStr += u'# Run `sed` using the multi-line script\n'
        sStr += u'# Pipe to temporary file\n'
        sStr += f"sed -n -f ${{ALF_HOME}}{galaxy}{dcName}/alf_replace.sed "\
            'alf.f90 >> alf_tmp.f90\n'
        sStr += u'mv alf_tmp.f90 alf.f90\n\n'
        sStr += u'make clean && make all && make clean\n\n'
    sStr += u'# Move executables to local directory\n'
    sStr += u'cd ${ALF_HOME}\n'
    sStr += f"mkdir {galaxy}{dcName}/bin\n"
    sStr += f"cp bin/* {galaxy}{dcName}/bin/\n"
    sStr += f'find "{galaxy}{dcName}" -name "alf*.qsys" -type f -exec sbatch '\
        u'{} \;\n'

    sf = io.open(gDir/'startAlf.qsys', 'w+', newline='')
    sf.write(sStr)
    sf.flush()
    sf.close()

    sStr = ''
    sStr += u"/'cz out of prior bounds, setting to 0.0'/ {\n"
    sStr += f'{ws: <4s}p;n;\n'
    sStr += f'{ws: <4s}/velz = 0.0/ {{\n'
    sStr += f'{ws: <8s}s/velz = 0.0/velz = 999/;\n'
    sStr += f'{ws: <8s}p;d;\n'
    sStr += f'{ws: <12s}}}\n'
    sStr += u'}\n'
    sStr += u'p;\n'
    sf = io.open(gDir/'alf_replace.sed', 'w+', newline='')
    sf.write(sStr)
    sf.flush()
    sf.close()

# ------------------------------------------------------------------------------

def getMass(mto, imf1, imf2, imfTop):
    """Compute mass in stars and remnants (normalized to 1 Msun at t=0).
    Assume an IMF that runs from 0.08 to 100 Msun.

    Parameters
    ----------
    mto : float
        The value of the main-sequence turn-off
    imf1 : float
        The slope of the IMF between low-mass cut-off and 0.5 M_Sun
    imf2 : float
        The slope of the IMF between 0.5 and 1.0 M_Sun
    imfTop : float
        The slope of the IMF between 1.0 M_Sun and the high-mass cut-off. This
            is usually fixed to be Salpeter=2.3

    Returns
    -------
    mass : float
        The normalised integrated mass in stars and stellar remnants

    Raises
    ------
    ExceptionName
        Why the exception is raised.

    Examples
    --------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    # Default parameter settings
    bhlim =  40.0  # Mass limit above which star becomes BH
    nslim =   8.5  # Mass above which star becomes NS
    m2    =   0.5  # Break mass for first IMF segment
    m3    =   1.0  # Break mass for second IMF segment
    mlo   =   0.08 # Low-mass cut-off assumed
    imfhi = 100.0  # Upper mass for integration

    # normalize the weights so that 1 Msun formed at t=0
    # This comes from defining the three-part piecewise linear IMF,
    # N(m)=-X log(m) + c,
    # establishing the constant needed for continuity, and integrating
    # m.N(m)dm within the three sections.
    imfnorm = (m2**(-imf1+2)-mlo**(-imf1+2))/(-imf1+2) +\
        m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) +\
        m2**(-imf1+imf2)*(imfhi**(-imfTop+2)-m3**(-imfTop+2))/(-imfTop+2)

    # stars still alive
    # First the low-mass segment, which is older than the Universe
    getmass = (m2**(-imf1+2)-mlo**(-imf1+2))/(-imf1+2)
    # Now the age-dependent part. mto is the mass of the main-sequence turn off,
    # and is age dependent.

    # if mto < m3, include whole of m2<m<m3
    if mto < m3:
        getmass += m2**(-imf1+imf2)*(mto**(-imf2+2)-m2**(-imf2+2))/(-imf2+2)

    # otherwise, add the two sections up to mto
    else:
        getmass += m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) +\
            m2**(-imf1+imf2)*(mto**(-imfTop+2)-m3**(-imfTop+2))/(-imfTop+2)

    # Normalise
    getmass = getmass/imfnorm

    # BH remnants
    # bhlim<M<imf_up leave behind a 0.5*M BH. bhlim=40, set above
    # According to the age-msto relation, a 40Msun star lives < 100,000yr
    getmass += 0.5*m2**(-imf1+imf2)*(imfhi**(-imfTop+2)-bhlim**(-imfTop+2))/\
        (-imfTop+2)/imfnorm

    # NS remnants
    # nslim<M<bhlim leave behind 1.4 Msun NS
    #  nslim = 8.5 defined above
    # According to the age-msto relation, an 8.5Msun star lives < 10Myr
    getmass += 1.4*m2**(-imf1+imf2)*(bhlim**(-imfTop+1)-nslim**(-imfTop+1))/\
        (-imfTop+1)/imfnorm

    # WD remnants
    # M<8.5 leave behind 0.077*M+0.48 WD
    # There are two parts that must be added: the 0.077* part, which is a
    # fraction of the MASS integral, and the 'fixed' WD mass, which is a mass
    # contribution based on the NUMBER of stars, so uses the NUMBER integral.

    # If mto lt m3, then must consider WD stars in two segments, up to nslim.
    # Otherwise, only the upper segment.
    if mto < m3:
        getmass += 0.48*m2**(-imf1+imf2)*(nslim**(-imfTop+1)-m3**(-imfTop+1))/\
            (-imfTop+1)/imfnorm
        getmass += 0.48*m2**(-imf1+imf2)*(m3**(-imf2+1)-mto**(-imf2+1))/\
            (-imf2+1)/imfnorm
        getmass += 0.077*m2**(-imf1+imf2)*(nslim**(-imfTop+2)-m3**(-imfTop+2))/\
            (-imfTop+2)/imfnorm
        getmass += 0.077*m2**(-imf1+imf2)*(m3**(-imf2+2)-mto**(-imf2+2))/\
            (-imf2+2)/imfnorm
    else:
        getmass += 0.48*m2**(-imf1+imf2)*(nslim**(-imfTop+1)-mto**(-imfTop+1))/\
            (-imfTop+1)/imfnorm
        getmass += 0.077*m2**(-imf1+imf2)*(nslim**(-imfTop+2)-mto**(-imfTop+2)
            )/(-imfTop+2)/imfnorm

    return getmass

# ------------------------------------------------------------------------------

def getM2L(mfn, logage, zh, imf1, imf2, imfTop, RZ=None, band='F814W',
        photFilt='WFPC2.F814W', **kwargs):

    # Variables
    lsun   = 3.839e33 # Solar luminosity in erg/s
    clight = 2.9979e10 # Speed of light (cm/s)
    pc2cm  = 3.08568e18 # cm in a pc

    model = np.loadtxt(curdir/'results'/f"{mfn}.bestspec2")
    mWave = model[:, 0]
    mSpec = model[:, 1]

    # First compute the Main-Sequence Turn Off mass (mto) via relation between
    # mto and (age, metallicity)
    # This was extracted from getm2l.f90, with coefficients from alf_vars.f90
    msto_t0 = 0.33250847
    msto_t1 = -0.29560944
    msto_z0 = 0.95402521
    msto_z1 = 0.21944863
    msto_z2 = 0.070565820
    mto = 10**(msto_t0 + msto_t1 * logage) *\
        (msto_z0 + msto_z1 * zh + msto_z2 * zh**2)

    mass = getMass(mto, imf1, imf2, imfTop)

    filter = svo.Filter(photFilt)
    fWave = filter.wave.to('angstrom').value.flatten()
    fTrans = filter.throughput.flatten()
    # Up-sample filter response
    nfWave = np.linspace(mWave.min(), mWave.max(), 9000)
    ups = interp1d(fWave, fTrans, fill_value='extrapolate')
    nfTrans = ups(nfWave)

    lint = interp1d(mWave, mSpec, fill_value='extrapolate')
    baseTemplate = lint(nfWave)
    # baseTemplate = spectres(fWave, mWave, mSpec)
    # linearly re-bin model spectrum to filter-curve wavelengths,
    # while conserving flux

    physSpec = baseTemplate * lsun/1e6 * nfWave**2/clight/1e8/4./np.pi/\
        pc2cm**2

    tempMag = RZ.shiftFnu(nfWave, physSpec, photFilt=photFilt, **kwargs)

    if tempMag <= 0.0:
        return 0.0

    else:
        # Read in solar spectrum and generate mag sun from filter curve
        swave, snu = map(lambda x: x.value, Read.SolarSpec(dDir/\
            'sun_reference_stis_002.fits'))
        solarMag = RZ.shiftFnu(swave, snu, photFilt=photFilt, **kwargs)

        mass2light = mass / 10.0**(2./5. * (solarMag-tempMag))

        if mass2light > 100.0:
            mass2light = 0.0

        return mass2light

# ------------------------------------------------------------------------------

def _dkAdd(key, val, galaxy=None, mPath=None, parent=None):
    """
    This function adds the key/value set to the specified dictionary
    Args
    ----
        key (str): the key to add to the dictionary
        val (float,int,arr): the parameter to add to `key`. Can be any
            picklable construct
        galaxy (str): the galaxy name. If given, the dictionary will be
            `<object>.xz` in the `obsData` directory. This keyword is checked
            first
        mPath (str): the model directory within `tri_models`. If given, the
            dictionary will be `./tri_models/<mPath>/infil.xz`
        parent (str): the key of which `key` will be a child of
    """

    if not isinstance(galaxy, type(None)):
        pfn = curdir/'obsData'/f"{galaxy}.xz"
    elif not isinstance(mPath, type(None)):
        pfn = curdir/'tri_models'/mPath/'infil.xz'
    else:
        raise IOError('No dictionary found.')

    if not pfn.is_file():
        Write.lzma(pfn, dict())
    dd = Load.lzma(pfn)
    if not isinstance(parent, type(None)):
        if parent not in dd.keys():
            dd[parent] = dict()
        dd[parent][key] = val
    else:
        dd[key] = val

    Write.lzma(pfn, dd)

# ------------------------------------------------------------------------------

def _dkRm(key, galaxy=None, mPath=None, parent=None):
    """
    This function remove the key from the specified dictionary
    Args
    ----
        key (str): the key to remove from the dictionary
        galaxy (str): the galaxy name. If given, the dictionary will be
            `<galaxy>.xz` in the `obsData` directory. This keyword is checked
            first
        mPath (str): the model directory within `tri_models`. If given, the
            dictionary will be `./tri_models/<mPath>/infil.xz`
        parent (str): the key of which `key` will be a child of
    """

    if not isinstance(galaxy, type(None)):
        pfn = curdir/'obsData'/f"{galaxy}.xz"
    elif not isinstance(mPath, type(None)):
        pfn = curdir/'tri_models'/mPath/'infil.xz'
    else:
        raise IOError('No dictionary found.')

    dd = Load.lzma(pfn)
    if not isinstance(parent, type(None)):
        if parent not in dd.keys():
            raise RuntimeError(
                f"`{parent}` not in keys:\n{'': <4s}{dd.keys()}")
        sub = dd[parent]
        sub.pop(key)
        dd[parent] = sub
    else:
        dd.pop(key)

    Write.lzma(pfn, dd)

# ------------------------------------------------------------------------------

def _dkRet(key, galaxy=None, mPath=None):
    """
    This function returns the key from the specified dictionary
    Args
    ----
        key (str): the key to print from the dictionary
        galaxy (str): the galaxy name. If given, the dictionary will be
            `<galaxy>.xz` in the `obsData` directory. This keyword is checked
            first
        mPath (str): the model directory within `tri_models`. If given, the
            dictionary will be `./tri_models/<mPath>/infil.xz`
    Returns
    -------
        dict[key] (dict/arr/list/float): the parameter of the dictionary
            matching `key`
    """

    if not isinstance(galaxy, type(None)):
        pfn = curdir/'obsData'/f"{galaxy}.xz"
    elif not isinstance(mPath, type(None)):
        pfn = curdir/'tri_models'/mPath/'infil.xz'
    else:
        raise IOError('No dictionary found.')

    dd = Load.lzma(pfn)
    keys = np.atleast_1d(key)
    return [dd[key] for key in keys]

# ------------------------------------------------------------------------------

def uniquePairs(xy):
    """
    Returns the unique pairs of `xy`, and the corresponding indices
    Args
    ----
        xy (arr): the (N,2) array of coordinates with repeated entries
    Returns
    -------
        uXY (arr): the (M,2) array of unique pairs
        uInd (arr): the (M,) array of indices of `xy` that form `uXY`
    """
    xyTup = [tuple(z) for z in xy]
    uXY = np.array(
        sorted(set(xyTup), key=lambda x: xyTup.index(x)), dtype=xy.dtype)
    uInd = np.array([xyTup.index(tuple(x)) for x in uXY], dtype=int)

    return uXY.T, uInd

# ------------------------------------------------------------------------------

def _viewCons(q, p, u, qMin):
    """
    Evaluates the mathematical constraints on the intrinsic shape parameters,
        `(q, p, u)` of a triaxial Schwarzschild code
    Args
    ----
        q (arr:float, float): the q values
        p (arr:float, float): the p values
        u (arr:float, float): the p values
        qMin (float): the minimum observed axis ratio
    Returns
    -------
        mask (arr:bool): as mask of where the conditions are met
    """

    q, p, u = map(np.atleast_1d, [q, p, u])

    q2 = q**2
    p2 = p**2
    u2 = u**2
    mask = np.ones(np.max([q.size, p.size, u.size]), dtype=bool)

    mask &= (~np.isnan(q2) & ~np.isnan(p2) & ~np.isnan(q) & ~np.isnan(p))

    mask &= ((q2 >= 0) & (p2 >= 0) & (q >= 0) & (p >= 0))

    TT = (1. - p2) / (1. - q2)
    mask &= ((0 <= TT) & (TT <= 1))

    mask &= (q <= p)

    if q.size == p.size == u.size:
        maxQP = np.nanmax([(q / qMin), p], axis=0)
        minQP = np.nanmin([(p / qMin), np.ones_like(p)], axis=0)
        mask &= ((maxQP <= u) & (u <= minQP))
    elif q.size > p.size and q.size > u.size:
        P = np.full_like(q, p)
        U = np.full_like(q, u)
        maxQP = np.nanmax([(q / qMin), P], axis=0)
        minQP = np.nanmin([(P / qMin), np.ones_like(P)], axis=0)
        mask &= ((maxQP <= U) & (U <= minQP))
    elif p.size > q.size and p.size > u.size:
        Q = np.full_like(p, q)
        U = np.full_like(p, u)
        maxQP = np.nanmax([(Q / qMin), p], axis=0)
        minQP = np.nanmin([(p / qMin), np.ones_like(p)], axis=0)
        mask &= ((maxQP <= U) & (U <= minQP))
    elif u.size > p.size and u.size > q.size:
        Q = np.full_like(u, q)
        P = np.full_like(u, p)
        maxQP = np.nanmax([(Q / qMin), P], axis=0)
        minQP = np.nanmin([(P / qMin), np.ones_like(P)], axis=0)
        mask &= ((maxQP <= u) & (u <= minQP))

    return mask

# ------------------------------------------------------------------------------

def _sec(x):

    s = 1. / np.cos(x)
    return s

# ------------------------------------------------------------------------------

def _cot(x):

    c = 1. / np.tan(x)
    return c

# ------------------------------------------------------------------------------

def testQPU(theta, phi, psi, psiOff, qObs):

    ths = np.linspace(1, 90., 45)
    phs = np.linspace(1, 180, 90)
    pss = np.linspace(1, 180, 90)

    keep = []

    for theta in ths:
        for phi in phs:
            for psi in pss:

                psiRad = (psi+psiOff) * np.pi/180.0
                thetaRad = theta * np.pi/180.0
                phiRad = phi * np.pi/180.0
                secTh = _sec(thetaRad)
                cotPh = _cot(phiRad)

                delq = 1.0-(qObs**2)
                nom1minq2 = delq*(2.0*np.cos(2.0*psiRad) +\
                    np.sin(2.0*psiRad)*(secTh*cotPh - np.cos(thetaRad)*np.tan(phiRad)))
                nomp2minq2 = delq*(2.0*np.cos(2.0*psiRad) + \
                    np.sin(2.0*psiRad)*(np.cos(thetaRad)*cotPh - secTh*np.tan(phiRad)))

                denom = 2.0*np.sin(thetaRad)**2*(delq*np.cos(psiRad)*\
                    (np.cos(psiRad) + secTh*cotPh*np.sin(psiRad)) - 1.0)

                qIntr2 = (1.0 - nom1minq2/denom)
                pIntr2 = (qIntr2 + nomp2minq2/denom)
                uIntr2 = (1./qObs)*np.sqrt(pIntr2*np.cos(thetaRad)**2 +\
                    qIntr2*np.sin(thetaRad)**2*(pIntr2*np.cos(phiRad)**2+\
                    np.sin(phiRad)**2))

                qIntr = np.sqrt(qIntr2)
                pIntr = np.sqrt(pIntr2)
                uIntr = np.sqrt(uIntr2)

                if np.all(_viewCons(qIntr, pIntr, uIntr, qObs.min())):
                    keep += [[theta, phi, psi]]
    pdb.set_trace()

# ------------------------------------------------------------------------------

def deprojIS(sMGE, tMGE, nis=int(100), plot=True, bDir=curdir):
    """
    This function finds the range of possible deprojections for a given set of
        of MGE
    Args
    ----
        sMGE (mge): the Mge object for the luminous tracer
        tMGE (mge): the Mge object for the mass density
        n (int): the number of samples in each spherical axis
        plot (bool): toggles whether to plot the output angles
        bDir (str): the path for the plot to be stored
    Returns
    -------
        angs (arr:float): the permissible spherical angles, of shape (3,M)
        irs (arr:float): the permissible intrinsic shapes, of shape (3,M)
    Output
    ------
        if `plot==True`:
            angles (png): the 3D distribution of permissible angles
    """

    sQ, sPsiOff = sMGE.q, sMGE.offset
    tQ, tPsiOff = tMGE.q, tMGE.offset
    qMin = tQ.min()
    print(f"qMin: {qMin: <5.6f}")

    qList = np.linspace(1e-15, 1.-1e-15, nis)
    pList = np.linspace(1e-15, 1.-1e-15, nis)
    uList = np.linspace(1e-15, 1.-1e-15, nis)
    Qi, Pi, Ui = map(lambda xl: xl.ravel(), np.meshgrid(qList, pList, uList))
    # thList = np.linspace(1e-5, 90., nis)
    # phList = np.linspace(1e-5, 90., nis)
    # psList = np.linspace(1e-5, 180., nis)
    # Ti, Pi, Si = map(lambda xl: xl.ravel(), np.meshgrid(thList, phList, psList))
    warnings.filterwarnings('ignore')

    tTh, tPh, tPs = Cu.QPUtoTPP(Qi, Pi, Ui, qMin)
    goods = np.bitwise_and.reduce(~np.apply_along_axis(np.isnan, 1,
        np.column_stack([tTh, tPh, tPs])), axis=1)
    inrs = np.row_stack([Qi[goods], Pi[goods], Ui[goods]])
    angs = np.row_stack([tTh[goods], tPh[goods], tPs[goods]])
    # mOffset = np.append(sMGE.offset, tMGE.offset)
    # mQ = np.append(sMGE.q, tMGE.q)

    _, _, psc = angs
    warnings.resetwarnings()
    print(f"N_angles: {np.count_nonzero(goods): <5d}")

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.gridspec import GridSpec as ggss
        gs = ggss(3, 3)
        fig = plt.figure(figsize=plt.figaspect(1.))
        for i in range(9):
            ax = fig.add_subplot(gs[i], projection='3d')
            ax.scatter3D(*angs, c='k', zorder=0, s=6)
            ax.scatter3D(*angs, c=psc, cmap='jet', zorder=5, s=2)
            ax.set_xlabel(r'$\vartheta$', fontsize=5., labelpad=-1.)
            ax.set_ylabel(r'$\varphi$', fontsize=5., labelpad=-1.)
            ax.set_zlabel(r'$\psi$', fontsize=5., labelpad=-1.)
            ax.tick_params(labelsize=5., pad=0.)
            ax.view_init(25., 15. + ((360. / 9.) * i))
            ax.relim()
        fig.savefig(bDir/'angles')
        plt.close('all')
    del psc, _
    return angs, inrs.clip(1e-7, 1. - 1e-7)

# ------------------------------------------------------------------------------

def _gridKey(bh=None, q=None, p=None, u=None, dm=None, df=None, ml=None):
    frame = incf()
    fargs, _, _, fvalues = ingav(frame)
    fpms = ['', '', '', '', '', '+', '']

    return keySep.join([f"{x}{fvalues[x]:{fpms[ii]}.7f}" for ii, x in
        enumerate(fargs) if not isinstance(fvalues[x], type(None))])

# ------------------------------------------------------------------------------

def rReplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

# ------------------------------------------------------------------------------

def _deetExtr(fnKey):
    # temp = rReplace(fnKey, plp.os.sep, keySep, 1).split(keySep)
    rstru = ''
    for ji in range(len(freez)-1):
        rstru += r'([a-z]+)([+-]?[0-9]{1,}(?:\.[0-9]+))'
        if ji < len(freez)-2:
            rstru += keySep
        else:
            rstru += r'[\/\\]?(?:([a-z]+)([+-]?[0-9]{1,}(?:\.[0-9]+)))?'
            # optional group with either path sep for M/L

    # Join `temp` instead of using `fnKey` to get rid of the `keySep`
    rmat = re.search(rstru, str(fnKey))
    if isinstance(rmat, type(None)):
        raise RuntimeError(
            f"Key did not return matches for:\n{fnKey}\n{rstru}")

    fpDict = dict()
    gpKeys = list(rmat.groups())[::2]
    gpVals = list(rmat.groups())[1::2]
    for key, val in zip(gpKeys, gpVals):
        if key:
            fpDict[key] = float(val)
    return fpDict

# ------------------------------------------------------------------------------

def _coordExtr(bDir, key=None):

    if isinstance(key, type(None)):
        key = 'bh*'
    gStr = str(plp.Path(key, 'ml*', 'nn_kinem.out'))

    files = list(plp.Path(bDir).rglob(gStr))
    mDirs = np.sort(list(set([plp.Path(x.parent.parent.name, x.parent.name)
        for x in files])))

    mCoords = []
    for my in mDirs:
        # remove the M/L subdirectory, then check if the model directory end in
            # digits
        test = re.search(r'\d+$', str(my.parent))
        if test is None or not str(my).startswith('bh'):
            warnings.warn(f"Key {my} is being skipped.", RuntimeWarning)
            continue
        fpd = _deetExtr(str(my))
        mCoords += [[fpd[aa] for aa in freez]]
    return mCoords, mDirs

# ------------------------------------------------------------------------------

def _clover(vals, steps, limits, FP, step=2., CMR=False, NIt=1, tsMass=0.):
    """
    Given a value for each parameter, this function returns the 2**N unique
        locations on either side of the N-D coordinate
    Args
    ----
        vals (list): a list of current values of the parameters
        steps (list): a list of the step sizes for each parameter.
        limits (list): a list of length-2 lists as [min, max] for each
            parameter
        ops (list): a list of operations to apply to the step, as a choice from
            [`mult`, `add`]. This will define if the step size is
            multiplicative or additive
        FP (list): a list of string keys symbolising the free parameters to run
        step (float): the multiple of the step size to go in both directions
            away from the current value for each parameter
        CMR (bool): toggles whether to use the concentration-mass relation of
            Dutton and Macciò (2014) [doi://10.1093/mnras/stu742]
        NIt (int): the number of potential parameters on either side of the
            centre
        tsMass (float): the stellar mass
    Returns
    -------
        nValues (arr): a (2**N,N) array where each [i,:] pair is the value
            [x1, x2, ..., x_N] that is <step> away from the input coordinate
    """
    vals, steps, limits = map(np.atleast_1d, [vals, steps, limits])
    dim = len(vals)
    delt = len(freez) - len(FP)
    nits = np.full_like(freez, NIt, dtype=int)
    nits[4:-1] = (nits[4:-1]/2).clip(1) # shrink DM parameters
    nits[-1] = np.min([2*nits[-1] + 1, 9])
    nValues = []
    for cc, item in enumerate(freez[:-1]):
        curV = vals[cc]
        if item in FP:
            curS = steps[cc]
            incr = curS * step
            stray = np.arange(-nits[cc], nits[cc]+1, 1)
            nextI = np.sort(np.unique((curV + (incr*stray)).clip(*limits[cc])))
            for tval in nextI:
                valI = np.delete(vals, freez.index('ml'))
                valI[cc] = tval
                nValues += [valI]

    # nValues[1] += (limits[1][-1]-1e-7 - np.max(nValues[1]))
    # nValues[2] = np.array([0.9959998, 0.9979998])
    # nValues[4] = np.array([18.0, 25.0])
    # nValues[5] = np.array([1.0, 2.0])
    # nValues = np.array(np.meshgrid(*nValues),
    #                    dtype=float).T.reshape(-1, dim - 1)

    if CMR:
        SHW = Schwarzschild()
        dmi = freez.index('dm')
        dfi = freez.index('df')
        nValues = np.asarray(list(map(lambda xu: np.concatenate((xu[:dmi],
            np.atleast_1d(SHW.dmConcentration((10.0**xu[dfi])*tsMass).value),
            xu[dfi:])), nValues)))

    cc, item = dim - 1, freez[dim - 1]
    curV = vals[cc]
    if item in FP:
        curS = steps[cc]
        incr = curS * step

        stray = np.arange(-nits[cc], nits[cc]+1, 1)
        nML = np.sort(np.unique(
            (curV + (incr*stray)).clip(*limits[cc])))
    else:
        nML = np.atleast_1d(curV)
    # nML = np.arange(5.0, 11.5, 0.5)

    return nValues, nML

# ------------------------------------------------------------------------------

def _cloverTwist(vals, steps, limits, FP, MGE, step=2., CMR=False, NIt=1,
    tsMass=0.):
    """
    Given a value for each parameter, this function returns the 2**N unique
        locations on either side of the N-D coordinate
    Args
    ----
        vals (list): a list of current values of the parameters
        steps (list): a list of the step sizes for each parameter.
        limits (list): a list of length-2 lists as [min, max] for each
            parameter
        ops (list): a list of operations to apply to the step, as a choice from
            [`mult`, `add`]. This will define if the step size is
            multiplicative or additive
        FP (list): a list of string keys symbolising the free parameters to run
        MGE (Mge): the mass MGE
        step (float): the multiple of the step size to go in both directions
            away from the current value for each parameter
        CMR (bool): toggles whether to use the concentration-mass relation of
            Dutton and Macciò (2014) [doi://10.1093/mnras/stu742]
        NIt (int): the number of potential parameters on either side of the
            centre
        tsMass (float): the stellar mass
    Returns
    -------
        nValues (arr): a (2**N,N) array where each [i,:] pair is the value
            [x1, x2, ..., x_N] that is <step> away from the input coordinate
    """
    vals, steps, limits = map(np.atleast_1d, [vals, steps, limits])
    dim = len(vals)
    delt = len(freez) - len(FP)
    nits = np.full_like(freez, NIt, dtype=int)
    nits[4:-1] = (nits[4:-1]/2).clip(1) # shrink DM parameters
    nits[-1] = np.min([2*nits[-1] + 1, 9])
    nValues = []

    nElem = 100
    theta = np.degrees(np.linspace(0.0, 0.5*np.pi, nElem+1,
        endpoint=False)[1:])
    phi = np.degrees(np.linspace(0.0, 0.5*np.pi, nElem+1,
        endpoint=False)[1:])
    phiPlus = np.degrees(np.linspace(0.0, 0.5*np.pi, nElem+1, endpoint=False
        )[1:]+np.pi)
    psi = np.degrees(np.linspace(0.5*np.pi, np.pi, nElem, endpoint=False))
    psiMinus = np.degrees(np.linspace(0.5*np.pi, np.pi, nElem,
        endpoint=False)-np.pi)
    # theta, phi, phiPlus, psi, psiMinus = map(lambda xray: xray[::int(step)],
    #     [theta, phi, phiPlus, psi, psiMinus])
    # take fewer elements for larger step size
    shap = Cu.TPPtoQPU(theta, phi, psi, MGE.offset, MGE.q)
    shapPM = Cu.TPPtoQPU(theta, phiPlus, psiMinus, MGE.offset, MGE.q)
    mask = np.logical_and.reduce(~np.apply_along_axis(
        np.isnan, 2, shap), axis=2)
    pdb.set_trace()
    for cc, item in enumerate(freez[:-1]):
        curV = vals[cc]
        if item in FP:
            curS = steps[cc]
            incr = curS * step
            stray = np.arange(-nits[cc], nits[cc]+1, 1)
            nextI = np.sort(np.unique(
                (curV + (incr*stray)).clip(*limits[cc])))
        else:
            nextI = np.atleast_1d(curV)
        nValues += [nextI]

    # nValues[1] += (limits[1][-1]-1e-7 - np.max(nValues[1]))
    # nValues[2] = np.array([0.9959998, 0.9979998])
    # nValues[4] = np.array([18.0, 25.0])
    # nValues[5] = np.array([1.0, 2.0])
    nValues = np.array(np.meshgrid(*nValues),
                       dtype=float).T.reshape(-1, dim - 1)

    if CMR:
        SHW = Schwarzschild()
        dmi = freez.index('dm')
        dfi = freez.index('df')
        nValues = np.asarray(list(map(lambda xu: np.concatenate((xu[:dmi],
            np.atleast_1d(SHW.dmConcentration((10.0**xu[dfi])*tsMass).value),
            xu[dfi:])), nValues)))

    cc, item = dim - 1, freez[dim - 1]
    curV = vals[cc]
    if item in FP:
        curS = steps[cc]
        incr = curS * step

        stray = np.arange(-nits[cc], nits[cc]+1, 1)
        nML = np.sort(np.unique(
            (curV + (incr*stray)).clip(*limits[cc])))
    else:
        nML = np.atleast_1d(curV)
    # nML = np.arange(5.0, 11.5, 0.5)

    return nValues, nML

# ------------------------------------------------------------------------------

def cleanErr(mPath):
    bDir = curdir/'tri_models'/mPath
    pDir = bDir/'progress'

    keys = np.array([*map(lambda xd: xd.stem, pDir.glob('*.err'))])
    proceed = input(f"{'--'*38}\nFound {keys.size:d} error keys.\n"+\
        'Delete directories and files ? Y/[N]\n')
    if proceed == 'Y': # strictly equals, and case-sensitive
        (bDir/'emptyd').mkdir(parents=True, exist_ok=True)
        for key in keys:
            cmdd = rf"rsync -a --delete {str(bDir/'emptyd')}/ {str(bDir/key)}/"
            cmdm = fr'find {str(pDir)} -name "{key}*.fin" -type f -delete'
            cmdr = fr'find {str(pDir)} -name "{key}*.run" -type f -delete'
            cmde = fr'find {str(pDir)} -name "{key}*.err" -type f -delete'
            print(f"Deleting {key}*")
            sp.call(cmdd, shell=True)
            sp.call(rf"rm -r {str(bDir/key)}", shell=True)
            sp.call(cmdm, shell=True)
            sp.call(cmdr, shell=True)
            sp.call(cmde, shell=True)
        sp.call(rf"rm -r {str(bDir/'emptyd')}", shell=True)
    else:
        print(keys)
        print('Exiting.')

# ------------------------------------------------------------------------------

def cleanIncomp(mPath, dry=True):
    bDir = curdir/'tri_models'/mPath
    pDir = bDir/'progress'

    keys = np.array([*map(lambda xd: xd.stem, pDir.glob('*.run'))])
    proceed = input(f"{'--'*38}\nFound {keys.size:d} run keys.\n"+\
                    'Delete directories and files ? Y/[N]\n')
    if proceed == 'Y': # strictly equals, and case-sensitive
        for key in keys:
            cmdm = fr'find {str(pDir)} -name "{key}*_ml*" -type f -delete'
            cmdr = fr'find {str(pDir)} -name "{key}*.run" -type f -delete'
            print(f"Deleting {key}*")
            sp.call(cmdm, shell=True)
            sp.call(cmdr, shell=True)
    else:
        print(keys)
        print('Exiting.')

# ------------------------------------------------------------------------------

def cleanIncompFull(mPath, dry=True):
    bDir = curdir/'tri_models'/mPath
    pDir = bDir/'progress'

    keys = np.array([*map(lambda xd: xd.stem, pDir.glob('*.run'))])
    proceed = input(f"{'--'*38}\nFound {keys.size:d} run keys.\n"+\
                    'Delete directories and files ? Y/[N]\n')
    if proceed == 'Y': # strictly equals, and case-sensitive
        for key in keys:
            cmdm = fr'find {str(pDir)} -name "{key}*_ml*" -type f -delete'
            cmdr = fr'find {str(pDir)} -name "{key}*.run" -type f -delete'
            cmdd = fr'find {str(bDir)} -name "{key}*" -type d -exec rm '\
                r'-r "{{}}" \;'
            print(f"Deleting {key}*")
            sp.call(cmdm, shell=True)
            sp.call(cmdr, shell=True)
            sp.call(cmdd, shell=True)
    else:
        print(keys)
        print('Exiting.')

# ------------------------------------------------------------------------------

def cleanCompress(mPath):
    bDir = curdir/'tri_models'/mPath

    key = plp.Path('orblib*.dat')
    cmdf = fr'find {str(bDir/"bh*"/"datfil")} -type f -name "{str(key)}"'+\
        r' -exec xz --threads=2 -v "{}" \;'
    sp.call(cmdf, shell=True)

# ------------------------------------------------------------------------------

def cleanCriteria(mPath, bh=None, q=None, p=None, u=None, dm=None, df=None,
        ml=None, chiR=None, greater=False):
    bDir = curdir/'tri_models'/mPath
    pDir = bDir/'progress'

    tfp = freez.copy()
    # tfp.pop(tfp.index('ml'))
    cfp = np.append(tfp, 'chiR')

    funcArgs, _, _, funcValues = ingav(incf())
    crits = list(map(lambda ki: funcValues[ki], funcArgs[1:-1]))

    if not isinstance(chiR, type(None)) or not isinstance(ml, type(None)):
        Cfn = bDir/'chi2.csv'
        oup = Table.read(Cfn, format='csv',
            names=['KinChi', 'chiR', *freez], data_start=1)
        soup = np.lib.recfunctions.structured_to_unstructured(
            oup[tfp+['chiR']].as_array())
    else:
        soup = np.array([list(_deetExtr(ley).values()) for ley in
            bDir.glob('bh*')])

    mask = np.zeros(soup.shape[0], dtype=bool)
    for ci, cc in enumerate(crits):
        if not isinstance(cc, type(None)):
            if greater:
                mask |= soup[:, ci] > cc
            else:
                mask |= soup[:, ci] < cc
    crData = soup[mask, :]
    proceed = input(f"{'--'*38}\nFound {len(crData):d} run keys.\n"+\
                    'Delete directories and files ? Y/[N]\n')
    if proceed == 'Y': # strictly equals, and case-sensitive
        (bDir/'emptyd').mkdir(parents=True, exist_ok=True)
        for row in tqdm(crData, total=crData.shape[0]):
            oke = _gridKey(*row[:7])
            ttdir = plp.Path(bDir, rReplace(oke, keySep, plp.os.sep, 1))
            bMod = ttdir.parent
            cmdd = rf"rsync -a --delete {str(bDir/'emptyd')}/ {str(ttdir)}/"
            sp.call(cmdd, shell=True)
            sp.call(rf"rm -r {str(ttdir)}", shell=True)
            leftovers = [di for di in bMod.glob('ml*')]
            if len(leftovers) < 1:
                cmdd = rf"rsync -a --delete {str(bDir/'emptyd')}/ {str(bMod)}/"
                sp.call(cmdd, shell=True)
                sp.call(rf"rm -r {str(bMod)}", shell=True)
        # sp.call(rf"rm -r {str(bDir/'emptyd')}", shell=True)
    else:
        print(crData)

# ------------------------------------------------------------------------------

def cleanTriaxmass(mPath):
    bDir = curdir/'tri_models'/mPath
    pDir = bDir/'progress'

    rstru = ''
    for ji in range(len(freez)-1):
        rstru += r'([a-z]+)([+-]?[0-9]{1,}(?:\.[0-9]+))'
        if ji < len(freez)-2:
            rstru += keySep
        else:
            rstru += r'[\/\\]?(?:([a-z]+)([+-]?[0-9]{1,}(?:\.[0-9]+)))?'
            # optional group with either path sep for M/L


    tamFiles = bDir.glob(str(plp.Path('*', 'triaxmass.log')))
    for tam in tamFiles:
        rmat = re.search(rstru, str(tam))
        mKey = _gridKey(*np.array([rg for rg in rmat.groups() if rg][1::2],
            dtype=float))
        tamp = open(tam, 'r')
        triax = tamp.read()
        tamp.close()
        # smat = re.search(r'\bSTOP\b', triax)
        if triax == '':
            cmdm = fr'find {str(pDir)} -name "*{mKey}*" -type f -delete'
            cmdd = fr'find {str(bDir)} -name "*{mKey}*" -type d -exec rm -r "{{}}" \;'
            # cmdm = fr'find {str(pDir)} -name "*{mKey}*" -type f'
            # cmdd = fr'find {str(bDir)} -name "*{mKey}*" -type d'
            sp.call(cmdm, shell=True)
            sp.call(cmdd, shell=True)
            # tmp = open(pDir/f"{mKey}.err", 'w+')
            # tmp.close()
    pdb.set_trace()

# ------------------------------------------------------------------------------

def deg2HMS(ra='', dec='', round=False):
    RA, DEC, rs, ds = '', '', '', ''
    if dec:
        if str(dec)[0] == '-':
            ds, dec = '-', abs(dec)
        deg = int(dec)
        decM = abs(int((dec - deg) * 60))
        if round:
            decS = int((abs((dec - deg) * 60) - decM) * 60)
        else:
            decS = (abs((dec - deg) * 60) - decM) * 60
        DEC = '{0}{1} {2} {3:2.4f}'.format(ds, deg, decM, decS)

    if ra:
        if str(ra)[0] == '-':
            rs, ra = '-', abs(ra)
        raH = int(ra / 15)
        raM = int(((ra / 15) - raH) * 60)
        if round:
            raS = int(((((ra / 15) - raH) * 60) - raM) * 60)
        else:
            raS = ((((ra / 15) - raH) * 60) - raM) * 60
        RA = '{0}{1} {2} {3:2.4f}'.format(rs, raH, raM, raS)

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

# ------------------------------------------------------------------------------

def HMS2deg(ra=None, dec=None):
    """
    Args
    ----
        ra, dec (float): The values in the format '<hrs>:<mins>:<secs>'
    Returns
    -------
        ra, dec (float): The decimal values of the above coordinates
    """

    RA, DEC, rs, ds = '', '', 1, 1
    if dec:
        D, M, S = [float(i) for i in dec.split(':')]
    if D < 0:
        ds, D = -1, abs(D)
    deg = D + (M / 60) + (S / 3600)
    DEC = deg * ds

    if ra:
        H, M, S = [float(i) for i in ra.split(':')]
    if H < 0:
        rs, H = -1, abs(H)
    deg = (H * 15) + (M / 4) + (S / 240)
    RA = deg * rs

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

# ------------------------------------------------------------------------------

def _nnProc(object, mPath, lKey, DOF=False, method='comp'):
    #, nMom=None, nGH=None, errSB=None, nBins=None, nDOF=None):
    """
    This function processes the NNLS outputs for a particular trial from a
        triaxial Schwarzschild model and computes its goodness-of-fit
    Args
    ----
        object (str): the name of the object
        mPath (str): the name of the directory where the model resides
        lKey (str): a specifically-formated string key containing the
            information about the model trial
        DOF (bool): toggles whether to compute the reduced χ^2/DOF
        method (str): choice of `['comp', 'read']` to compute or read in the
            χ^2 of the dynamical models
        nMom (int): the number of kinematic moments
        nGH (int): the number of Guass-Hermite coefficients
        errSB (float): the tolerance on the surface brightness fitting
        nBins (int): the number of spatial apertures
        nDOF (int): the degrees of freedom
    """

    bDir = curdir/'tri_models'/mPath

    if method == 'comp':
        try:
            '''
                    index   |   moment
                    --------|-----------
                        0   |   mass
                        1   | mass model
                        2   | GH01 model
                        3   |    GH01
                        4   | GH01 error
                       ...  |     ...
            '''

            nlen, N, KIN, nkMom = Read.kinematics(bDir/lKey/'nn_kinem.out')
            mIdx = np.array([2 + (k * 3)  for k in range(nkMom)])
            # the model indices
            kIdx = np.array([2 + (k * 3) + 1 for k in range(nkMom)
                if np.any(np.nonzero(KIN[k, :]))])
            if np.any([np.all(np.isnan(KIN[mi, :])) for mi in mIdx]):
                return np.nan
            kinChi2 = np.sum(np.array([((KIN[i-1, :] - KIN[i, :])/
                KIN[i+1, :])**2 for i in kIdx[1:]]))


        except:
            print(lKey)
            print(nData)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exc()
            print(f"LINE {exc_traceback.tb_lineno}\n{'': <4s}{exc_type}\n"\
                  f"{'': <4s}{exc_value}")
            pdb.set_trace()
    else:
        with open(bDir/lKey/'nn_nnls.out', 'r+') as ff:
            nnls = [x.strip().split() for x in ff.readlines()]
        nData = np.array(nnls).T
        nTypes = [str, int, float, float, float]
        dat, nConVec, chi2, sqChi2, redChi2 = [x.astype(fn) for x, fn in
                                               zip(nData, nTypes)]
        kinChi2 = redChi2[-2]

    return kinChi2#, chi, allChi2

# ------------------------------------------------------------------------------

def _NC2S(ix, iy, iz, eps):
    """
    This function produces the 3D transformation matrix from Cartesian to
        Spherical coordinates
    Args
    ----
        ix (arr:float): the x-coordinate in Cartesian
        iy (arr:float): the y-coordinate in Cartesian
        iz (arr:float): the z-coordinate in Cartesian
    Returns
    -------
        out (arr:float): the transformation matrix to output `(r, θ, φ)`
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_Cartesian_coordinates_2
    """

    cylR = np.sqrt(ix**2 + iy**2)
    RR = np.sqrt(ix**2 + iy**2 + iz**2)

    out = np.ma.zeros((3, 3))

    if (RR > eps) and (cylR > eps):
        out[0, 0] = ix / RR
        out[0, 1] = iy / RR
        out[0, 2] = iz / RR

        out[1, 0] = (ix * iz) / (cylR * RR)
        out[1, 1] = (iy * iz) / (cylR * RR)
        out[1, 2] = -cylR / (RR)

        out[2, 0] = -iy / (cylR)
        out[2, 1] = ix / (cylR)
        out[2, 2] = 0.0

    return out

# ------------------------------------------------------------------------------

def CAR2SPHmu12(x, y, z, mu1Car, mu2Car, eps=None):
    assert x.size == y.size == z.size,\
        'Input cartesian vectors must be equal shape.'

    nL = x.size
    if isinstance(eps, type(None)):
        eps = 1e-15

    mu1Sph = np.ma.ones(mu1Car.shape)*np.nan
    mu2Sph = np.ma.ones(mu2Car.shape)*np.nan
    for ij in range(nL):
        N = _NC2S(x[ij], y[ij], z[ij], eps)
        mu1Sph[ij, :] = np.ma.dot(N, mu1Car[ij, :])

        for jj in range(3):
            for kj in range(3):
                mu2Sph[ij, jj, kj] = np.ma.sum(
                    np.ma.outer(N[jj, :], N[kj, :]) * mu2Car[ij, :, :])
    mu1Sph = np.ma.masked_invalid(mu1Sph)
    mu2Sph = np.ma.masked_invalid(mu2Sph)
    return mu1Sph, mu2Sph

# ------------------------------------------------------------------------------

def _NC2C(ix, iy, iz, eps):
    """
    This function produces the 3D transformation matrix from Cartesian to
        Cylindrical coordinates
    Args
    ----
        ix (arr:float): the x-coordinate in Cartesian
        iy (arr:float): the y-coordinate in Cartesian
        iz (arr:float): the z-coordinate in Cartesian
    Returns
    -------
        out (arr:float): the transformation matrix to output `(r, θ, z)`
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_Cartesian_coordinates_2
    """

    cylR = np.sqrt(ix**2 + iy**2)

    out = np.ma.zeros((3, 3))

    if isinstance(eps, type(None)):
        eps = 1e-10

    if (cylR > eps) and (cylR**2 > eps):
        out[0, 0] = ix / cylR
        out[0, 1] = iy / cylR
        out[0, 2] = 0.0

        out[1, 0] = -iy / (cylR)
        out[1, 1] = ix / (cylR)
        out[1, 2] = 0.0

        out[2, 0] = 0.0
        out[2, 1] = 0.0
        out[2, 2] = 1.0

    return out

# ------------------------------------------------------------------------------

def CAR2CYLmu12(x, y, z, mu1Car, mu2Car, eps=None):
    assert x.size == y.size == z.size,\
        'Input cartesian vectors must be equal shape.'

    nL = x.size
    if isinstance(eps, type(None)):
        eps = 1e-10

    mu1Cyl = np.ma.ones(mu1Car.shape)*np.nan
    mu2Cyl = np.ma.ones(mu2Car.shape)*np.nan
    for ij in range(nL):
        N = _NC2C(x[ij], y[ij], z[ij], eps)
        mu1Cyl[ij, :] = np.ma.dot(N, mu1Car[ij, :])

        for jj in range(3):
            for kj in range(3):
                mu2Cyl[ij, jj, kj] = np.ma.sum(
                    np.ma.outer(N[jj, :], N[kj, :]) * mu2Car[ij, :, :])
                pdb.set_trace()
    mu1Cyl = np.ma.masked_invalid(mu1Cyl)
    mu2Cyl = np.ma.masked_invalid(mu2Cyl)
    return mu1Cyl, mu2Cyl

# ------------------------------------------------------------------------------

def _NS2C(ir, it, ip):
    """
    This function produces the 3D transformation matrix from Spherical to
        Cylindrical coordinates
    Args
    ----
        ir (arr:float): the r-coordinate in Spherical
        it (arr:float): the θ-coordinate in Spherical
        ip (arr:float): the φ-coordinate in Spherical
    Returns
    -------
        out (arr:float): the transformation matrix to output `(r, θ, z)`
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_Cartesian_coordinates_2
    """

    out = np.ma.zeros((3, 3))

    out[0, 0] = np.sin(ip)
    out[0, 1] = ir * np.cos(ip)
    out[0, 2] = 0.0

    out[1, 0] = 0.0
    out[1, 1] = 0.0
    out[1, 2] = 1.0

    out[2, 0] = np.cos(ip)
    out[2, 1] = -1. * ir * np.sin(ip)
    out[2, 2] = 0.0

    return out

# ------------------------------------------------------------------------------

def covv(X):
    cov = np.array([[(i * j).mean() - (i.mean() * j.mean()) for j in X]
        for i in X])

    return cov

# ------------------------------------------------------------------------------

def _aperView(object, mPath, x, y):
    """
    This function converts an (x,y) position into the corresponding bin numbers
        that are near that position
    """
    import matplotlib.pyplot as plt
    from cap_display_pixels import display_pixels as dispp

    bDir = opj(curdir, 'tri_models', mPath)
    try:
        INF = Load.lzma(opj(bDir, 'infil.xz'))
        gal = Load.lzma(opj(curdir, 'obsData', "{}.xz".format(object)))
        PIXS = gal['pix']['scale']
        vmin, vmax = gal['vmin'], gal['vmax']
        # kDict = _readKin( nDir )
        # vmax = np.max(np.abs(kDict['v']))
        # smin, smax = np.min(kDict['s']), np.max(kDict['s'])
        # ndmr = [200., smax]

        # Get aperture positions
        apI = INF['aperture']
        theta = apI['angle']
        del apI
        print("{: <20s}{: <15.5f}".format('Theta:', theta))

        ang = np.radians(theta)

        biI = INF['bins']
        grid = np.array(biI['grid'], dtype=int).ravel() - 1
        nbins = np.max(grid).astype(int) + 1
        ss = np.where(grid >= 0)[0]
        # sGrid = grid[ss]
        # xtss, ytss = xt[ss], yt[ss]
        xtss, ytss = gal['pmoms']['x'], gal['pmoms']['y']
        tlx = (xtss * np.cos(np.radians(theta)) -
               ytss * np.sin(np.radians(theta)))
        tly = (xtss * np.sin(np.radians(theta)) +
               ytss * np.cos(np.radians(theta)))

        radius = np.sqrt((tlx - x)**2 + (tly - y)**2)
        rsore = np.argsort(radius)
        fr = np.where(radius < 3)[0]  # 3'' radius
        sfr = rsore[fr]

        aw = grid[ss][sfr]

        mask = np.zeros_like(grid[ss], dtype=bool)
        mask[sfr] = True

        plt.clf()
        dispp(xtss[mask], ytss[mask], gal['moms']['1'][grid[ss]]
              [mask], pixelsize=PIXS, angle=theta, vmin=-0.2, vmax=0.2)
        plt.savefig(opj(bDir, "aper_x{:3.3f}y{:3.3f}.png".format(
            float(x), float(y))), format='png')
        plt.close('all')
    except:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        print(f"LINE {exc_traceback.tb_lineno}\n{'': <4s}{exc_type}\n"\
              f"{'': <4s}{exc_value}")
        pdb.set_trace()

    print('Apers:')
    print(aw)

# ------------------------------------------------------------------------------

def _mlPowerLaw(ellR, ML0, delML, tau):
    """
    The M_*/L power-law function from Mitzkus+16
    """
    return ML0 + delML * (1. - 10.**(-ellR / tau))

# ------------------------------------------------------------------------------

def _qRatio(inc, qObsFlat, deg=True):
    """
    Converts inclination to intrinsic axis ratio based on the flattest projected
        axis ratio, using Eq. (14) from Cappellari et al. (2008)
    Args
    ----
        inc (array): the inclination, in [degrees or radians]
        qObsFlat (array): the projected axis ratio for each inclination
        deg (bool): toggles whether `inc` is specified in [degrees]
    Returns
    -------
        qObs (array): the intrinsic axis ratio
    """
    qObsFlat = np.atleast_1d(qObsFlat)
    if deg:
        inc = np.radians(inc)
    qObs = np.sqrt(qObsFlat**2 - np.cos(inc)**2) / \
        np.sin(inc)  # q for oblate spheroid
    return qObs


"""
To find the inclination from the axis ratio, one must invert the equation evluated in ``_qRatio``.
The result of this is
    i = arccos( sqrt( (q^2 - (q')^2)/(q^2 - 1) ) )
"""

# ------------------------------------------------------------------------------

def _mgeProfile(r, mass, sigma, q):
    """
    This function computes the radial profile of a given MGE
    Args
    ----
        r (float, arr:float): the values at which to compute the profile
        mass (arr:float): the mass density counts, of length `nG`
        sigma (arr:float): the dispersion, of length `nG`, in physical units
        q (arr:float): the axis ratios, of length `nG`
    """

    r = np.atleast_1d(r)[:, np.newaxis]

    e = np.sqrt(1. - (q**2))
    total = np.sum(
        (
            mass * np.exp(-(r**2) / (2. * (sigma**2))) * sserf(
                (r * e) / (q * sigma * np.sqrt(2.))
            )
        ) /
        (4. * np.pi * (sigma**2) * r * e),
        axis=1  # Do the summation for each r value
    )
    return total

# ------------------------------------------------------------------------------

def _sersicSB(R, Ie, A0, bN, n):
    """
    This function returns the surface brightness as a Sersic function
    Args
    ----
        R (arr:float): the array of radii to sample, in units of R_e
        Ie (float): the surface brightness at R_e
        bN (float): the characteristic scale, which usually depends on `n`. For
            n=4, k~7.669. for n=1, k~1.68
        n (float): the Sersic index
    Returns
    -------
        Ir (arr:float): the surface brightness of a Sersic profile at points
            along `R`
    """
    return (Ie * np.exp(- bN * ((R**(1.0 / n)) - 1.0))) + A0

# ------------------------------------------------------------------------------

def intrDensNFW(R, rS, rhoS):
    """
    This function computes the 3D density of an NFW profile. It has been
        transcoded from `NFW_3ddensity` in `schw_enclosemass.pro`
    Args
    ----
        R (arr:float): the radius array, of shape (N,)
        rS (float): the break radius of the NFW profile
        rhoS (float): the density at `rS`
    Returns
    -------
        density (arr:float): the density of an NFW profile, of shape (N,)
    """
    density = rhoS / ((R / rS) * (1.0 + R / rS)**2)

    return density

# ------------------------------------------------------------------------------

def encMassNFW(R, rS, rhoS):
    """
    This function computes the enclosed mass of an NFW profile. It has been
        transcoded from `NFW_enclosemass` in `schw_enclosemass.pro`
    Args
    ----
        R (arr:float): the radius array, of shape (N,)
        rS (float): the break radius of the NFW profile
        rhoS (float): the density at `rS`
    Returns
    -------
        M (arr:float): the enclosed mass within a sphere of radius R[i] of an
            NFW profile, of shape (N,)
    """
    mass = 4.0 * np.pi * rhoS * (rS**3) * (
        np.log((rS + R) / rS) - R / (rS + R)
    )

    return mass

# ------------------------------------------------------------------------------

def specColour(lmin, lmax, wave, spec):
    """
    This function computes a `photometric colour' from a spectrum based on the
        specified band pass
    Args
    ----
        lmin (float): the lower bound on the band pass, in [nm]
        lmax (float): the upper bound on the bans pass, in [nm]
        wave (arr:float): the wavelength of the spectrum, in [nm]
        spec (arr:float): the un-nnormalised spectrum
    Returns
    -------
        colour (float): the total flux within the band pass
    """

    mask = (lmin <= wave) & (wave <= lmax)
    colour = np.nansum(np.compress(mask, spec, axis=0), axis=0)

    return colour

# ------------------------------------------------------------------------------

def VI2gi(VI):
    """
    This function converts a Johnson V-I into an SDSS g-i colour
    Args
    ----
        VI (arr:float): Johnson V-I colour
    Returns
    -------
        gi (arr:Float): SDSS g-i colour
    """
    gi = (VI - 0.364) / 0.675
    return gi

# ------------------------------------------------------------------------------

def gi2VI(gi):
    """
    This function converts an SDSS g-i into a Johnson V-I colour
    Args
    ----
        gi (arr:Float): SDSS g-i colour
    Returns
    -------
        VI (arr:float): Johnson V-I colour
    """
    VI = 0.675 * gi + 0.364
    return VI

# ------------------------------------------------------------------------------

def updateGal(galaxy):
    """
    This function reads in the editable `json` property file and updates the
        non-readable galaxy dictionary object.
    Args
    ----
        galaxy (str): the galaxy to update
    """
    dDir = _ddir()
    pfn = curdir/'obsData'/f"{galaxy}.xz"
    jfn = dDir/'galaxy-props'/f"{galaxy}.json"

    gal = Load.lzma(pfn)
    JS = Load.json(jfn)

    for key in JS:
        if key in gal.keys():
            print(f"Updating {key}\n{'': <4s}{'from': <7s}{gal[key]}\n"\
                f"{'': <4s}{'to': <7s}{JS[key]}")
        gal[key] = JS[key]

    Write.lzma(pfn, gal)

# ------------------------------------------------------------------------------

def quickChi2(galaxy=None, mPath=None, key=None, bML=None, SN=100,
    full=False, redraw=False):
    r"""
    This function compares the model and data by producing figures of relevant
        quantities for a specific coordinate in the parameter space
    Args
    ----
        galaxy (str): the galaxy name
        mPath (str): the directory containing the input and output directories
        key (str): an appropriately-formatted string containing part of the
            filename of a specific model
        nML (float): the M/L to analyse
        SN (int): the S/N of the data
        full (bool): toggles whether the full spectral range was used for the
            spectral fitting
        redraw (bool): toggles whether to read in all models
    Returns
    -------
        bChiR (float): the chi^2 of the kinematic moments of the best-fitting
            model
        bBH (float): the black-hole mass of the best-fitting model
        bQ (float): the q value of the best-fitting model
        bP (float): the p value of the best-fitting model
        bU (float): the u value of the best-fitting model
        bDM (float): the dark-matter scaling of the best-fitting model
        bML (float): the mass-to-light ratio of the best-fitting model
        KINCHI (arr:float): the \chi^2 measurements for all models, of shape
            `(N,)`
        cData (arr:float): the positions in parameter-space of all models, of
            shape `(N, nParam)`
    """

    if not mPath:
        bDir = copy(curdir)
    else:
        bDir = curdir/'tri_models'/mPath

    try:
        INF = Load.lzma(bDir/'infil.xz')
        nKin = int(INF['kin']['nbins'] * INF['nn']['nGH'])
        qMin = np.min([INF['parameters']['tMGE'].q.min(),
            INF['parameters']['sMGE'].q.min()])

        Cfn = bDir/'chi2.csv'
        Dfn = bDir/'chi2.dat'

        mCoords, modDirs = _coordExtr(bDir, key=key)
        nFiles = len(mCoords)
        if (nFiles < 1) and (not Cfn.is_file()):
            raise RuntimeError('No completed models.')
        print(f"Models: {nFiles:04d}")

        if not isinstance(key, type(None)):
            KINCHI = np.empty(nFiles, dtype=float)
            params = []
            for ki, (lKey, mCoord) in enumerate(zip(modDirs, mCoords)):
                kinChi2 = _nnProc(galaxy, mPath, lKey, method='comp')
                KINCHI[ki] = kinChi2
                # allCHI[ki,:] = allChi2
                # CHI[ki] = chi2
                params += [[kinChi2, *mCoord]]
            chiR = KINCHI * nKin / np.nanmin(KINCHI)

            params = np.asarray(params)
            mDirs = np.asarray(mCoords)

            if not isinstance(bML, type(None)):
                sor = np.bitwise_or.reduce(np.apply_along_axis(
                    lambda xn: np.isclose(xn, bML), 1, params), axis=1)
            else:
                sor = np.argsort(chiR)

            chiR = chiR[sor]
            KINCHI = KINCHI[sor]
            mDirs = mDirs[sor, :]
            params = params[sor, :]

        elif redraw or not Cfn.is_file():
            # CHI = np.empty(nFiles, dtype=float)
            # allCHI = np.empty([nFiles, nGH], dtype=float)
            KINCHI = np.empty(nFiles, dtype=float)
            params = []
            for ki, (lKey, mCoord) in enumerate(zip(modDirs, mCoords)):
                kinChi2 = _nnProc(galaxy, mPath, lKey, method='comp')
                KINCHI[ki] = kinChi2
                # allCHI[ki,:] = allChi2
                # CHI[ki] = chi2
                params += [[kinChi2, *mCoord]]
            chiR = KINCHI * nKin / np.nanmin(KINCHI)

            params, pindx = np.unique(params, axis=0, return_index=True)

            chiR = chiR[pindx]
            KINCHI = KINCHI[pindx]
            mDirs = np.take(mCoords, pindx, axis=0)
            # params[:,0] = params[:,0]*nKin/np.nanmin(params[:,0])
            nFiles = len(chiR)

            # Write out outputs
            symbs = ['' for ji in freez] # `freez` includes M/L
            symbs[-2] = '+'
            outC = ''
            outD = ''
            for fi in range(nFiles):
                outC += f"{params[fi, 0]: <.5f},{chiR[fi]: <.9f},"+\
                    ','.join([f"{ppi: <{symbs[si]}.7f}"
                    for si, ppi in enumerate(params[fi, 1:])])+'\n'
                outD += f"{params[fi, 0]: <15.5f} {chiR[fi]: <19.9f} "+\
                    ''.join([f"{ppi: <{symbs[si]}12.7f}"
                    for si, ppi in enumerate(params[fi, 1:])])+'\n'
            oup = open(Cfn, 'w+')
            oup.write(f"{nFiles:05d}\n")
            oup.write(outC)
            oup.close()
            oup = open(Dfn, 'w+')
            oup.write(f"{nFiles:05d}\n")
            oup.write(outD)
            oup.close()

        else:
            oup = open(Cfn, 'r+')
            nDirs = int(oup.readline())
            params = np.array([mp.strip().split(',') for mp in
                oup.readlines()], dtype=float)
            oup.close()
            KINCHI = params[:, 0]
            tempChiR = params[:, 1]
            mDirs = params[:, 2:]
            params = np.delete(params, 1, 1) # remove `chiR` column
            nCFiles = KINCHI.size

            MCKeys = np.array([_gridKey(*mCoord) for mCoord in mCoords])
            MDKeys = np.array([_gridKey(*mCoord) for mCoord in mDirs])
            tdCoords = np.setdiff1d(MCKeys, MDKeys)

            print(f"New models: {tdCoords.size:04d}")

            for lKey in tdCoords:
                lKey = rReplace(lKey, keySep, os.sep, 1)
                mCoord = [*_deetExtr(lKey).values()]
                kinChi2 = _nnProc(galaxy, mPath, lKey, method='comp')
                KINCHI = np.append(KINCHI, kinChi2)
                # allCHI = np.append( allCHI, [allChi2], axis=0 )
                # CHI = np.append( CHI, chi2 )
                params = np.append(params, [[kinChi2, *mCoord]], axis=0)
                mDirs = np.append(mDirs, [[*mCoord]], axis=0)

            chiR = params[:, 0]*nKin/np.nanmin(params[:, 0])

            params, pindx = np.unique(params, axis=0, return_index=True)

            chiR = chiR[pindx]
            KINCHI = KINCHI[pindx]
            mDirs = np.take(mDirs, pindx, axis=0)
            nFiles = params.shape[0]

            # Write out outputs
            symbs = ['' for ji in freez] # `freez` includes M/L
            symbs[-2] = '+'
            outC = ''
            outD = ''
            for fi in range(nFiles):
                outC += f"{params[fi, 0]: <.5f},{chiR[fi]: <.9f},"+\
                    ','.join([f"{ppi: <{symbs[si]}.7f}"
                    for si, ppi in enumerate(params[fi, 1:])])+'\n'
                outD += f"{params[fi, 0]: <15.5f} {chiR[fi]: <19.9f} "+\
                    ''.join([f"{ppi: <{symbs[si]}12.7f}"
                    for si, ppi in enumerate(params[fi, 1:])])+'\n'
            oup = open(Cfn, 'w+')
            oup.write(f"{nFiles:05d}\n")
            oup.write(outC)
            oup.close()
            oup = open(Dfn, 'w+')
            oup.write(f"{nFiles:05d}\n")
            oup.write(outD)
            oup.close()

        # Corner data
        cData = np.array(params[:, 1:]).T
        bMC = mDirs[0]
        bChi = KINCHI[0]
        bChiR = chiR[0]
        chiR = KINCHI*nKin/np.nanmin(KINCHI)
        plotChi = (chiR - np.nanmin(chiR)) / np.sqrt(2.*nKin)
        bBH, bQ, bP, bU, bDM, bDF, bML = bMC
        bMKey = _gridKey(bBH, bQ, bP, bU, bDM, bDF)
        bLKey = rReplace(_gridKey(bBH, bQ, bP, bU, bDM, bDF, bML),
            keySep, plp.os.sep, 1)
        oneSig = plotChi <= 1.
        stderr = np.nanstd(params[oneSig, 1:], axis=0)
        lls = dict(bh=dict(value=bBH, label=r'$\log_{10}(M_\bullet)$',
                str='m_BH', ster=stderr[0]),
            q=dict(value=bQ, label=r'$q$', str='Q', ster=stderr[1]),
            p=dict(value=bP, label=r'$p$', str='P', ster=stderr[2]),
            u=dict(value=bU, label=r'$u$', str='U', ster=stderr[3]),
            dm=dict(value=bDM, label=r'$C_{\rm DM}$', str='C_DM',
                ster=stderr[4]),
            df=dict(value=bDF, str='f_DM', label=\
                r'$\log_{10}\left[f_{\rm DM}\left(r_{200}\right)\right]$',
                ster=stderr[5]),
            ml=dict(value=bML, label=r'$\Upsilon$', str='M/L', ster=stderr[6]))

        bStr = ''.join([
            f"{'Best': <5s}{lls[fs]['str']: <25s}{lls[fs]['value']: <15.7} "\
            f"+/- {lls[fs]['ster']: <5.4}\n" for fs in freez])
        bst = [float(f'{x1: 3.7f}') for x1 in Cu.oneQPUtoTPP(bQ, bP, bU, qMin)]
        bStr += f"{'(theta, phi, psi):': <30s}{bst}\n"
        bStr += f"{'TotalNKin(=chi2_r)': <30s}{nKin: <15d}\n"
        bStr += f"{'kin chi2': <30s}{bChi: <15.7f}\n"
        bStr += f"{'kin chi2 / DOF': <30s}{bChi/nKin: <15.7f}\n"
        # bStr += "{: <20s}{: <15.2f}\n".format('kin chi2_r', bChiR)
        bStr += f"{'Key': <10s}\n{'': <4s}{bMKey: <75s}"
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        print(f"LINE {exc_traceback.tb_lineno}\n{'': <4s}{exc_type}\n"\
              f"{'': <4s}{exc_value}")
        pdb.set_trace()

    print(bStr)

    return bChiR, bBH, bQ, bP, bU, bDM, bDF, bML, params

# ------------------------------------------------------------------------------

def reverseGenINF(galaxy, mPath):
    bDir = curdir/'tri_models'/mPath
    iDir = bDir/'infil'
    ifn = bDir/'infil.xz'

    INF = dict()

    # Aperture
    minX, maxX, nX, minY, maxY, nY, angle = Read.aperture(iDir/\
        'aperture.dat')
    INF['aperture'] = dict()
    INF['aperture']['mmX'] = np.array([minX, maxX])
    INF['aperture']['mmY'] = np.array([minY, maxY])
    INF['aperture']['angle'] = angle
    INF['aperture']['nXY'] = np.array([nX, nY], dtype=int)

    # Bins
    _, grid = Read.bins(iDir/'bins.dat')
    INF['bins'] = dict()
    INF['bins']['nXY'] = np.array([nX, nY], dtype=int)
    INF['bins']['grid'] = grid
    tgrid = np.array(grid, dtype=int).T.ravel()-1
    ss = np.nonzero(tgrid >= 0)
    uPix, pInverse, pCounts = np.unique(tgrid[ss], return_inverse=True,
        return_counts=True)
    INF['bins']['pCountsBin'] = pCounts
    INF['bins']['pCountsPix'] = pCounts[pInverse].T

    # Kinematics
    nbins, nMom, KIN = Read.kinData(iDir/'kin_data.dat')
    INF['kin'] = dict()
    INF['kin']['nbins'] = nbins
    INF['kin']['moms'] = KIN
    INF['kin']['pmoms'] = KIN
    INF['kin']['nMom'] = nMom

    # Parameters
    param = Read.parameters(iDir/'parameters.in')
    paras = Read.parameters(iDir/'paramsb.in')
    INF['parameters'] = dict()
    INF['parameters']['sMGE'] = Mge(paras['mCounts'], paras['mSigmaArc'],
        paras['mQ'], paras['mPsiOff'], 'flux')
    INF['parameters']['angle'] = param['PA']
    INF['parameters']['sbML'] = paras['ML']
    INF['parameters']['gpML'] = param['ML']
    INF['parameters']['tMGE'] = Mge(param['mCounts'], param['mSigmaArc'],
        param['mQ'], param['mPsiOff'], 'mass')
    INF['parameters']['distance'] = param['distance']
    INF['parameters']['theta'] = param['theta']
    INF['parameters']['phi'] = param['phi']
    INF['parameters']['psi'] = param['psi']
    INF['parameters']['mBH'] = param['mBH']
    INF['parameters']['bhSoft'] = param['bhSoft']
    INF['parameters']['nE'] = param['nE']
    INF['parameters']['rLogMin'] = param['rLogMin']
    INF['parameters']['rLogMax'] = param['rLogMax']
    INF['parameters']['nI2'] = param['nI2']
    INF['parameters']['nI3'] = param['nI3']
    INF['parameters']['nDith'] = param['nDith']
    INF['parameters']['dmType'] = param['dmType']
    INF['parameters']['nDM'] = param['nDM']
    INF['parameters']['dmParams'] = param['dmParams']

    # Orblib
    oPeriod, oPoints, oStart, oNumber, oacc, nPSF, nGs, psfCounts,\
        psfSigmas, nAper, apertures, usePSF, hWidth, hCen, hBins, useBin =\
        Read.orbin(iDir/'orblib.in')
    bPeriod, bPoints, bStart, bNumber, bacc, nPSF, nGs, psfCounts,\
        psfSigmas, nAper, apertures, usePSF, hWidth, hCen, hBins, useBin =\
        Read.orbin(iDir/'orblib.in')
    INF['orblib'] = dict()
    INF['orblib']['nOP'] = oPeriod
    INF['orblib']['nPSTube'] = oPoints
    INF['orblib']['nPSBox'] = bPoints
    INF['orblib']['accTube'] = oacc
    INF['orblib']['accBox'] = bacc
    INF['orblib']['stOrb'] = oStart
    INF['orblib']['nIO'] = oNumber
    INF['orblib']['psfs'] = nPSF
    INF['orblib']['ngPSF'] = [nGs]
    INF['orblib']['ws'] = np.array([psfCounts, psfSigmas]).T
    INF['orblib']['nAP'] = nAper
    INF['orblib']['usePSF'] = usePSF
    INF['orblib']['histP'] = [hWidth, hCen, hBins]
    INF['orblib']['useBIN'] = useBin

    # Triaxmass
    INF['triaxmassbin'] = dict()
    INF['triaxmassbin']['nAP'] = nAper
    INF['triaxmassbin']['ngPSF'] = [nGs]
    INF['triaxmassbin']['ws'] = INF['orblib']['ws']

    # NNLS
    reg, nMom, relErrs, lumErr, sbErr, velScale, NNLSSolve = Read.nnin(iDir/\
        'nn.in')
    INF['nn'] = dict()
    INF['nn']['reg'] = reg # regularisation
    INF['nn']['nGH'] = nMom # number of Gauss-Hermite
    INF['nn']['ghSysErr'] = relErrs
    INF['nn']['errL'] = lumErr
    INF['nn']['errI'] = sbErr
    INF['nn']['vScale'] = velScale
    INF['nn']['nnType'] = NNLSSolve

    Write.lzma(ifn, INF)
    Write.lzma(curdir/'obsData'/f"{galaxy}.xz", dict(distance=param['distance']
        ))

# ------------------------------------------------------------------------------
