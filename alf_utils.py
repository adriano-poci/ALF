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
v1.9:   Added `oneSpec` to analyse a given spectral fit in isolation. 7 July
            2023
v1.10:  Pass `dcName` to `alf_aperRead.py` in `alfWrite`. 14 July 2023
v1.11:  Added much-improved `spec` figure to `oneSpec`. 16 November 2023
v1.12:  Vectorised `getM2L` and `getMass`. 20 December 2023
v1.13:  Read `NCPU` from `qProps`;
        Use `RZ` in `alfWrite` to set initial guess of the systemic velocity. 11
            March 2024
v1.14:  Make `alf*.qsys` write to array-aware output and error files. 24 March
            2024
v1.15:  Added `checkResults` to ensure every spectrum has been fit, and enable
            re-fitting if needed;
        Correctly implemented MPI functionality in sbatch scripts. 26 March 2024
v1.16:  Allow to not use Slurm arrays if preferred. 1 May 2024
v1.17:  Added `SolarAbundance` and `SumMetals` to get total metallicity from
            individual abundances. 7 June 2025
v1.18:  Cleaned out unecessary functions. 26 July 2025
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
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import seaborn as sns
from functools import partial
from tqdm import tqdm
import subprocess as sp
from inspect import getargvalues as ingav, currentframe as incf
from svo_filters import svo

from alf.Alf import Alf

# Custom modules
from dynamics.IFU.Galaxy import Mge, Schwarzschild, Redshift
from dynamics.IFU.FileIO import Load, Write, Read
from dynamics.IFU.Constants import Constants, UnitStr
from cythonModules import C_utils as Cu
from spectres import spectres

from plotbin.sauron_colormap import register_sauron_colormap as srsc

curdir = plp.Path(__file__).parent
icefire = sns.color_palette('icefire', as_cmap=True)
rocket = sns.color_palette('rocket', as_cmap=True)
rocketr = sns.color_palette('rocket_r', as_cmap=True)

UTS = UnitStr()

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
    if 'NGC4365' in galaxy:
        np.savetxt(curdir/'indata'/f"{galaxy}_SN{SN:02d}_aperture.dat",
            np.column_stack((tPix, binSpec[:, 0], binStat[:, 0],
            weights, velRes)), fmt='%20.10f',
            header=f"{wRange[0]*1e-4:.5f} {wRange[1]*1e-4:.5f}")
    else:
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

def alfWrite(galaxy, SN, nbins, RZ, hours=48, priors=True, dcName='',
    qProps=dict(timeMax=168, module=[], array=True)):

    gDir = curdir/f"{galaxy}{dcName}"

    hours = np.ceil(hours).astype(int)
    hours = np.min((hours, qProps['timeMax']))
    nCPU = int(qProps.pop('NCPU', 16))
    sarray = qProps.pop('sarray', True)
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

    ws = '' # whitespace
    nl = r'\n' # newline

    if sarray:
        for ss in range(nScripts):

            add = nSteps*ss
            top = nSteps
            if remain-nSteps < 0:
                top = copy(remain)
            remain -= nSteps

            sStr = ''
            sStr += u'#!/bin/bash -l\n'
            if 'owner' in qProps.keys():
                sStr += f"#SBATCH -A {str(qProps['owner'])}\n"
            if 'queue' in qProps.keys():
                sStr += f"#SBATCH -p {str(qProps['queue'])}\n"
            sStr += f'#SBATCH --job-name="alf_{galaxy}_SN{SN:02d}"\n'
            sStr += f'#SBATCH -D "{str(gDir)}"\n'
            sStr += f"#SBATCH --time={timeStr}\n"
            sStr += u'#SBATCH --nodes=1\n'
            sStr += f"#SBATCH --ntasks={nCPU:d}\n"
            sStr += u'#SBATCH --mem-per-cpu=2200\n'
            sStr += u'#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n'
            sStr += u'#SBATCH --mail-user=adriano.poci@physics.ox.ac.uk\n'
            # sStr += f'#SBATCH -o "{str(gDir)}/out_%A_%a.log" '\
            #     u'# Standard out to galaxy\n'
            # sStr += f'#SBATCH -e "{str(gDir)}/out_%A_%a.log" '\
            #     u'# Standard err to galaxy\n'
            # sStr += f'#SBATCH --open-mode=append\n'
            sStr += f'#SBATCH --array=0-{top}\n\n'

            sStr += u'source ${HOME}/.bashrc\n\n'
            sStr += u'module purge\n'
            for mod in qProps['module']:
                sStr += f"module load {'/'.join(mod)}\n"
            sStr += f'export ALF_HOME={curdir}{plp.os.sep}\n'
            sStr += f'export PSM2_CUDA=0\n\n'
            sStr += u'cd ${ALF_HOME}\n'
            if add > 0:
                sStr += u'declare idx=$(printf %04d '\
                    f"$((${{SLURM_ARRAY_TASK_ID}} + {add})))\n"
            else:
                sStr += u'declare idx=$(printf %04d ${SLURM_ARRAY_TASK_ID})\n'
            sStr += u'mpirun --bind-to core --map-by core '\
                f'./{galaxy}{dcName}/bin/alf.exe '\
                f'"{galaxy}_SN{SN:02d}_${{idx}}" '\
                f'2>&1 | tee -a "{galaxy}{dcName}/out_${{idx}}.log"\n'

            sf = io.open(gDir/f"alf{ss:02d}.qsys", 'w+', newline='')
            sf.write(sStr)
            sf.flush()
            sf.close()
    else:
        sStr = ''
        sStr += u'#!/bin/bash -l\n'
        if 'owner' in qProps.keys():
            sStr += f"#SBATCH -A {str(qProps['owner'])}\n"
        if 'queue' in qProps.keys():
            sStr += f"#SBATCH -p {str(qProps['queue'])}\n"
        # sStr += f'#SBATCH --job-name="alf_{galaxy}_SN{SN:02d}"\n'
        sStr += f'#SBATCH -D "{str(gDir)}"\n'
        sStr += f"#SBATCH --time={timeStr}\n"
        sStr += u'#SBATCH --nodes=1\n'
        sStr += f"#SBATCH --ntasks={nCPU:d}\n"
        sStr += u'#SBATCH --mem-per-cpu=2200\n'
        sStr += u'#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n'
        sStr += u'#SBATCH --mail-user=adriano.poci@students.mq.edu.au\n\n'
        # sStr += f'#SBATCH -o "{str(gDir)}/out_%A_%a.log" '\
        #     u'# Standard out to galaxy\n'
        # sStr += f'#SBATCH -e "{str(gDir)}/out_%A_%a.log" '\
        #     u'# Standard err to galaxy\n'
        # sStr += f'#SBATCH --open-mode=append\n\n'
        sStr += u'source ${HOME}/.bashrc\n\n'
        for mod in qProps['module']:
            sStr += f"module load {'/'.join(mod)}\n"
        sStr += f'export ALF_HOME={curdir}{plp.os.sep}\n'
        sStr += f'export PSM2_CUDA=0\n\n'
        sStr += u'cd ${ALF_HOME}\n'
        sStr += u'while getopts ":d:" arg; do\n'
        sStr += f'{ws: <4s}case $arg in\n'
        sStr += f'{ws: <8s}d) idx=$OPTARG;;\n'
        sStr += f'{ws: <4s}esac\n'
        sStr += 'done\n\n'
        sStr += u'declare pidx=$(printf %04d ${idx})\n'
        sStr += u'mpirun --bind-to core --map-by core '\
            f'./{galaxy}{dcName}/bin/alf.exe '\
            f'"{galaxy}_SN{SN:02d}_${{pidx}}" '\
            f'2>&1 | tee -a "{galaxy}{dcName}/out_${{pidx}}.log"\n'
        sf = io.open(gDir/'alfS.qsys', 'w+', newline='')
        sf.write(sStr)
        sf.flush()
        sf.close()

    sStr = ''
    sStr += u'#!/bin/bash -l\n'

    if 'owner' in qProps.keys():
        sStr += f"#SBATCH -A {str(qProps['owner'])}\n"
    if 'queue' in qProps.keys():
        sStr += f"#SBATCH -p {str(qProps['queue'])}\n"
    sStr += f'#SBATCH --job-name="alf_{galaxy}_SN{SN:02d}_aperture"\n'
    sStr += f'#SBATCH -D "{str(gDir)}"\n'
    sStr += f"#SBATCH --time={timeStr}\n"
    sStr += u'#SBATCH --nodes=1\n'
    sStr += f"#SBATCH --ntasks={nCPU:d}\n"
    sStr += u'#SBATCH --mem-per-cpu=2200\n'
    sStr += u'#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL\n'
    sStr += u'#SBATCH --mail-user=adriano.poci@students.mq.edu.au\n'
    sStr += f'#SBATCH -o "{str(gDir)}/out.log" '\
        u'# Standard out to galaxy\n'
    sStr += f'#SBATCH -e "{str(gDir)}/out.log" '\
        u'# Standard err to galaxy\n'
    sStr += f'#SBATCH --open-mode=append\n\n'

    sStr += u'source ${HOME}/.bashrc\n\n'
    for mod in qProps['module']:
        sStr += f"module load {'/'.join(mod)}\n"

    sStr += f'export ALF_HOME={curdir}{plp.os.sep}\n'
    sStr += f'export PSM2_CUDA=0\n\n'
    sStr += u'### Compile clean version of `alf`\n'
    sStr += u'cd ${ALF_HOME}src\n'
    sStr += u'cp alf.perm.f90 alf.f90\n'
    sStr += u'## Set fallback value to systemic velocity of the galaxy\n'
    sStr += u'# Replace the placeholder value in `sed` script\n'
    sStr += f"cp ${{ALF_HOME}}{galaxy}{dcName}/alf_replace.sed "\
        f"${{ALF_HOME}}{galaxy}{dcName}/alf_replace.tmp\n"
    sStr += f'sed -i "s/velz = 999/velz = {RZ.toVSys()}/g" '\
        f"${{ALF_HOME}}{galaxy}{dcName}/alf_replace.tmp\n"
    sStr += u'# Run `sed` using the multi-line script\n'
    sStr += u'# Pipe to temporary file\n'
    sStr += f"sed -n -f ${{ALF_HOME}}{galaxy}{dcName}/alf_replace.tmp "\
        'alf.f90 >> alf_tmp.f90\n'
    sStr += u'mv alf_tmp.f90 alf.f90\n'
    sStr += f"rm ${{ALF_HOME}}{galaxy}{dcName}/alf_replace.tmp\n"
    sStr += u'# Remove prior placeholders on velz\n'
    sStr += u'sed -i "s/prlo%velz = -999.0/prlo%velz = '\
        f'{Redshift(redshift=RZ.zShift*0.75).toVSys()}/g" alf.f90\n'
    sStr += u'sed -i "s/prhi%velz = 999.0/prhi%velz = '\
        f'{Redshift(redshift=RZ.zShift*1.25).toVSys()}/g" alf.f90\n'
    sStr += f'sed -i "s/velz=9999/velz={RZ.toVSys()*0.98})/g" '\
        u'alf.f90\n'
    sStr += u'make clean && make all && make clean\n\n'
    sStr += u'cd ${ALF_HOME}\n'
    sStr += f"[[ ! -d {galaxy}{dcName}/bin ]] && mkdir -p {galaxy}{dcName}/bin\n"
    sStr += f"rm {galaxy}{dcName}/bin/*\n"
    sStr += f"cp bin/* {galaxy}{dcName}/bin/\n"
    sStr += u'# Run aperture fit\n'
    sStr += u'mpirun --bind-to core --map-by core '\
        f'./{galaxy}{dcName}/bin/alf.exe "{galaxy}_SN{SN:02d}_aperture" 2>&1 '\
        f'| tee -a "{galaxy}{dcName}/out_aperture.log"\n\n'
    sStr += '# Read in the aperture fit\n'
    sStr += u"Ipy='ipython --pylab --pprint --autoindent'\n"
    sStr += f"galax='{galaxy}'\n"
    sStr += f"SN={SN:d}\n"
    sStr += u'pythonOutput=$($Ipy alf_aperRead.py -- -g "$galax" -sn "$SN" '\
        f'-dc "{dcName}")\n'
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
        sStr += u'cd ${ALF_HOME}src\n'
        sStr += u'cp alf.perm.f90 alf.f90\n'
        sStr += u'# `bc` arithmetic to define the lower and upper velocity bounds\n'
        sStr += u'newVLo=$(bc -l <<< "(${aperKin[0]} - ${aperKin[1]}) - '\
            u'10.0 * (${aperKin[2]} + ${aperKin[3]})")\n'
        sStr += u'newVHi=$(bc -l <<< "(${aperKin[0]} + ${aperKin[1]}) + '\
            u'10.0 * (${aperKin[2]} + ${aperKin[3]})")\n'
        sStr += u'sed -i "s/prlo%velz = -999.0/prlo%velz = ${newVLo}/g" alf.f90\n'
        sStr += u'sed -i "s/prhi%velz = 999.0/prhi%velz = ${newVHi}/g" alf.f90\n'
        sStr += f'sed -i "s/velz=9999/velz={RZ.toVSys()*0.98})/g" '\
            u'alf.f90\n'
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
    sStr += f"cp bin/* {galaxy}{dcName}/bin/\n"
    if sarray:
        sStr += f'find "{galaxy}{dcName}" -name "alf*.qsys" -type f -exec '\
        r'sbatch {} \;\n'
    else:
        sStr += f"NBIN={nbins}\n"
        sStr += u'for ((ix=0;ix<=NBIN;ix++)); do\n'
        sStr += f'{ws: <4s}sbatch '\
            f'--job-name="alf_{galaxy}_SN{SN:02d}_${{ix}}_{dcName}"'\
            f" {galaxy}{dcName}/alfS.qsys -d ${{ix}}\n"
        sStr += u'done\n'

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

def getMass(mto, imf1, imf2, imfTop, **kwargs):
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
    mlo   =   kwargs.get('imflo', 0.08) # Low-mass cut-off assumed
    imfhi =   kwargs.get('imfhi', 100.0)  # Upper mass for integration

    mto, imf1, imf2, imfTop = map(np.atleast_1d, [mto, imf1, imf2, imfTop])

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
    mtl = np.where(mto < m3)
    mtg = np.where(mto >= m3)
    getmass[mtl] += m2**(-imf1[mtl]+imf2[mtl])*(mto[mtl]**(-imf2[mtl]+2)-m2**(
        -imf2[mtl]+2))/(-imf2[mtl]+2)
    # otherwise, add the two sections up to mto
    getmass[mtg] += m2**(-imf1[mtg]+imf2[mtg])*(m3**(-imf2[mtg]+2)-m2**(
        -imf2[mtg]+2))/(-imf2[mtg]+2) +\
        m2**(-imf1[mtg]+imf2[mtg])*(mto[mtg]**(-imfTop[mtg]+2)-m3**(
        -imfTop[mtg]+2))/(-imfTop[mtg]+2)

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
    getmass[mtl] += 0.48*m2**(-imf1[mtl]+imf2[mtl])*(nslim**(-imfTop[mtl]+1)
        -m3**(-imfTop[mtl]+1))/(-imfTop[mtl]+1)/imfnorm[mtl]
    getmass[mtl] += 0.48*m2**(-imf1[mtl]+imf2[mtl])*(m3**(-imf2[mtl]+1)
        -mto[mtl]**(-imf2[mtl]+1))/(-imf2[mtl]+1)/imfnorm[mtl]
    getmass[mtl] += 0.077*m2**(-imf1[mtl]+imf2[mtl])*(nslim**(-imfTop[mtl]+2)
        -m3**(-imfTop[mtl]+2))/(-imfTop[mtl]+2)/imfnorm[mtl]
    getmass[mtl] += 0.077*m2**(-imf1[mtl]+imf2[mtl])*(m3**(-imf2[mtl]+2)
        -mto[mtl]**(-imf2[mtl]+2))/(-imf2[mtl]+2)/imfnorm[mtl]
    # Otherwise, only the upper segment.
    getmass[mtg] += 0.48*m2**(-imf1[mtg]+imf2[mtg])*(nslim**(-imfTop[mtg]+1)
        -mto[mtg]**(-imfTop[mtg]+1))/(-imfTop[mtg]+1)/imfnorm[mtg]
    getmass[mtg] += 0.077*m2**(-imf1[mtg]+imf2[mtg])*(nslim**(-imfTop[mtg]+2)
        -mto[mtg]**(-imfTop[mtg]+2))/(-imfTop[mtg]+2)/imfnorm[mtg]

    return getmass

# ------------------------------------------------------------------------------

def getM2L(mfn, logage, zh, imf1, imf2, imfTop, RZ=None, band='F814W',
        photFilt='WFPC2.F814W', **kwargs):
    
    logage, zh, imf1, imf2, imfTop = map(np.atleast_1d, [logage, zh, imf1, imf2,
        imfTop])

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

    mass = getMass(mto, imf1, imf2, imfTop, **kwargs)

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

    tempMag, _, _ = RZ.shiftFnu(nfWave, physSpec, photFilt=photFilt, **kwargs)

    if tempMag <= 0.0:
        return np.full_like(0.0, imf1)

    else:
        # Read in solar spectrum and generate mag sun from filter curve
        swave, snu = map(lambda x: x.value, Read.SolarSpec(dDir/\
            'sun_reference_stis_002.fits'))
        solarMag, _, _ = RZ.shiftFnu(swave, snu, photFilt=photFilt, **kwargs)

        mass2light = mass / 10.0**(2./5. * (solarMag-tempMag))

        np.ma.masked_greater(mass2light, 100.)

        return np.squeeze(mass2light)

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

def oneSpec(spectrum, labels=['velz', 'sigma', 'h3', 'h4', 'logage', 'zH',
        'FeH', 'Na', 'IMF1', 'IMF2',], pplots=['input', 'fit', 'corn'],
        redshift=0.0):
    """
    Plot results for a single isolated run of alf

    Parameters
    ----------
    spectrum : str
        _description_
    labels : list, optional
        _description_, by default ['velz', 'sigma', 'h3', 'h4', 'logage', 'zH', 'FeH', 'Na', 'IMF1', 'IMF2',]
    pplots : list, optional
        _description_, by default ['input', 'fit', 'corn']

    Returns
    -------
    _type_
        _description_
    
    Examples
    --------
    au.oneSpec('SNL1_NFMESOouterError_1arcs_dust', labels=['velz', 'sigma', 'h3', 'h4', 'logage', 'zH', 'FeH', 'Na', 'Ti', 'IMF1', 'C', 'N', 'Si', 'K', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Sr', 'Ba', 'Eu'])
    """    
    import matplotlib.pyplot as plt

    alf = Alf(curdir/'results'/spectrum, mPath=curdir)
    alf.get_total_met()
    alf.normalize_spectra()
    alf.abundance_correct()

    ifn = curdir/'indata'/f"{spectrum}.dat"
    waves, tPix, spec, err, weights, vel = readSpec(ifn)

    if 'input' in pplots:
        fig = plt.figure(figsize=plt.figaspect(1./10.))
        ax = fig.gca()
        for wpair in waves:
            ww = np.where((tPix >= wpair[0]*1e4) & (tPix <= wpair[1]*1e4))[0]
            ax.plot(tPix[ww], spec[ww], lw=0.4, c='r')
        ax.fill_between(tPix, weights*spec.max(), alpha=0.2, facecolor='k',
            zorder=0)
        ax.set_ylim(top=(spec*weights).max()*1.1)
        fig.savefig(curdir/f"{spectrum}_input.pdf", format='pdf')

    if 'fit' in pplots:
        # alf.plot_model(curdir/f"{spectrum}_fit.pdf")
        mwave, model, sinp, merr, _, mres = np.loadtxt(curdir/'results'/\
            f"{spectrum}.bestspec", unpack=True)
        fig = plt.figure(figsize=plt.figaspect(2.5/10.)*0.7)
        gs = gridspec.GridSpec(2, 1, hspace=0, wspace=0)
        ax = fig.add_subplot(gs[0, 0])
        for wpair in waves:
            ww = np.where((tPix >= wpair[0]*1e4) & (tPix <= wpair[1]*1e4))[0]
            ax.plot(tPix[ww], spec[ww], lw=0.5, c='k')
        ax.plot(mwave, model, lw=0.5, c='r')
        ax.fill_between(tPix, (1.0-weights)*spec.max(), alpha=0.2,
            facecolor='k', zorder=0)
        ax.set_ylim(bottom=spec[weights>0][:-2].min()*0.7,
            top=spec[weights>0].max()*1.05)
        ax.set_xlim(min(mwave.min(), tPix.min())-20.,
            max(mwave.max(), tPix.max())+20.)
        ax.set_xticklabels([])
        ax.set_ylabel('Flux')

        ax = fig.add_subplot(gs[1, 0])
        mask = (tPix >= (mwave.min()-1.)) & (tPix <= (mwave.max()+1.))
        newm = np.zeros_like(mwave)
        if np.any(weights[mask] < 0.5):
            temp = tPix[mask][np.ma.getmaskarray(np.ma.masked_less(weights[mask], 0.5))]
            mwm = np.array([np.argmin(np.abs(tempi-mwave)) for tempi in temp])
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
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
        fig.savefig(curdir/f"{spectrum}_spec.pdf")
        fig.savefig(curdir/f"{spectrum}_spec.png")
    
    if 'corn' in pplots:
        from corner import corner
        lidx = np.array([np.where(np.in1d(alf.labels, clab))[0] for clab in
            labels]).ravel()
        # ensure the order of the labels matches the order of the data
        # columns
        plabels = alf.labels[lidx].tolist()
        if 'a' in plabels:
            plabels[plabels.index('a')] = 'O'
        if 'FeH' in plabels:
            plabels[plabels.index('FeH')] = 'Fe'
        fig = plt.figure(figsize=plt.figaspect(1.)*1.6)
        fig = corner(alf.mcmc[:, lidx],
            labels=plabels, smooth=0.8, plot_contours=False, labelpad=0.5,
            max_n_ticks=2, plot_datapoints=False, plot_density=True,
            fig=fig, pcolor_kwargs=dict(cmap=rocket))
        fig.savefig(curdir/f"{spectrum}_corn.png")
    plt.close('all')

    if not (curdir/'results'/f"{spectrum}.bestspec2").is_file() and \
        (curdir/'results'/f"{spectrum}.bestspec").is_file():
        # Generate model on longer wavelength range
        sp.check_call([f"{str(curdir)}/bin/spec_from_sum.exe", spectrum])

    arm = alf.results
    # pdb.set_trace()
    types = arm['Type'].tolist()
    bidx = types.index('chi2')
    eidx = types.index('error')

    MLF814W = getM2L(f"{spectrum}", arm['logage'][bidx], arm['zH'][bidx],
        arm['IMF1'][bidx], arm['IMF1'][bidx], 2.3,
        RZ=Redshift(redshift=redshift), imflo=arm['IMF3'][bidx])
    print(f"{'Param': >15s} | {'Value': >30s}")
    print('-'*48)
    for lab in labels:
        ers = f"{arm[lab][bidx]: .4f} +/- {arm[lab][eidx]: .4f}"
        print(f"{lab: >15s} | {ers: >30s}")
    print(f"{'M/L_{F814W}': >15s} | {MLF814W: >30.4f}")
    pdb.set_trace()
    return alf

# ------------------------------------------------------------------------------

def checkResults(galaxy, SN=100, full=True, dcName=''):
    if not full: # Clip the spectral data if required
        tEnd = 'trunc'
    else:
        tEnd = 'full'
    
    mDir = curdir/f"{galaxy}{dcName}"
    vofs = mDir/f"voronoi_SN{SN:02d}_{tEnd}.xz"
    VO = Load.lzma(vofs)
    nSpat = VO['xbin'].size
    outs = np.sort([xi for xi in plp.Path(curdir/'results').glob(
        f"{galaxy}_SN{SN:02d}_*.mcmc")])[:nSpat] # omit 'aperture'
    miss = []
    rstruc = r'[a-zA-Z\_0-9]+(\_[0-9]{4})'
    idxs = [f"{ji:04d}" for ji in np.arange(nSpat)]
    for ofil in outs:
        rmat = re.search(rstruc, ofil.stem)
        if rmat:
            idx = rmat.groups()[0]
            idxs.pop(idxs.index(idx.lstrip('_')))
    
    if len(idxs) > 0:
        sh.copy2(mDir/'alf00.qsys', mDir/'alfRedo.qsys')
        with open(mDir/'alfRedo.qsys', 'r') as script:
            lines = script.readlines()
        with open(mDir/'alfRedo.qsys', 'w') as script:
            for line in lines:
                script.write(re.sub(r'^#SBATCH --array=.*',
                    rf"#SBATCH --array={','.join([qj.lstrip('0') for qj in idxs])}", line))
    print(f"Need to redo {len(idx):d} spectra.")

# ------------------------------------------------------------------------------

def SolarAbundance(element):
    """
    Returns the solar abundance, as defined in Asplund et al. (2009)
    """
    asplund = Load.json(dDir/'SolarAbundances.json')
    if element not in asplund.keys():
        raise ValueError(f"Element {element} not found in Solar Abundances.")
    
    # in log10(N/H) + 12.0, i.e. [X/H] + 12.0
    return asplund[element]['abundance'] - 12.0
    # np.log10(N_element/N_H)

# ------------------------------------------------------------------------------

def SumMetals(metalDict):
    """
    This function sums the metallicities in a dictionary, where the keys are
        the elements and the values are the abundances in [X/H]
    Args
    ----
        metalDict (dict): a dictionary of elements and their abundances
            in [X/H]
    Returns
    -------
        met (float): the total metallicity, [Z/H]
    """
    # log_10(e/H)_galaxy = [e/H] + log_10(e/H)_sun
    if not isinstance(metalDict, dict):
        raise TypeError("Input must be a dictionary of elements and their abundances.")
    solar = np.array([10.0**SolarAbundance(el) for el in metalDict.keys()])
    # linear (e/H)_solar 
    gal = np.atleast_2d(np.array([10.0**metalDict[el] for el in
        metalDict.keys()]).T).T * solar[:, np.newaxis]
    # linear (e/H)_galaxy
    met = np.log10(np.sum(gal, axis=0) / np.sum(solar))
    # [M/H] = log_10( M_galaxy / M_solar )
    return met

# ------------------------------------------------------------------------------