# -*- coding: utf-8 -*-
"""
    alf_aperRead.py
    Adriano Poci
    Durham University
    2022

    <adriano.poci@durham.ac.uk>

    Platforms
    ---------
    Unix, Windows

    Synopsis
    --------
    This module loads an aperture-spectrum fit, to be executed on the command-
        line.

    Author
    ------
    Adriano Poci <adriano.poci@durham.ac.uk>

History
-------
v1.0:   27 September 2022
v1.1:   Receive `dcName` command-line argument. 14 July 2023
v1.2:   Generate `*.bestspec2` for aperture fit. 7 November 2023
"""
from __future__ import print_function, division

# General modules
import pathlib as plp
from copy import copy
import shutil as su
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import seaborn as sns

# Custom modules
from alf.Alf import Alf
import alf.alf_utils as au
from alf_MUSE import makeSpecFromSum
from dynamics.IFU.Constants import UnitStr


curdir = plp.Path(__file__).parent
dDir = au._ddir()
rocket = sns.color_palette("rocket", as_cmap=True)

UTS = UnitStr()

import argparse
parser = argparse.ArgumentParser(
    description='Read an alf fit to an aperture spectrum.',
    usage='python alf_aperRead.py -g <galaxy> -sn <SN>'
)
parser.add_argument('-g', '--galaxy', dest='galaxy', type=str)
parser.add_argument('-sn', '--SN', dest='SN', type=int)
parser.add_argument('-dc', '--dcName', dest='dcName', type=str)
args = parser.parse_args()

mDir = curdir/f"{args.galaxy}{args.dcName}"
cfn = mDir/'config.xz'
CFG = au.Load.lzma(cfn)


inp = curdir/'indata'/f"{args.galaxy}_SN{args.SN:02d}_aperture.dat"
out = curdir/'results'/f"{args.galaxy}_SN{args.SN:02d}_aperture.mcmc"
print(out)
alf = Alf(out.parent/out.stem, mPath=out.parent)
alf.get_total_met()
alf.normalize_spectra()
alf.abundance_correct()
mIdx = alf.results['Type'].tolist().index('mean')
eIdx = alf.results['Type'].tolist().index('error')

apV = alf.results['velz'][mIdx]
apS = alf.results['sigma'][mIdx]
aph3 = alf.results['h3'][mIdx]
aph4 = alf.results['h4'][mIdx]
apVe = alf.results['velz'][eIdx]
apSe = alf.results['sigma'][eIdx]
aph3e = alf.results['h3'][eIdx]
aph4e = alf.results['h4'][eIdx]


# spectral fit
waves, tPix, spec, err, weights, vel = au.readSpec(inp)
alf.plot_model(curdir/mDir/'specFit_aperture.pdf')
mwave, model, sinp, merr, _, mres = np.loadtxt(out.parent/\
    f"{out.stem}.bestspec", unpack=True)
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
fig.savefig(mDir/'spec_aperture.pdf')

# corner
clabels = ['velz', 'sigma', 'h3', 'h4', 'logage', 'zH', 'IMF1', 'IMF2', 'FeH',
    'Na', 'a', 'Ti', 'C', 'N', 'Mg', 'Ca']
if int(CFG['imf_type']) > 0:
    clabels += ['IMF2']
lidx = np.array([np.where(np.in1d(alf.labels, clab))[0] for clab in clabels]
    ).ravel()
# ensure the order of the labels matches the order of the data columns
plabels = alf.labels[lidx].tolist()
if 'a' in plabels:
    plabels[plabels.index('a')] = 'O'
if 'FeH' in plabels:
    plabels[plabels.index('FeH')] = 'Fe'
from corner import corner

fig = plt.figure(figsize=plt.figaspect(1.)*1.6)
fig = corner(alf.mcmc[:, lidx], labels=plabels, smooth=0.8, plot_contours=False,
    labelpad=0.9, max_n_ticks=2, plot_datapoints=False, plot_density=True,
    fig=fig, pcolor_kwargs=dict(cmap=rocket))
fig.savefig(mDir/'corner_aperture')

makeSpecFromSum(args.galaxy, args.SN, full=True, apers=['aperture'],
    dcName=args.dcName)

print()
print(f"\n{apV:.7f},{apVe:.7f},{apS:.7f},{apSe:.7f},{aph3:.7f},{aph3e:.7f},"\
    f"{aph4:.7f},{aph4e:.7f}")