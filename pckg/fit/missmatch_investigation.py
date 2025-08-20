import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Slider
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from IPython.display import clear_output
import scipy.odr as odr
import imageio
import gc
from matplotlib.widgets import EllipseSelector, RectangleSelector
from matplotlib.pyplot import cm
from scipy.special import eval_legendre
from scipy.optimize import curve_fit
import scipy.fftpack as fft
from matplotlib.colors import LogNorm
import csv
from OAH_refocus import *
import time
import sys
import argparse
import glob
import csv
import matplotlib
import math
from scipy.optimize import curve_fit
from fitfunctions import gaussmod, tfmod, bimodalmod, tfaxialmod, gaussmod_OAH, tfmod_OAH, bimodalmod_OAH
from scipy.ndimage.interpolation import rotate
from scipy.special import zeta

# Import own fit functions
import OAH_functions as f1
# import OAHDEV_functions as f2
from OAHDEV_functions import *
# from ..data_analysis import fits_analysis as fa

# Constants:
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
pix_size = 6.5E-6 / 2.63
# Light field properties
lamb0 = 589.1E-9  # Wavelength
k0 = 2 * np.pi / lamb0  # k-vector



date        = 20231024
shot        = 350
num         = 0
dz_focus    = 0.0084
quad_cut    = "quad1"
cut         = [1031, 1111, 577, 1567]
px_fac      = 1
sc_fac      = 1
ly          = 0
mask_w      = 0.51
peakswithfitting  = False
deconvolution     = False
px_fac      = 1
multi_loop  = 0
el_x        = 10
el_z        = 100
edge_x      = 10
edge_z      = 80
e_w         = 0.1


# ------------------------------------- OPEN FILES --------------------------------------------------
atoms, flat, dark = openFiles(date, shot, num, multi_loop=multi_loop)

# ----------------------------------- MASK THE EDGE -------------------------------------------------
atoms      = f1.squaroid(atoms - dark, width=mask_w)
flat       = f1.squaroid(flat - dark, width=mask_w)

# ------------------------------------- TAKE FFT ----------------------------------------------------
fft_atoms = np.fft.fft2(atoms)
fft_flat = np.fft.fft2(flat)

# ----------------------------------- TRIM THE EDGE -------------------------------------------------
fft_atoms = fft_atoms#[5:-5, 5:-5]
fft_flat = fft_flat#[5:-5, 5:-5]

# ----------------------------- DECONVOLUTION TO CORRECT PIXEL SHAPE --------------------------------
P_k = np.ones(fft_atoms.shape)
pixelsize =  12.5E-6
fft_kx = np.fft.fftfreq(fft_atoms.shape[0], d=pixelsize)  # Discrete FFT Sample Frequency in x
fft_ky = np.fft.fftfreq(fft_atoms.shape[1], d=pixelsize)
P_k = np.sinc(fft_kx * pixelsize * 2)[:, None] * np.sinc(fft_ky * pixelsize * 2)[None, :]
P_k *= px_fac
# P_k = fft.fftshift(P_k)       
fft_atoms_pk = fft_atoms / P_k
fft_flat_pk = fft_flat / P_k



fig, ax = plt.subplots(1, 3, figsize=(15, 9))
im0 = ax[0].imshow(np.log(abs(fft_atoms)))
im1 = ax[1].imshow(P_k)
im2 = ax[2].imshow(np.log(abs(fft_atoms_pk)), vmin=np.min(np.log(abs(fft_atoms))), vmax=np.max(np.log(abs(fft_atoms))))

ax[0].set_title("FFT Atoms")
ax[1].set_title("SINC Function")
ax[2].set_title("FFT Atoms / SINC Function")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
plt.tight_layout()


# ---------------------------- CUT QUADRANT AND MAKE ELLIPSOID  -------------------------------------
all_data = []
for quad_cut in ["quad1", "quad2"]:
    quad1, q1peak  = f1.box_cutter_pad_ellips(fft_atoms_pk, quad_cut, el_x, el_z, edge_x=edge_x, edge_z=edge_z, e_w=e_w, peakswithfitting=peakswithfitting)
    flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat_pk,  quad_cut, indices=q1peak) # same indices as for quad
    quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

    # ---------------------------------- SHIFT FFT TO CENTER  -------------------------------------------
    fft1 = np.fft.fftshift(quad1cut)
    flatfft1 = np.fft.fftshift(flatq1cut)

    # -------------------------------------- REFOCUSING --------------------------------------------------
    fft_kx = np.fft.fftfreq(fft1.shape[1], d=pix_size)  # Discrete FFT Sample Frequency in x
    fft_ky = np.fft.fftfreq(fft1.shape[0], d=pix_size)  # Discrete FFT Sample Frequency in z
    fft_k2 = fft_kx[None, :] ** 2 + fft_ky[:, None] ** 2  # Discrete FFT Sample Frequency in main axes multiplied
    #     ly = 1e6  # .5E6#-3E6#.                                           # Adjusting the fft_ky array
    coma_y_arg = ly * fft_ky[:, None] * (3 * fft_k2 / k0 ** 2) / k0
    lin_y = np.exp(-1j * coma_y_arg)
    focus = np.exp(-1j * dz_focus * np.sqrt(k0 **2 - fft_k2))

    fft1 = fft1 * focus * lin_y
    flatfft1 = flatfft1 * focus * lin_y

    # ------------------------------------- INVERSE FFT -------------------------------------------------
    inv1 = np.fft.ifft2(fft1) / np.fft.ifft2(flatfft1)
    inv1 = inv1[border_x:-border_x, border_z:-border_z]

    # Get Phase
    ang1 = np.angle(inv1)
    ang1 = f1.unwrapper(ang1)

    # Get amp
    amp = np.abs(inv1)
    amp = np.log(amp**2)

    # Normalize
    normfactor = ang1.mean()
    ang1 = ang1 - normfactor
    ang1 = normalize(ang1)[0] 
    all_data.append(ang1)
    
    
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax[0].imshow(all_data[0][600:1450, 800:1400], vmin=-2, vmax=3, cmap="afmhot_r", aspect='auto', origin='lower')
ax[1].imshow(all_data[1][600:1450, 800:1400], vmin=-2, vmax=3, cmap="afmhot_r", aspect='auto', origin='lower')
ax[2].imshow(all_data[0][600:1450, 800:1400] - all_data[1][600:1450, 800:1400], vmin=-2, vmax=3, cmap="afmhot_r", aspect='auto', origin='lower')
plt.show()



