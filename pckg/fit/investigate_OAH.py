# Script to study the phase of the atoms, normalised to the background. Taken and modified based on Jasper's old script
# for such analysis.

# The main function in this script takes .fits files taken through OAH, and prepares the files in a single .fits file,
# where the data has been manalysed. Later on, these files can be analyzed.

# Last updated in November 2021 by Nejc Blaznik

# ---------------------------------------------------- IMPORTS -------------------------------------------------------
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Slider
from matplotlib import colors

import math
# Import own fit functions
import OAH_functions as f1
# from ..data_analysis import fits_analysis as fa
from scipy.optimize import curve_fit

# --------------------------------------------------- CONSTANTS ------------------------------------------------------
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
border_x = 5
border_z = 50
# Cutout
# ((xmin, xmax), (zmin, zmax)) = ((400, -200), (400, -200))
# ((xmin, xmax), (zmin, zmax)) = ((1000, 1400), (1000, 1400))
((xmin, xmax), (zmin, zmax)) = ((1, -1), (1, -1))
# ((xmin, xmax), (zmin, zmax)) = ((350, -350), (250, -250))


# --------------------------------------------------
def normalize(ang):
    """
    Returns the normalized image of the cloud - it fits each horizontal line of the image with a simple linear fit.
    Then it performs the vertical linear fit. This way, we get the background image, which we subtract from the actual image.
    """
    def linear_f(x, a, b):
        return a * x + b

    xdata = np.arange(0, len(ang[0]))
    ydata = np.arange(0, len(ang))
    arr_gradient = []
    arr_gradient_2 = []
    # Linear fit - horizontal gradient (for each line individualy)
    for i in range(ang.shape[0]):
        pars, cov = curve_fit(f=linear_f, xdata=xdata, ydata=ang[i], p0=[0, 0])
        arr_line = linear_f(xdata, pars[0], pars[1])
        arr_gradient.append(arr_line)
    # Linear fit - vertical gradient (for each line individualy)
    arr_gradient = np.transpose(arr_gradient)  # Transpose
    for i in range(ang.shape[1]):
        pars, cov = curve_fit(f=linear_f, xdata=ydata, ydata=arr_gradient[i], p0=[0, 0])
        arr_line = linear_f(ydata, pars[0], pars[1])
        arr_gradient_2.append(arr_line)
    arr_gradient = np.transpose(arr_gradient_2)  # Transpose back

    ang = ang - arr_gradient
    return ang, arr_gradient


def openFiles(date, shot, num):
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    # Opening files
    atoms = pyfits.open(path + image + '0.fits')[0].data.astype(float)[num][xmin:xmax, zmin:zmax]
    flat = pyfits.open(path + image + '1.fits')[0].data.astype(float)[0][xmin:xmax, zmin:zmax]
    dark = pyfits.open(path + image + '2.fits')[0].data.astype(float).mean(axis=0)[xmin:xmax, zmin:zmax]
    return atoms, flat, dark


def cutSquaroid(atoms, flat, dark):
    atoms_copy = f1.squaroid(atoms - dark, width=0.9)
    flat_copy = f1.squaroid(flat - dark, width=0.9)
    atoms = f1.squaroid(atoms - dark, width=0.51)
    flat = f1.squaroid(flat - dark, width=0.51)
    return atoms_copy, flat_copy, atoms, flat


def takeFFT(atoms, flat, atoms_copy, flat_copy):
    # Take the FTT of the atoms
    fft_atoms = np.fft.fft2(atoms)
    fft_flat = np.fft.fft2(flat)
    fft_atoms_copy = np.fft.fft2(atoms_copy)
    fft_flat_copy = np.fft.fft2(flat_copy)

    # Get rid of the borders with zeroth peak
    fft_atoms = fft_atoms[5:-5, 5:-5]
    fft_flat = fft_flat[5:-5, 5:-5]
    fft_atoms_copy = fft_atoms_copy[5:-5, 5:-5]
    fft_flat_copy = fft_flat_copy[5:-5, 5:-5]
    return fft_atoms, fft_flat, fft_atoms_copy, fft_flat_copy



rawplot = True
darkedgeplot = True
fftplot = True
cutplots = True
logFFTplot = True

# Nice narrow one with multiple shots
# date = 20210923
# shot = 204
# num = 0
# dz_focus = -0.003

# # # Nice full frame one
# date = 20210324
# shot = 48
# num = 0
# dz_focus = -0.003

# Nice double interference
date = 20220128
shot = 173 # to 177
num = 0
dz_focus = -0.003

# # Nice double interference
# date = 20220502
# shot = 60
# num = 0
# dz_focus = -0.003

# Nice double interference
# date = 20220517
# shot = 39 # to 177
# num = 0
# dz_focus = -0.000

#
# date = 20220523
# shot = 30 # to 177
# num = 1
# dz_focus = -0.000

# ###### Intermediate little analysis - linecuts of different runs, just simple interference
# for i in range(10):
#     atoms, flat, dark = openFiles(20220428, 23, i)
#     plt.plot((flat[1000] - dark[1000]) + i*2000, alpha=0.5)
#     # plt.plot(atoms[1000] + i*2000, alpha=0.5)
#     # plt.plot((atoms[1000] - dark[1000]) / (flat[1000] - dark[1000]), alpha=0.1)
#
# plt.show()
# quit()




atoms, flat, dark = openFiles(date, shot, num)

if rawplot:
    # FIRST PLOTTING - raw files
    f, ax = plt.subplots(2, 2)
    ax[0][0].imshow(atoms)
    ax[0][1].imshow(flat)
    ax[1][0].imshow(dark)
    ax[1][1].imshow((atoms - dark) / (flat - dark))
    ax[0][0].set_title("Atoms")
    ax[0][1].set_title("Flat")
    ax[1][0].set_title("Dark")
    ax[1][1].set_title("Pic")
    plt.show()


atoms_copy, flat_copy, atoms, flat = cutSquaroid(atoms, flat, dark)


if darkedgeplot:
    # SECOND PLOTTING - show the dark edge applied
    f, ax = plt.subplots(2, 2)
    ax[0][0].imshow(atoms)
    ax[0][1].imshow(flat)
    ax[1][0].imshow(atoms_copy)
    ax[1][1].imshow(flat_copy)
    ax[0][0].set_title("Atoms with dark edge")
    ax[0][1].set_title("Flat with dark edge")
    ax[1][0].set_title("Atoms with more dark edge")
    ax[1][1].set_title("Flat with more dark edge")
    plt.show()


fft_atoms, fft_flat, fft_atoms_copy, fft_flat_copy = takeFFT(atoms, flat, atoms_copy, flat_copy)


if fftplot:
    # THIRD PLOTTING - Check the log-normalized fft's. We expect two peaks in the top-left and bottom-right quadrants
    f, ax = plt.subplots(2, 2)
    ax[0][0].imshow(abs(fft_atoms), norm=colors.LogNorm())
    ax[0][1].imshow(abs(fft_flat), norm=colors.LogNorm())
    ax[1][0].imshow(abs(fft_atoms_copy), norm=colors.LogNorm())
    ax[1][1].imshow(abs(fft_flat_copy.imag), norm=colors.LogNorm())
    ax[0][0].set_title("FFT atoms with dark edge (all log norm)")
    ax[0][1].set_title("FFT flat with dark edge")
    ax[1][0].set_title("FFT atoms more dark edge")
    ax[1][1].set_title("FFT flat more dark edge")
    plt.show()

if logFFTplot:
    # THIRD PLOTTING - Check the log-normalized fft's. We expect two peaks in the top-left and bottom-right quadrants
    f, ax = plt.subplots(1, 1)
    ax.imshow(abs(fft_atoms), cmap=plt.get_cmap('gist_heat'), norm=colors.LogNorm())
    ax.set_title("FFT atoms with dark edge (all log norm)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# We create the Tukey windows cutouts for the data and the background. Additional cuts in x,z direction can be
# passed, but this is optional; the default value is 0. Note that because you later cut everything in the
# same size, this may affect the rest too.

quad1, q1peak = f1.box_cutter_pad_ellips(fft_atoms, "quad1", 10, 100, edge_x=10, edge_z=80)
flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat, "quad1", indices=q1peak)

if cutplots:
    empty = np.full_like(quad1.real, 1)
    elipse = f1.ellipsoid(empty)
    boxelipse, i = f1.box_cutter_pad_ellips(empty, "quad1", 20, 200, edge_x=10, edge_z=80, indices=q1peak)
    box, i = f1.box_cutter(fft_atoms, "quad1", 20, 200, edge_x=10, edge_z=80, dx=0, dz=0)

    f, ax = plt.subplots(5)
    ax[0].imshow(elipse)
    ax[1].imshow(abs(boxelipse))
    ax[2].imshow(np.log(abs(fft_atoms)))
    ax[3].imshow(np.log(abs(box)))
    ax[4].imshow(np.log(abs(quad1)), vmin=np.log(abs(box)).min(), vmax=np.log(abs(box)).max())
    # plt.show()

# Cutting the quads in the same sizes.
quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

# ------------------------------------------------ FFT SHIFT ----------------------------------------------------
# Now we FFT shift the zero frequency to the center
fft1 = np.fft.fftshift(quad1cut)
flatfft1 = np.fft.fftshift(flatq1cut)

# ------------------------------------------------ REFOCUSING ---------------------------------------------------
fft_kx = np.fft.fftfreq(fft1.shape[1], d=pix_size)  # Discrete FFT Sample Frequency in x
fft_ky = np.fft.fftfreq(fft1.shape[0], d=pix_size)  # Discrete FFT Sample Frequency in z
fft_k2 = fft_kx[None, :] ** 2 + fft_ky[:, None] ** 2  # Discrete FFT Sample Frequency in main axes multiplied
ly = 0  # .5E6#-3E6#.                                           # Adjusting the fft_ky array
coma_y_arg = ly * fft_ky[:, None] * (3 * fft_k2 / k0 ** 2) / k0
lin_y = np.exp(-1j * coma_y_arg)

# Determine the focus factor and refocus
focus = np.exp(-1j * fft_k2 * dz_focus / (2 * k0))
fft1 = fft1 * focus * lin_y
flatfft1 = flatfft1 * focus * lin_y

# ------------------------------------- INVERSE FFT -------------------------------------------------
inv1 = np.fft.ifft2(fft1) / np.fft.ifft2(flatfft1)
inv1 = inv1[border_x:-border_x, border_z:-border_z]

# Get Phase
ang1 = -np.angle(inv1)
ang1 = f1.unwrapper(ang1)

# Get amp
amp = np.abs(inv1)
amp = np.log(amp**2)


# Normalize
normfactor = ang1.mean()  # [300:900, 300:900].mean()
ang1 = ang1 - normfactor
ang1 = normalize(ang1)[0] # Use the function above to normalize the image.

f_full, ax_full = plt.subplots(1, 2, sharex=True, sharey=True)
im = ax_full[0].imshow(ang1, cmap='afmhot', interpolation='none', origin="lower")#, vmin=0, vmax=1)
im2 = ax_full[1].imshow(amp, cmap='afmhot', interpolation='none', origin="lower")#, vmin=0, vmax=2)

fig2, ax2 = plt.subplots(2, 1, sharex=True)
m = np.where(ang1 == ang1.min())[0]
m= 1074
print(m)
ax2[0].imshow(ang1, cmap='afmhot', interpolation='none', origin="lower", aspect="auto")  #, vmin=0, vmax=1)
ax2[0].axhline(y=m, alpha=0.7, ls='--')
ax2[1].plot(ang1[int(m)])


plt.colorbar(im, ax=ax_full[0], orientation='horizontal', ticks=[0, 1], pad=0.01)
plt.colorbar(im2, ax=ax_full[1], orientation='horizontal', ticks=[0, 2], pad=0.01)
ax_full[0].set_title("Phase delay")
ax_full[1].set_title("Optical density")
for i in range(2):
    ax_full[i].set_xticks([])
    ax_full[i].set_yticks([])

f_full.tight_layout()

plt.show()



quit()




## Plot the difference between the unwrapped and non-unwrapped image
nounwrap = np.load("/home/bec_lab/Desktop/no_unwrap.npy")
unwrap = np.load("/home/bec_lab/Desktop/unwrap.npy")
diff = nounwrap - unwrap
fig2, ax2 = plt.subplots(2, 3, sharex=True)
m = 1074
ax2[0][0].imshow(nounwrap, cmap='afmhot', interpolation='none', origin="lower", aspect="auto")  #, vmin=0, vmax=1)
ax2[0][1].imshow(unwrap, cmap='afmhot', interpolation='none', origin="lower", aspect="auto")  #, vmin=0, vmax=1)
ax2[0][2].imshow(diff, cmap='afmhot', interpolation='none', origin="lower", aspect="auto")  #, vmin=0, vmax=1)
ax2[0][0].axhline(y=m, alpha=0.7, ls='--')
ax2[0][1].axhline(y=m, alpha=0.7, ls='--')
ax2[0][2].axhline(y=m, alpha=0.7, ls='--')
ax2[1][0].plot(nounwrap[int(m)])#, vmin=0, vmax=1)
ax2[1][1].plot(unwrap[int(m)])#, vmin=0, vmax=1)
ax2[1][2].plot(diff[int(m)])#, vmin=0, vmax=1)
plt.show()
