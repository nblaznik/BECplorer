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
import csv
import matplotlib
from scipy.optimize import curve_fit
from fitfunctions import gaussmod, tfmod, bimodalmod, tfaxialmod, gaussmod_OAH, tfmod_OAH, bimodalmod_OAH
from scipy.ndimage.interpolation import rotate
from scipy.special import zeta


# Import own fit functions
import OAH_functions as f1
from OAHDEV_functions import *


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

# Pretty Matplotlib Text
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


def OAH_full_process(date, shot, num, dz_focus, quad_cut, ly=0,
                     mask_w=0.51, multi_loop=0, peakswithfitting=False, 
                     deconvolution=False, px_fac=1, scaling_factor=1.5, regularization=1e-3):
    """ 
    Function that processes the raw data, and outputs processed OAH image. Recently incorporated
    also the deconvolution argument. 
    
    """
    multi_loop  = 0
    el_x        = 10
    el_z        = 100
    edge_x      = 10
    edge_z      = 80
    e_w         = 0.1

    # ---------------------------------------------------------------------------------------------------
    # ------------------------------ OAH ENTIRE PROCESSING FUNCTION -------------------------------------
    # ---------------------------------------------------------------------------------------------------

    # ------------------------------------- OPEN FILES --------------------------------------------------
    atoms, flat, dark = openFiles(date, shot, num, multi_loop=multi_loop)

    # ----------------------------------- MASK THE EDGE -------------------------------------------------
    atoms      = f1.squaroid(atoms - dark, width=mask_w)
    flat       = f1.squaroid(flat - dark, width=mask_w)
#     atoms      = f1.ellipsoid(atoms - dark, width=mask_w, type="tukey")
#     flat       = f1.ellipsoid(flat - dark, width=mask_w, type="tukey")

    
    # ------------------------------------- TAKE FFT ----------------------------------------------------
    fft_atoms = np.fft.fft2(atoms)
    fft_flat = np.fft.fft2(flat)

    # ----------------------------------- TRIM THE EDGE -------------------------------------------------
    fft_atoms = fft_atoms#[5:-5, 5:-5]
    fft_flat = fft_flat#[5:-5, 5:-5]
        # ----------------------------- DECONVOLUTION TO CORRECT PIXEL SHAPE --------------------------------
    P_k = np.ones(fft_atoms.shape)
    
    if deconvolution:
        # Define the pixel width
        pixelsize =  6.5E-6
        fft_kx = np.fft.fftfreq(fft_atoms.shape[0], d=pixelsize)  # Discrete FFT Sample Frequency in x
        fft_ky = np.fft.fftfreq(fft_atoms.shape[1], d=pixelsize)
        # Compute the Fourier Transform of the pixel
        P_k = np.sinc(fft_kx * pixelsize * 2)[:, None] * np.sinc(fft_ky * pixelsize * 2)[None, :]
#         P_k *= px_fac
#         P_k = fft.fftshift(P_k)       
        
        fft_atoms /= P_k
        fft_flat /= P_k

    # ---------------------------- CUT QUADRANT AND MAKE ELLIPSOID  -------------------------------------
    quad1, q1peak  = f1.box_cutter_pad_ellips(fft_atoms, quad_cut, el_x, el_z, 
                                              edge_x=edge_x, edge_z=edge_z, e_w=e_w, 
                                              peakswithfitting=peakswithfitting)
    flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat,  quad_cut, indices=q1peak) # same indices as for quad
    quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

    # ---------------------------------- SHIFT FFT TO CENTER  -------------------------------------------
    fft1 = np.fft.fftshift(quad1cut)
    flatfft1 = np.fft.fftshift(flatq1cut)
    
#     fft1 = quad1cut
#     flatfft1 = flatq1cut

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
    
    
    return atoms, flat, ang1, amp, P_k, fft_atoms, fft_flat 

def binImage(pic, xbin, zbin):
    """ A function to bin the pic file based on the bin parameters. """
    # If pic file not a multiple of bin, cut from the edge so it is.
    if pic.shape[0] % xbin != 0:
        pic = pic[:-(pic.shape[0] % xbin), :]
    if pic.shape[1] % zbin != 0:
        pic = pic[:, :-(pic.shape[1] % zbin)]
    pic = pic.reshape(pic.shape[0] // xbin, xbin, pic.shape[1] // zbin, zbin).mean(axis=3).mean(axis=1)
    return pic


def getPhysical(all_fits):
    phys_vars = []
    for shot_info in all_fits:
        xbin = 4
        zbin = 4
        pixelsize = 2.47148288973384e-06 
        kB = 1.38064852E-23
        m = 3.81923979E-26
        hb = 1.0545718E-34
        asc = 2.802642E-9
        mu0 = 1e-50
        e0 = 8.854187E-12
        fx = 115
        fz = 15
        wavelength = 589e-9
        detuning = 0
        prefactor = float((1 + 4 * detuning ** 2) * 2 * np.pi / (3 * (wavelength ** 2)) * 18. / 5.)

        par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'tfamp', 'tfxw', 'tfyw', 'gamp', 'gxw', 'gyw']
        bin_scaling = np.array([1., 1., 1., xbin, zbin, 1., xbin, zbin, 1., xbin, zbin])
        rng_offset = np.array([0., 0., 0., xmin, zmin, 0., 0., 0., 0., 0., 0.])
        to_physical = np.array([1., 1., 1., pixelsize, pixelsize, prefactor, pixelsize, pixelsize, prefactor, pixelsize, pixelsize])

        # Converts the fit results to absolute pixel values in the unbinned image.

        fit_results = shot_info * bin_scaling + rng_offset
        phys_results = fit_results * to_physical

        tof = 0
        ntherm = 0
        ntf = 0
        tx = 0
        tz = 0
        mux = 0
        muz = 0
        mun = 0

        ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]
        ntherm = 2 * np.pi * phys_results[8] * phys_results[9] * phys_results[10] 
        tx = 1 / kB * m / 1 * (fx * np.pi * 2 * phys_results[9]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        tz = 1 / kB * m / 1 * (fz * np.pi * 2 * phys_results[10]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mux = m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        muz = m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mun = 1.47708846953 * np.power(
            ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
            2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)

        phys_vars.append([ntf, ntherm, tx, tz, mux, muz, mun])
    return phys_vars 
        


for shot in range(103, 123):
    date        = 20241218
    # shot        = 55
    dz_focus    = 0.0024
    # Fitting

    ### FULL LOOP
    itter = 0
    all_fits_1 = []
    all_fits_2 = []


    initial_fit_vals = {
        "offset": 0.,
        "amp_ov": 1.,
        "ang": 0,
        "center": (1070//4, 930//4),
        "tfa": 1,
        "tfw": (5, 50),
        "ga": 1,
        "gw": (100, 10),
        "j_guess": 40 ,
        "axamp": 1,
        "x_shift": 0.0,
        "squeeze_par": 1.0,
    }

    save_folder = f"/home/bec_lab/Desktop/test_folder/" #imgs/SOAH/OAH_missmatch/{date}_{shot}_MT_Full/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for num in range(1):
        ang1 = HI_refocus(date, shot, num, 0.000, quad="quad1")
        ang2 = HI_refocus(date, shot, num, 0.000, quad="quad2")
        ang1 = normalize(binImage(ang1[:, 550:], 4, 4))[0]
        ang2 = normalize(binImage(ang2[:, 550:], 4, 4))[0]
    #     ang1 = normalize(ang1[:, 750:])[0]
    #     ang2 = normalize(ang2[:, 750:])[0]

        # Initial values
#         if itter > 1:  # should if if a previous fit was successful, somehow? 
#             initial_fit_vals['offset'] = fit1[-1].beta[0]
#             initial_fit_vals['amp_ov'] = fit1[-1].beta[1]
#             initial_fit_vals['ang'] = fit1[-1].beta[2]
#             initial_fit_vals['center'] = (fit1[-1].beta[3], fit1[-1].beta[4]) 
#             initial_fit_vals['tfa'] = fit1[-1].beta[5]
#             initial_fit_vals['tfw'] = (fit1[-1].beta[6], fit1[-1].beta[7])
#             initial_fit_vals['ga'] = fit1[-1].beta[8]
#             initial_fit_vals['gw'] = (fit1[-1].beta[9], fit1[-1].beta[10])

        if itter > 1:  # should if if a previous fit was successful, somehow? 
            initial_fit_vals['offset'] = fit1[-1].beta[0]
            initial_fit_vals['amp_ov'] = fit1[-1].beta[1]
            initial_fit_vals['ang'] = fit1[-1].beta[2]
            initial_fit_vals['center'] = (fit1[-1].beta[3], fit1[-1].beta[4]) 
#             initial_fit_vals['tfa'] = fit1[-1].beta[5]
#             initial_fit_vals['tfw'] = (fit1[-1].beta[6], fit1[-1].beta[7])
            initial_fit_vals['ga'] = fit1[-1].beta[5]
            initial_fit_vals['gw'] = (fit1[-1].beta[6], fit1[-1].beta[7])

        fit1 = fitting(ang1, "gauss", init_guess=initial_fit_vals, OAH=True)
        fit2 = fitting(ang2, "gauss", init_guess=initial_fit_vals, OAH=True)

        fig, [[ax1, ax2], [ax1n, ax2n], [ax3, ax4]] = plt.subplots(3, 2, figsize=(14, 7), sharex=True, sharey=True)
        ax1.imshow(ang1, vmin=-1.2, vmax=1.2, cmap="afmhot_r", aspect='auto', origin='lower')
        ax2.imshow(ang2, vmin=-1.2, vmax=1.2, cmap="afmhot_r", aspect='auto', origin='lower')
        ax1n.imshow(fit1[2], vmin=-1.2, vmax=1.2, cmap="afmhot_r", aspect='auto', origin='lower')
        ax2n.imshow(fit2[2], vmin=-1.2, vmax=1.2, cmap="afmhot_r", aspect='auto', origin='lower')
        ax3.imshow(fit1[1], vmin=-1.2, vmax=1.2, cmap="afmhot_r", aspect='auto', origin='lower')
        ax4.imshow(fit2[1], vmin=-1.2, vmax=1.2, cmap="afmhot_r", aspect='auto', origin='lower')

    #     ax1.axhline(y=fit1[-1].beta[3], c='C0')
    #     ax2.axhline(y=fit1[-1].beta[3], c='C1', ls = '--')

    #     ax2.axhline(y=fit2[-1].beta[3], c='C0')
    #     ax1.axhline(y=fit2[-1].beta[3], c='C1', ls = '--')

        ax1.set_ylabel("Data")
        ax1n.set_ylabel("Guess")
        ax3.set_ylabel("Fit")

        for ax in fig.axes:
            ax.set_yticks([])
            ax.set_xticks([])

        plt.suptitle(f"{date}-{shot}-{num}: \ntime = {round(num*0.5, 1)} s")

        all_fits_1.append(fit1[-1].beta)
        all_fits_2.append(fit2[-1].beta)

        plt.tight_layout()
        plt.savefig(f"{save_folder}{shot}_{str(num).zfill(4)}.png")
        plt.close()
        itter += 1

    np.save(f"{save_folder}vars_fits_1", all_fits_1)
    np.save(f"{save_folder}vars_fits_2", all_fits_2)