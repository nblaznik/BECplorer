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


def select_coords(pic, title_string=None):
    """
        A function to make a range selection a piece of cake. Plots the image (which is taken as an input).
        it is actually still a pain in the ass. It works in principle, but it's very annoying to implement it.
    """

    def select_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        global xtemp1, xtemp2, ytemp1, ytemp2
        xtemp1, ytemp1 = eclick.xdata, eclick.ydata
        xtemp2, ytemp2 = erelease.xdata, erelease.ydata
        title.set_text("Coordinates Recorded: \n X = {:} - {:}, Y = {:} - {:} \nPress Enter to save.".format(round(xtemp1), round(xtemp2), round(ytemp1), round(ytemp2)))
        fig.canvas.draw_idle()

    def toggle_selector(event):
        print('Key pressed.')
        if event.key == 't':
            for selector in selectors:
                name = type(selector).__name__
                if selector.active:
                    print(f'{name} deactivated.')
                    selector.set_active(False)
                else:
                    print(f'{name} activated.')
                    selector.set_active(True)
        elif event.key == 'enter':
            # print("Enter pressed")
            plt.close()
            t_var = True

    t_var = False
    fig, ax = plt.subplots()
    ax.imshow(pic)

    N = 100000  # If N is large one can see improvement by using blitting.
    x = np.linspace(0, 10, N)

    selectors = []
    for selector_class in [RectangleSelector]:
        selectors.append(selector_class(
            ax, select_callback,
            useblit=True,
            button=[1, 3],  # disable middle button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True))
        fig.canvas.mpl_connect('key_press_event', toggle_selector)

    if title_string is None:
        title = ax.set_title("Make a selection and \n press Enter to save the coordinates.")
    else:
        title = ax.set_title("{:} and \n press Enter to save the coordinates.".format(title_string))

    fig.canvas.draw_idle()

    return xtemp1, xtemp2, ytemp1, ytemp2


def wrapper(pic):
    xtemp1, xtemp2, ytemp1, ytemp2 = select_coords(pic)
    return xtemp1, xtemp2, ytemp1, ytemp2


def fitting(pic, mode, init_guess=initial_fit_vals, normalize=False, invert=False, OAH=False):
    # Initial values
    offset = init_guess["offset"]
    amp_ov = init_guess["amp_ov"]
    ang = init_guess["ang"]
    center = init_guess["center"]
    tfa = init_guess["tfa"]
    tfw = init_guess["tfw"]
    ga = init_guess["ga"]
    gw = init_guess["gw"]
    j_guess = init_guess["j_guess"]
    axamp = init_guess["axamp"]
    x_shift = init_guess["x_shift"]
    squeeze_par = init_guess["squeeze_par"]

    #     print(init_guess)

    # Normalize everything between 1 and 0
    if normalize:
        #         pic = pic[10:-10, 10:-10]
        pic = pic + abs(pic.min()) + 0.00001
        avg_bg = pic.mean()
        pic = pic / avg_bg

    if invert:
        # Normalize such that the background is at 1, the peak is at zero.
        pic = - pic + 0.5

    mask = pic == 0
    pic = np.ma.array(pic, mask=mask)
    # Generate empty arrays of the pic size which we will feed into the fitting procedure.
    # Create a 'fitvars' array of x and y coordinates
    x = np.arange(pic.shape[0])
    y = np.arange(pic.shape[1])
    xv, yv = np.meshgrid(x, y, indexing='ij')
    fitvars = np.array([xv, yv]).reshape(2, -1)

    xbin = 1
    zbin = 1

    # Constants
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

    xmin, xmax, zmin, zmax = 0, 0, 0, 0

    par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'tfamp', 'tfxw', 'tfyw', 'gamp', 'gxw', 'gyw']
    bin_scaling = np.array([1., 1., 1., xbin, zbin, 1., xbin, zbin, 1., xbin, zbin])
    rng_offset = np.array([0., 0., 0., xmin, zmin, 0., 0., 0., 0., 0., 0.])
    init_guess = np.array([offset, amp_ov, ang, center[0], center[1], tfa, tfw[0], tfw[1], ga, gw[0], gw[1]])
    to_physical = np.array([1., 1., 1., pixelsize, pixelsize, prefactor, pixelsize, pixelsize, prefactor, pixelsize, pixelsize])
    corr_guess = (init_guess - rng_offset) / bin_scaling

    if OAH:
        if mode == "gauss":
            corr_guess = np.append(corr_guess[:5], corr_guess[-3:])
            bin_scaling = np.append(bin_scaling[:5], bin_scaling[-3:])
            rng_offset = np.append(rng_offset[:5], rng_offset[-3:])
            par_names = np.append(par_names[:5], par_names[-3:])
            to_physical = np.append(to_physical[:5], to_physical[-3:])
            odrmodel = odr.Model(gaussmod_OAH)  # Store information for the gaussian fitting model

        # if mode == "doubletf":
        #     corr_guess = np.append(corr_guess[:8], [center2[0], center2[1], tfa2, tfw2[0], tfw2[1]])
        #     bin_scaling = np.append(bin_scaling[:8], bin_scaling[3:8])
        #     rng_offset = np.append(rng_offset[:8], rng_offset[3:8])
        #     par_names = np.append(par_names[:8], par_names[3:8])
        #     to_physical = np.append(to_physical[:8], to_physical[3:8])
        #     odrmodel = odr.Model(double_tf_OAH)  # Store information for the gaussian fitting model

        if mode == "tf":
            corr_guess = corr_guess[:8]
            bin_scaling = bin_scaling[:8]
            rng_offset = rng_offset[:8]
            par_names = par_names[:8]
            to_physical = to_physical[:8]
            odrmodel = odr.Model(tfmod_OAH)  # Store information for the tf fitting model

        if mode == "bimodal":
            odrmodel = odr.Model(bimodalmod_OAH)  # Store information for the bimodal fitting model


    else:
        if mode == "gauss":
            corr_guess = np.append(corr_guess[:5], corr_guess[-3:])
            bin_scaling = np.append(bin_scaling[:5], bin_scaling[-3:])
            rng_offset = np.append(rng_offset[:5], rng_offset[-3:])
            par_names = np.append(par_names[:5], par_names[-3:])
            to_physical = np.append(to_physical[:5], to_physical[-3:])
            odrmodel = odr.Model(gaussmod)  # Store information for the gaussian fitting model

        if mode == "tf":
            corr_guess = corr_guess[:8]
            bin_scaling = bin_scaling[:8]
            rng_offset = rng_offset[:8]
            par_names = par_names[:8]
            to_physical = to_physical[:8]
            odrmodel = odr.Model(tfmod)  # Store information for the tf fitting model

        if mode == "bimodal":
            odrmodel = odr.Model(bimodalmod)  # Store information for the bimodal fitting model

        if mode == "timecrystal":
            corr_guess = corr_guess[:8]
            corr_guess = np.append(corr_guess, j_guess)
            corr_guess = np.append(corr_guess, axamp)
            corr_guess = np.append(corr_guess, x_shift)
            corr_guess = np.append(corr_guess, squeeze_par)
            bin_scaling = bin_scaling[:8]
            bin_scaling = np.append(bin_scaling, 1.)
            bin_scaling = np.append(bin_scaling, 1.)
            bin_scaling = np.append(bin_scaling, 1.)
            bin_scaling = np.append(bin_scaling, 1.)
            rng_offset = rng_offset[:8]
            rng_offset = np.append(rng_offset, 0)
            rng_offset = np.append(rng_offset, 0)
            rng_offset = np.append(rng_offset, 0)
            rng_offset = np.append(rng_offset, 0)
            par_names = par_names[:8]
            par_names = np.append(par_names, "j parameter")
            par_names = np.append(par_names, "axamp")
            par_names = np.append(par_names, "x_shift")
            par_names = np.append(par_names, "squeeze_par")
            to_physical = to_physical[:8]
            to_physical = np.append(to_physical, 1.)
            to_physical = np.append(to_physical, 1.)
            to_physical = np.append(to_physical, 1.)
            to_physical = np.append(to_physical, 1.)
            odrmodel = odr.Model(tfaxialmod)

    print(corr_guess)
    # Run the ODR Fit procedure.
    odrdata = odr.Data(fitvars[:, ~pic.mask.flatten()], pic.flatten()[~pic.mask.flatten()])
    odrobj = odr.ODR(odrdata, odrmodel, beta0=corr_guess)
    odrobj.set_job(2)  # Ordinary least-sqaures fitting
    odrout = odrobj.run()
    #     odrout.pprint()

    if OAH:
        # This sets the angle to be correct wrt x and y.
        if np.abs(odrout.beta[2] % np.pi) > np.pi / 4.:
            odrout.beta[2] = odrout.beta[2] - np.pi / 2.
            print("Performing xy swap due to angle.")
            if mode == "bimodal":
                tmp_tfx = odrout.beta[6]
                tmp_gx = odrout.beta[-2]
                odrout.beta[6] = odrout.beta[7]
                odrout.beta[7] = tmp_tfx
                odrout.beta[-2] = odrout.beta[-1]
                odrout.beta[-1] = tmp_gx
            else:
                tmp = odrout.beta[6]
                odrout.beta[6] = odrout.beta[7]
                odrout.beta[7] = tmp

        if mode == "gauss":
            fitresult = gaussmod_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = gaussmod_OAH(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

        # if mode == "doubletf":
        #     fitresult = double_tf_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
        #     fitguess = double_tf_OAH(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
        #     fitresultgauss = []
        #     fitresulttf = []

        if mode == "tf":
            fitresult = tfmod_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = tfmod_OAH(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

        if mode == "bimodal":
            fitresult = bimodalmod_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresulttf = tfmod_OAH(odrout.beta[:8], fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = gaussmod_OAH(np.append(odrout.beta[:5], odrout.beta[-3:]), fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = bimodalmod_OAH(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])

    else:
        # This sets the angle to be correct wrt x and y.
        if np.abs(odrout.beta[2] % np.pi) > np.pi / 4.:
            odrout.beta[2] = odrout.beta[2] - np.pi / 2.
            print("Performing xy swap due to angle.")
            if mode == "bimodal":
                tmp_tfx = odrout.beta[6]
                tmp_gx = odrout.beta[-2]
                odrout.beta[6] = odrout.beta[7]
                odrout.beta[7] = tmp_tfx
                odrout.beta[-2] = odrout.beta[-1]
                odrout.beta[-1] = tmp_gx
            else:
                tmp = odrout.beta[6]
                odrout.beta[6] = odrout.beta[7]
                odrout.beta[7] = tmp

        if mode == "gauss":
            fitresult = gaussmod(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = gaussmod(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

        if mode == "tf":
            fitresult = tfmod(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = tfmod(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

        if mode == "bimodal":
            fitresult = bimodalmod(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresulttf = tfmod(odrout.beta[:8], fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = gaussmod(np.append(odrout.beta[:5], odrout.beta[-3:]), fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = bimodalmod(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])

        if mode == "timecrystal":
            fitresult = tfaxialmod(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = tfaxialmod(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

    #     print("The shape of the pic file: \n{:}\n".format(pic.shape))
    #     print("The guess parameters: \n{:}\n".format(corr_guess))
    #     print("Odrout: \n{:}\n".format(odrout.beta))

    # As the entire output, except for the angle and the offset, has to be positive,
    # we take the absolute value of the entire list, then put the angle and offset back in.
    offset_temp = odrout.beta[0]
    ang_temp = odrout.beta[2]
    odrout.beta = np.abs(odrout.beta)
    odrout.beta[0] = offset_temp
    odrout.beta[2] = ang_temp % np.pi

    # Converts the fit results to absolute pixel values in the unbinned image.
    fit_results = odrout.beta * bin_scaling + rng_offset
    phys_results = fit_results * to_physical

    tof = 0
    ncount = - np.log(pic.flatten()).sum() * prefactor * pixelsize ** 2 * xbin * zbin
    ntherm = 0
    ntf = 0
    tx = 0
    tz = 0
    mux = 0
    muz = 0
    mun = 0

    if mode == "gauss":
        ntherm = 2 * np.pi * phys_results[5] * phys_results[6] * phys_results[7]
        tx = 1 / kB * m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        tz = 1 / kB * m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mux = m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        muz = m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mun = 1.47708846953 * np.power(
            ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
            2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)
    if mode == "tf":
        ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]  # 2/5 = 8/15 / (4/3)
        tx = 1 / kB * m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        tz = 1 / kB * m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mux = m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        muz = m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mun = 1.47708846953 * np.power(
            ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
            2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)
    if mode == "timecrystal":
        ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]  # 2/5 = 8/15 / (4/3)
        tx = 1 / kB * m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        tz = 1 / kB * m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mux = m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        muz = m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mun = 1.47708846953 * np.power(
            ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
            2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)
    if mode == "bimodal":
        ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]
        ntherm = 2 * np.pi * phys_results[8] * phys_results[9] * phys_results[10]
        tx = 1 / kB * m / 1 * (fx * np.pi * 2 * phys_results[9]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        tz = 1 / kB * m / 1 * (fz * np.pi * 2 * phys_results[10]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mux = m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        muz = m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
        mun = 1.47708846953 * np.power(
            ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
            2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)

    ntotal = ntherm + ntf

    fitted_vars = (ncount, ntherm, ntf, tx, tz, mux, muz, mun)

    return pic, fitresult, fitguess, fitresultgauss, fitresulttf, fitted_vars, odrout


def normalize(ang):
    """
    Returns the normalized image of the cloud - it fits each horizontal line of the image with a simple linear fit.
    Then it performs the vertical linear fit. This way, we get the background image, which we subtract from the actual image.
    It seems however that we might even have some quadratic, or maybe a gaussian would be easier?
    """

    def linear_f(x, a, b):
        """ Faster, but does poor job. """
        return a * x + b

    def quadratic_f(x, a, b, c):
        """ Slower, but better equiped for normalizing"""
        return a * x ** 2 + b * x + c

    xdata = np.arange(0, len(ang[0]))
    ydata = np.arange(0, len(ang))
    arr_gradient = []
    arr_gradient_2 = []
    # Linear fit - horizontal gradient (for each line individualy)
    for i in range(ang.shape[0]):
        pars, cov = curve_fit(f=quadratic_f, xdata=xdata, ydata=ang[i], p0=[0, 0, 0])
        arr_line = quadratic_f(xdata, pars[0], pars[1], pars[2])
        arr_gradient.append(arr_line)
    lingrad = arr_gradient.copy()

    # Linear fit - vertical gradient (for each line individualy)
    arr_gradient = np.transpose(arr_gradient)  # Transpose
    for i in range(ang.shape[1]):
        pars, cov = curve_fit(f=quadratic_f, xdata=ydata, ydata=arr_gradient[i], p0=[0, 0, 0])
        arr_line = quadratic_f(ydata, pars[0], pars[1], pars[2])
        arr_gradient_2.append(arr_line)
    arr_gradient = np.transpose(arr_gradient_2)  # Transpose back

    ang = ang - arr_gradient
    return ang, arr_gradient, lingrad


### Load and cut properly refocused data
date = 20231207
shots = range(185, 205)
dz_focus = 0.005
shot_time = 2  # time between shots in ms
cut = [1, -1, 1, -250]
n = 50

save_folder = f"/home/bec_lab/Desktop/imgs/SOAH/polarizations_contrast/{date}/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

## if want to load
ALL_DATA = np.load(save_folder + "all_data.npy", allow_pickle=True)
shot_time = 2

for shot, quads in ALL_DATA:
    if shot not in [202]:
        continue
    print(f"{shot=}")

    save_folder_fit = f"/home/bec_lab/Desktop/imgs/SOAH/polarizations_contrast/{date}/fitted_images/{shot}/"
    if not os.path.exists(save_folder_fit):
        "Making a folder"
        os.makedirs(save_folder_fit)

    fits_1 = []  # For quad1
    fits_2 = []  # For quad2

    initial_g_1 = {'offset': 0.0,
                   'amp_ov': 1.0,
                   'ang': 0,
                   'center': (50, 1150),
                   'tfa': 1,
                   'tfw': (20, 200),
                   'ga': 1,
                   'gw': (200, 200),
                   'j_guess': 40,
                   'axamp': 1,
                   'x_shift': 0.0,
                   'squeeze_par': 1.0}
    initial_g_2 = {'offset': 0.0,
                   'amp_ov': 1.0,
                   'ang': 0,
                   'center': (50, 1250),
                   'tfa': 1,
                   'tfw': (20, 200),
                   'ga': 1,
                   'gw': (20, 200),
                   'j_guess': 40,
                   'axamp': 1,
                   'x_shift': 0.0,
                   'squeeze_par': 1.0}

    for i in range(50):
        # Do some extra background subtraction.
        pic1 = normalize(quads[0][i])[0]
        pic2 = normalize(quads[1][i])[0]

        fit1 = fitting(pic1, "tf", init_guess=initial_g_1, OAH=True)
        fit2 = fitting(pic2, "tf", init_guess=initial_g_2, OAH=True)

        fig, ax = plt.subplots(3, 2, figsize=(8, 4))
        fig.suptitle(f"{shot}- t = {0 + i * shot_time} ms")
        ax[0][0].set_title("Quad 1")
        ax[0][1].set_title("Quad 2")
        ax[0][0].imshow(fit1[0], aspect='auto', cmap='afmhot_r', vmin=-1, vmax=1, origin='lower')
        ax[1][0].imshow(fit1[1], aspect='auto', cmap='afmhot_r', vmin=-1, vmax=1, origin='lower')
        ax[2][0].imshow(fit1[2], aspect='auto', cmap='afmhot_r', vmin=-1, vmax=1, origin='lower')
        ax[0][1].imshow(fit2[0], aspect='auto', cmap='afmhot_r', vmin=-1, vmax=1, origin='lower')
        ax[1][1].imshow(fit2[1], aspect='auto', cmap='afmhot_r', vmin=-1, vmax=1, origin='lower')
        ax[2][1].imshow(fit2[2], aspect='auto', cmap='afmhot_r', vmin=-1, vmax=1, origin='lower')
        for axs in ax:
            for a in axs:
                a.set_xticks([])
                a.set_yticks([])
        plt.tight_layout()
        plt.savefig(save_folder_fit + f"{str(i).zfill(3)}")
        plt.close()

        fits_1.append(fit1)
        fits_2.append(fit2)

        # Set initial guess to the last guess.
        initial_g_1 = {'offset': fit1[-1].beta[0],
                       'amp_ov': fit1[-1].beta[1],
                       'ang': fit1[-1].beta[2],
                       'center': (fit1[-1].beta[3], fit1[-1].beta[4]),
                       'tfa': fit1[-1].beta[5],
                       'tfw': (fit1[-1].beta[6], fit1[-1].beta[7]),
                       'ga': 1,
                       'gw': (200, 200),
                       'j_guess': 40,
                       'axamp': 1,
                       'x_shift': 0.0,
                       'squeeze_par': 1.0}

        initial_g_2 = {'offset': fit2[-1].beta[0],
                       'amp_ov': fit2[-1].beta[1],
                       'ang': fit2[-1].beta[2],
                       'center': (fit2[-1].beta[3], fit2[-1].beta[4]),
                       'tfa': fit2[-1].beta[5],
                       'tfw': (fit2[-1].beta[6], fit2[-1].beta[7]),
                       'ga': fit2[-1].beta[5],
                       'gw': (fit2[-1].beta[6], fit2[-1].beta[7]),
                       'j_guess': 40,
                       'axamp': 1,
                       'x_shift': 0.0,
                       'squeeze_par': 1.0}

        clear_output()

    np.save(save_folder_fit + "fitted_run.npy", np.asarray([fits_1, fits_2], dtype="object"))