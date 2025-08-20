# ------------------------------------------------------------------------------------------------------------------
# SUPPLEMENTARY FUNCTIONS FOR OFF-AXIS HOLOGRAPHY IMAGING

# Original version by Sanne - modified by Jasper - edited, corrected and modified by Nejc
# Last checked in April 2021
# ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------- IMPORTS ----------------------------------------------------
import astropy.io.fits as pyfits
import numpy as np
import os
import pylab
import abel
import datetime
import glob
import csv
import sys, argparse
import matplotlib.pyplot as plt
import scipy.odr as odr
from scipy import ndimage
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
# import hartreefockfit as hff
from skimage.restoration import unwrap_phase

# --------------------------------------------------- CONSTANTS ------------------------------------------------------
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
pix_size = 6.5E-6 / 2.63


# -------------------------------------------------- PARAMETERS ------------------------------------------------------
# Frequencies
# MT
# fx = 91.9
# fz = 15.2
# Decompressed trap 2.5
fx = 72.6
fz = 15.1
# Decompressed trap 1.5
# fx = 44.6
# fz = 15.4
# --------------------------------------------------
# Light field properties
lamb0 = 589.1E-9  # Frequency
kvec = 2 * np.pi / lamb0  # k-vector
# --------------------------------------------------


# --------------------------------------------------- FUNCTIONS ------------------------------------------------------
# ARRAY MODIFICATIONS

def polarizability(det_1, ang, lamb0, linew=9.7):
    """
    The function for defining polarizability, based on detuning, angle and initial wavelength.
    Besides that, not really sure yet what this function does exactly.
    :param det_1: Detuning parameter
    :param ang: Angle parameter
    :param lamb0: The wavelength of the light
    :param linew: Linewidth
    :return: Polarizability
    """
    det_0 = det_1 + 15.8
    det_2 = det_1 - 34.4
    sigm_l = 3 * lamb0 ** 2 / (2 * np.pi)
    k0 = 2 * np.pi / lamb0
    D0 = 4. / 24 * np.sin(ang) ** 2
    D1 = 5. / 24 * (1 + np.cos(ang) ** 2)
    D2 = 6. / 24 + 1 / 24 * np.sin(ang) ** 2

    return 1j * sigm_l / k0 * (D0 / (1 - 2j * det_0 / linew) + D1 / (1 - 2j * det_1 / linew) + D2 / (1 - 2j * det_2 / linew))

def ellipsoid(arr, width=0.05, type="tukey", output_filter=False):
    """
    The ellipsoid filter.
    :param arr: The array to be filtered,
    :param width: the width of each attenuation area relative to the entire array.
    :param type: which type of the window to use. Options 'tukey', 'blackman', 'hanning', 'bartlett', 'hamming', 'kaiser'
    :param output_filter: Boolean, if output the filter or the filter array
    :return: Filtered array or just a filter (depends on the ouput_filter parameter)
    """
    xs = arr.shape[0]
    ys = arr.shape[1]

    if type == "tukey":
        x = 1. * np.arange(xs) - xs // 2
        y = 1. * np.arange(ys) - ys // 2
        filt = np.zeros(arr.shape, dtype=float)
        # In region 1: all 1
        rsq = np.sqrt((x[:, None] / (xs / 2)) ** 2 + (y[None, :] / (ys / 2)) ** 2)
        filt[rsq <= 1.] = np.cos(np.pi / 2. * (rsq[rsq <= 1.] - 1 + width) / width) ** 2
        filt[rsq <= 1 - width] = 1

    padx = xs//8
    pady = ys//8
    xs = xs - 2*padx
    ys = ys - 2*pady

    if type == "blackman":
        windowx = abs(np.blackman(xs)) #otherwise negative values near the edge
        windowy = abs(np.blackman(ys))
        filt = np.sqrt(np.outer(windowx, windowy))
        filt = np.pad(filt, ((padx, padx), (pady, pady)), mode='linear_ramp', end_values=((0, 0), (0, 0)))

    elif type == "hanning":
        windowx = np.hanning(xs)
        windowy = np.hanning(ys)
        filt = np.sqrt(np.outer(windowx, windowy))
        filt = np.pad(filt, ((padx, padx), (pady, pady)), mode='linear_ramp', end_values=((0, 0), (0, 0)))

    elif type == "bartlett":
        windowx = np.bartlett(xs)
        windowy = np.bartlett(ys)
        filt = np.sqrt(np.outer(windowx, windowy))
        filt = np.pad(filt, ((padx, padx), (pady, pady)), mode='linear_ramp', end_values=((0, 0), (0, 0)))

    elif type == "hamming":
        windowx = np.hamming(xs)
        windowy = np.hamming(ys)
        filt = np.sqrt(np.outer(windowx, windowy))
        filt = np.pad(filt, ((padx, padx), (pady, pady)), mode='linear_ramp', end_values=((0, 0), (0, 0)))

    elif type == "kaiser":
        windowx = np.kaiser(xs, width)
        windowy = np.kaiser(ys, width)
        filt = np.sqrt(np.outer(windowx, windowy))
        filt = np.pad(filt, ((padx, padx), (pady, pady)), mode='linear_ramp', end_values=((0, 0), (0, 0)))

    if output_filter == False:
        return filt * arr
    else:
        return filt

def squaroid(arr, width=0.01, output_filter=False):
    xs = arr.shape[0]
    ys = arr.shape[1]

    x = 1. * np.arange(xs) - xs // 2
    y = 1. * np.arange(ys) - ys // 2

    xw = int(width * xs)
    yw = int(width * ys)

    filt = np.zeros(arr.shape, dtype=float)
    filt1 = filt.copy()
    filt2 = filt.copy()

    rsq = np.sqrt((x[:, None] / (xs / 2)) ** 2) + 0. * y[None, :]

    filt1[rsq <= 1.] = np.cos(np.pi / 2. * (rsq[rsq <= 1.] - 1 + width) / width) ** 2
    filt1[rsq <= 1 - width] = 1
    # rsq1 = rsq
    rsq = np.sqrt((y[None, :] / (ys / 2)) ** 2) + 0. * x[:, None]
    filt2[rsq <= 1.] = np.cos(np.pi / 2. * (rsq[rsq <= 1.] - 1 + width) / width) ** 2
    filt2[rsq <= 1 - width] = 1

    filt = filt1 * filt2
    # filt[np.logical_and(rsq<=1.,rsq1<=1-width)]      = np.cos(np.pi/2. * (rsq[np.logical_and(rsq<=1.,rsq1<=1-width)]-1+width)/width )**2
    # filt[np.logical_and(rsq<=1-width,rsq1<=1-width)] = 1

    ## In the transition region (2), sin^2 decay to 0
    if output_filter == False:
        return filt * arr
    else:
        return filt

def fix_angle(arr, ind_a, ind_x, ind_y):
    angle = arr[ind_a]

    if np.abs(angle) > np.pi / 4.:
        arr[ind_a] = angle - np.pi / 2.
        tmp = arr[ind_x]
        arr[ind_x] = arr[ind_y]
        arr[ind_y] = tmp

    return arr


# UNWRAPPING
def unwrapper(data):
    # return np.unwrap(np.unwrap(np.unwrap(np.unwrap(data,axis=0),axis=1),axis=0),axis=1)
    # return np.unwrap(np.unwrap(np.unwrap(data,axis=0),axis=1),axis=0)
    # return np.unwrap(np.unwrap(data,axis=1),axis=0)
    # return np.unwrap(data,axis=1)
    return unwrap_phase(data)  # ,wrap_around=(True, False))


# FIT FUNCTIONS
# This   the output parameters and the fit result.
# Use as: output, fitresult = fitfunc(data, model, guess parameters)

def fitfunc(data, model, guess, parlist):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    xv, zv = gridmaker(data)

    fitvars = np.array([xv, zv]).reshape(2, -1)

    fitmod = odr.Model(model, extra_args=(pix_x, pix_z, polariz))
    myData = odr.Data(fitvars, data.flatten())
    myODR = odr.ODR(myData, fitmod, beta0=guess)
    myODR.set_job(fit_type=2)
    output = myODR.run()
    output.pprint()

    ### Set sizes to positive
    for i in [1, 3, 4, 5, 6, 7, 8, 9, 10]:
        output.beta[i] = np.abs(output.beta[i])

    ### Change angle
    ang_temp = output.beta[2]
    output.beta[2] = ang_temp % np.pi

    fitresult = model(output.beta, fitvars, *(pix_x, pix_z, polariz)).reshape(data.shape[0], data.shape[1])

    return output.beta, fitresult


def hffit_old(data, guess, mu, temp, parlist):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    ### 0 bg, 1 ang, 2 x cent, 3 z cent, 4 mu, 5 T, 6 omega x, 7 omega z
    physres = physresults(guess, parlist)
    hffguess = [physres[0], 0., physres[3], physres[4], mu, temp, 2. * np.pi * fx, 2. * np.pi * fz, 1.0]

    ### Grid
    hff_r, hff_z = gridmaker(data)

    ### Detract gradient and scale with polariz
    data2 = (data - guess[11] * hff_r - guess[12] * hff_z - guess[0]) / parlist[2]

    ### Scaling grid to meters
    hff_r = parlist[0] * hff_r
    hff_z = parlist[1] * hff_z
    hff_fitvars = np.array([hff_r, hff_z]).reshape(2, -1)

    ### fit!
    hff_model = hff.new_model_omgfit
    hff_fitmod = odr.Model(hff_model, extra_args=(fx, fz))

    hff_myData = odr.Data(hff_fitvars, data2.flatten())
    hff_myODR = odr.ODR(hff_myData, hff_fitmod, beta0=hffguess)

    hff_myODR.set_job(fit_type=2)
    hff_output = hff_myODR.run()

    hff_output.pprint()

    ### the fit function
    hff_tf, hff_th = hff.new_model_omgfit(hff_output.beta, hff_fitvars, for_plot=True)
    hff_tf = hff_tf.reshape(hff_r.shape[0], hff_z.shape[1])
    hff_th = hff_th.reshape(hff_r.shape[0], hff_z.shape[1])

    print(hff_output.beta[8])

    return data2, hff_output.beta, hff_tf, hff_th


def hffit_ll(data, guess, mu, temp, parlist, fit=True):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]
    det_1 = parlist[6]
    kvec = parlist[7]

    ### 0 bg, 1 ang, 2 x cent, 3 z cent, 4 mu, 5 T, 6 omega x, 7 omega z
    physres = physresults(guess, parlist)
    hffguess = [physres[0], np.pi / 2., physres[3], physres[4], mu, temp, 2. * np.pi * fx, 2. * np.pi * fz, 1.0]

    ### Grid
    hff_r, hff_z = gridmaker(data)

    ### Detract gradient and scale with polariz
    data2 = (data - guess[11] * hff_r - guess[12] * hff_z - guess[0])

    ### Scaling grid to meters
    hff_r = parlist[0] * hff_r
    hff_z = parlist[1] * hff_z
    hff_fitvars = np.array([hff_r, hff_z]).reshape(2, -1)

    ### fit!
    if fit:
        hff_model = hff.model_ll_correct
        hff_fitmod = odr.Model(hff_model, extra_args=(fx, fz, det_1, kvec))

        hff_myData = odr.Data(hff_fitvars, data2.flatten())
        hff_myODR = odr.ODR(hff_myData, hff_fitmod, beta0=hffguess)

        hff_myODR.set_job(fit_type=2)
        hff_output = hff_myODR.run()

        hff_fitres = hff_output.beta

        hff_output.pprint()
    else:
        hff_fitres = hffguess

    ### the fit function
    hff_tf, hff_th = hff.model_ll_correct(hff_fitres, hff_fitvars, fx, fz, det_1, kvec, for_plot=True)
    hff_tf = hff_tf.reshape(hff_r.shape[0], hff_z.shape[1])
    hff_th = hff_th.reshape(hff_r.shape[0], hff_z.shape[1])

    return data2, hff_fitres, hff_tf, hff_th


def popovfit(data, guess, mu, temp, parlist, fit=True, return_grids=False):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    ### 0 bg, 1 ang, 2 x cent, 3 z cent, 4 mu, 5 T, 6 omega x, 7 omega z
    physres = physresults(guess, parlist)
    hffguess = [physres[0], np.pi / 2., physres[3], physres[4], mu, temp]  # , 2.*np.pi*fx, 2.*np.pi*fz,1.0]

    #	print(physres)

    ### Grid
    hff_r, hff_z = gridmaker(data)

    ### Detract gradient and scale with polariz
    data2 = (data - guess[11] * hff_r - guess[12] * hff_z - guess[0]) / parlist[2]

    ### Scaling grid to meters
    hff_r = parlist[0] * hff_r
    hff_z = parlist[1] * hff_z
    hff_fitvars = np.array([hff_r, hff_z]).reshape(2, -1)

    ### fit!
    if fit:
        hff_model = hff.model_popov
        hff_fitmod = odr.Model(hff_model, extra_args=(fx, fz))

        hff_myData = odr.Data(hff_fitvars, data2.flatten())
        hff_myODR = odr.ODR(hff_myData, hff_fitmod, beta0=hffguess)

        hff_myODR.set_job(fit_type=2)
        hff_output = hff_myODR.run()

        hff_fitres = hff_output.beta

        hff_output.pprint()
    else:
        hff_fitres = hffguess

    ### the fit function
    hff_tf, hff_th = hff.model_popov(hff_fitres, hff_fitvars, fx, fz, for_plot=True)
    hff_tf = hff_tf.reshape(hff_r.shape[0], hff_z.shape[1])
    hff_th = hff_th.reshape(hff_r.shape[0], hff_z.shape[1])

    if return_grids:
        return data2, hff_fitres, hff_tf, hff_th, hff_r, hff_z

    return data2, hff_fitres, hff_tf, hff_th


def hffit(data, guess, mu, temp, parlist, fit=True):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    ### 0 bg, 1 ang, 2 x cent, 3 z cent, 4 mu, 5 T, 6 omega x, 7 omega z
    physres = physresults(guess, parlist)
    hffguess = [physres[0], np.pi / 2., physres[3], physres[4], mu, temp, 2. * np.pi * fx, 2. * np.pi * fz, 1.0]

    ### Grid
    hff_r, hff_z = gridmaker(data)

    ### Detract gradient and scale with polariz
    data2 = (data - guess[11] * hff_r - guess[12] * hff_z - guess[0]) / parlist[2]

    ### Scaling grid to meters
    hff_r = parlist[0] * hff_r
    hff_z = parlist[1] * hff_z
    hff_fitvars = np.array([hff_r, hff_z]).reshape(2, -1)

    ### fit!
    if fit:
        hff_model = hff.new_model_omgfit
        hff_fitmod = odr.Model(hff_model, extra_args=(fx, fz))

        hff_myData = odr.Data(hff_fitvars, data2.flatten())
        hff_myODR = odr.ODR(hff_myData, hff_fitmod, beta0=hffguess)

        hff_myODR.set_job(fit_type=2)
        hff_output = hff_myODR.run()

        hff_fitres = hff_output.beta

        hff_output.pprint()
    else:
        hff_fitres = hffguess

    ### the fit function
    hff_tf, hff_th = hff.new_model_omgfit(hff_fitres, hff_fitvars, fx, fz, for_plot=True)
    hff_tf = hff_tf.reshape(hff_r.shape[0], hff_z.shape[1])
    hff_th = hff_th.reshape(hff_r.shape[0], hff_z.shape[1])

    return data2, hff_fitres, hff_tf, hff_th


##################### Fit Functions #####################
def gauss1D(x, amp, mu, sig):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def gauss2D(x, amp, mu, sig):
    return gauss1D(x[0], amp, mu[0], sig[0]) * gauss1D(x[1], 1., mu[1], sig[1])


def rotvars(rot, mid, x):
    x0 = np.cos(rot) * (x[0] - mid[0]) - np.sin(rot) * (x[1] - mid[1])
    x1 = np.cos(rot) * (x[1] - mid[1]) + np.sin(rot) * (x[0] - mid[0])
    return [x0, x1]


def thomasfermi(x, amp, r, mid):
    value = 1 - ((x[0] - mid[0]) / r[0]) ** 2 - ((x[1] - mid[1]) / r[1]) ** 2
    value[value < 0] = 0.
    return amp * np.power(value, 3. / 2.)


##############
# 0 bg, 1 amp, 2 angle, 3 x_cent, 4 z_cent,
# 5 tf_A, 6 tf_Rx,7 tf_Rz, 8 g_A, 9 gw_x, 10 gw_z,
# 11 x_grad, 12 y_grad
def gaussmod(B, x):
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[1] * np.exp(-gauss2D(xrot, B[8], [0., 0.], [B[9], B[10]]))


def tfmod(B, x):
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[1] * np.exp(-thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]))


def bimodalmod(B, x, *args):
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[1] * np.exp(
        -1. * (thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]) + gauss2D(xrot, B[8], [0., 0.], [B[9], B[10]]))) + B[
               11] * x[0] + B[12] * x[1] + B[0]


##############

def phasegaussmod(B, x):
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + gauss2D(xrot, B[8], [0., 0.], [B[9], B[10]]) * B[1]


def phasetfmod(B, x):
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]) * B[1]


def phasebimodalmod(B, x, *args):
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + B[1] * (
                thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]) + gauss2D(xrot, B[8], [0., 0.], [B[9], B[10]])) + B[
               11] * x[0] + B[12] * x[1]


##############

### 0 bg, 1 amp, 2 angle, 3 x_cent, 4 z_cent, 5 mu,
### 6 x_grad, 7 z_grad, 8 g_A, 9 gw_x, 10 gw_z
def TFProfile(B, x, pix_x, pix_z, polariz):
    ## TF profile in terms of mu
    xrot = rotvars(B[2], [B[3], B[4]], x)
    sx = np.sqrt(2 * B[5] / (m * (2 * np.pi * fx) ** 2)) / pix_x
    sy = sx * pix_x
    sz = np.sqrt(2 * B[5] / (m * (2 * np.pi * fz) ** 2)) / pix_z

    amp = (4 / 3) * (B[5] / mu0) * sy * polariz
    profile = (1 - xrot[0] ** 2 / sx ** 2 - xrot[1] ** 2 / sz ** 2)
    profile[profile < 0] = 0

    return B[0] + amp * np.power(profile, 3. / 2.) + B[11] * x[0] + B[12] * x[1] + B[1] * gauss2D(xrot, B[8], [0., 0.],
                                                                                                  [B[9], B[10]])


################################# Physical results #################################
### Input is output.beta, output is phys_results

def physresults(array, parlist):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    to_physical = np.array(
        [1., 1., 1., pix_x, pix_z, 1. / polariz, pix_x, pix_z, 1. / polariz, pix_x, pix_z, pix_x, pix_z])

    return array * to_physical


################################# N, T and mu #################################
### Input is output.beta, output is a list with values

def NTMu(output, parlist):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    phys_results = physresults(output, parlist)

    ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]
    ntherm = 2. * np.pi * phys_results[8] * phys_results[9] * phys_results[10]
    ntotal = ntherm + ntf

    tx = 1. / kB * m / 2. * (fx * np.pi * 2 * phys_results[9]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
    tz = 1. / kB * m / 2. * (fz * np.pi * 2 * phys_results[10]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)

    mux = m / 2. * (fx * np.pi * 2. * phys_results[6]) ** 2 / (1. + (tof * fx * np.pi * 2.) ** 2)
    muz = m / 2. * (fz * np.pi * 2. * phys_results[7]) ** 2 / (1. + (tof * fz * np.pi * 2.) ** 2)
    mun = 1.47708846953 * np.power(ntf * asc / (np.sqrt(hb / (m * np.power(8. * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
                                   2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)

    print("\n ntf = %E, ntherm = %E, ntotal = %E" % (ntf, ntherm, ntotal))
    print("tx = %E, tz = %E K" % (tx, tz))
    print("mux = %E, muz = %E, mun = %E Hz\n" % (
    mux / (2. * np.pi * hb), muz / (2. * np.pi * hb), mun / (2. * np.pi * hb)))

    return [ntf, ntherm, ntotal, tx, tz, mux, muz, mun]


### Then sorting this list.
def sortfunc(list1, list2, list3, list4):
    ntflist = [list1[0], list2[0], list3[0], list4[0]]
    nthlist = [list1[1], list2[1], list3[1], list4[1]]
    ntotlist = [list1[2], list2[2], list3[2], list4[2]]
    txlist = [list1[3], list2[3], list3[3], list4[3]]
    tzlist = [list1[4], list2[4], list3[4], list4[4]]
    muxlist = [list1[5], list2[5], list3[5], list4[5]]
    muzlist = [list1[6], list2[6], list3[6], list4[6]]
    munlist = [list1[7], list2[7], list3[7], list4[7]]

    ntflist = np.array(ntflist)
    nthlist = np.array(nthlist)
    ntotlist = np.array(ntotlist)
    txlist = np.array(txlist)
    tzlist = np.array(tzlist)
    muxlist = np.array(muxlist)
    muzlist = np.array(muzlist)
    munlist = np.array(munlist)

    return ntflist, nthlist, ntotlist, txlist, tzlist, muxlist, muzlist, munlist


######################## Create a grid #########################
def gridmaker(data):
    x = np.arange(data.shape[0])
    z = np.arange(data.shape[1])

    xv, zv = np.meshgrid(x, z, indexing='ij')

    return xv, zv


#################### Cutout around BEC #######################
def BECcutout(ang1, amp1, numx=None, numz=None, indices=None):
    '''gausang = ndimage.filters.gaussian_filter(ang1,4)

    if 0:
        gausang = gausang[300:700,10:500]
        ang1    = ang1[300:700,10:500]
        amp1    = amp1[300:700,10:500]

    indices1 = np.where(gausang==gausang.max())'''
    if indices == None:
        indices1 = BECcenter(ang1)
    else:
        indices1 = indices
    # print indices1

    numx = numx or 50
    numz = numz or 75

    ang2 = ang1[indices1[0] - numx:indices1[0] + numx, indices1[1] - numz:indices1[1] + numz]
    amp2 = amp1[indices1[0] - numx:indices1[0] + numx, indices1[1] - numz:indices1[1] + numz]

    if 0:
        figx, ((axx1, axx2), (axx3, axx4)) = plt.subplots(2, 2)

        axx1.imshow(gausang, cmap='bwr', interpolation='none', origin="lower")
        #	axx2.imshow(np.log(gausamp), cmap='bwr', interpolation='none',origin="lower")
        axx3.imshow(ang2, cmap='bwr', interpolation='none', origin="lower")
        axx4.imshow(np.log(np.abs(amp2)) * 2, cmap='bwr', interpolation='none', origin="lower")

        plt.show()

    return ang2, amp2


############################## Size Cutter ##############################
### Use this as list1_cut, list2_cut = sizecomp(list1,list2)
### Cuts two 2D arrays to the same size, by taking away from the edges.

def sizecomp(list1, list2):
    corr0 = (list1.shape[0] - list2.shape[0]) // 2
    corr1 = (list1.shape[1] - list2.shape[1]) // 2

    ### First we look at x
    if corr0 > 0:
        list1 = list1[corr0:list1.shape[0] - corr0, :]

    elif corr0 < 0:
        corr0 = np.abs(corr0)
        list2 = list2[corr0:list2.shape[0] - corr0, :]

    ### Now we look at z
    if corr1 > 0:
        list1 = list1[:, corr1:list1.shape[1] - corr1]

    elif corr1 < 0:
        corr1 = np.abs(corr1)
        list2 = list2[:, corr1:list2.shape[1] - corr1]

    # print(list1.shape, list2.shape)

    return list1, list2


############################## List Cutter ##############################
### Function to automatically cut 4 lists in the same sizes.
### Use: quad1cut, quad2cut, quad3cut, quad4cut = f1.listcut(quad1,quad2,quad3,quad4)
def listcut(list1, list2, list3, list4):
    # print "Original sizes:", list1.shape, list4.shape, list3.shape, list4.shape

    list1, list2 = sizecomp(list1, list2)
    list3, list4 = sizecomp(list3, list4)

    # print "First cut:", list1.shape, list2.shape, list3.shape, list4.shape

    list1, list3 = sizecomp(list1, list3)
    list2, list4 = sizecomp(list2, list4)

    # print "Final cut:", list1.shape, list1.shape, list1.shape, list1.shape

    return list1, list2, list3, list4


############################## Abel transform ##############################
### input: the image, its fit parameters, a choice of mu (mux, muz, ...),
### and the pixel sizes

def AbelT(ang, amp, outputlist, muchoice, parlist):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]
    fx = parlist[4]
    fz = parlist[5]

    angle = outputlist[2] * 180. / np.pi

    ang, amp = BECcutout(ang, amp)

    print("Angle = %f" % angle)

    ## rotating the data
    ang1 = ndimage.rotate(ang, angle, reshape=False)
    amp1 = ndimage.rotate(amp, angle, reshape=False)

    ### Blur for the fit
    gauss = ndimage.filters.gaussian_filter(ang1, 4)

    ### Making a fit to determine the new center coordinates
    extra_guess = [-1., 5., 0., 50., 450., 0., 100., 100., 0., 200., 200., 0., 0.]
    # extra_guess = outputlist
    # extra_guess[2] = 0

    extra_guess[3] = 50
    extra_guess[4] = 50

    output1pic, fit1pic = fitfunc(ang1, phasebimodalmod, extra_guess, parlist)
    # output1pic, fit1pic = fitfunc(gauss, phasebimodalmod, extra_guess, parlist)

    xcenter = int(output1pic[3])  # +1
    zcenter = int(output1pic[4])  # +0

    # print "\nCenter coords are %i, %i" %(xcenter, zcenter)

    ## creating the theoretical model:
    mu = muchoice

    wx = 2. * np.pi * fx
    wz = 2. * np.pi * fz
    R_rho2 = 2. * mu / (m * wx ** 2) / pix_x ** 2
    R_z2 = 2. * mu / (m * wz ** 2) / pix_z ** 2

    xv, zv = gridmaker(ang1)

    ## alternatively, use output2.beta[3] and [4] (note, the +-1 must match x,z center)
    TFP = mu / mu0 * (1 - (xv - xcenter) ** 2 / R_rho2 - (zv - zcenter) ** 2 / R_z2)
    TFP[TFP < 0] = 0
    # TFP = TFP + outputlist[1] * gauss2D([xv,zv],outputlist[8],[0., 0.],[outputlist[9], outputlist[10]])

    ## centering the condensate for the abel transform
    right = 50
    left = 30
    ang2 = ang1[xcenter - left:xcenter + left, zcenter - right:zcenter + right]
    TFP = TFP[xcenter - left:xcenter + left, zcenter - right:zcenter + right]
    amp2 = amp1[xcenter - left:xcenter + left, zcenter - right:zcenter + right]

    # print "ang2: (%i,%i)" %(ang2.shape[0],ang2.shape[1])

    if 0:
        figx, ((axx1, axx2), (axx3, axx4)) = plt.subplots(2, 3)

        axx1.pcolormesh(zv, xv, ang1, cmap='bwr')  # , interpolation='none',origin="lower")
        axx1.contour(zv, xv, fit1pic, 4, colors='k')

        axx2.imshow(TFP, cmap='bwr', interpolation='none', origin="lower")
        axx2.set_title("Theory")

        axx3.imshow(ang2, cmap='bwr', interpolation='none', origin="lower")
        axx3.set_title("Phase")

        axx4.imshow(amp2, cmap='bwr', interpolation='none', origin="lower")
        axx4.set_title("Amplitude")

        plt.show()

    ## Abel transform on the data:
    abel_ang = abel.Transform(ang2.T, direction='inverse', method='basex').transform
    abel_amp = abel.Transform(amp2.T, direction='inverse', method='basex').transform

    ## model; you directly acces the refractive index, no abeltransform needed
    abel_th = TFP.T * polariz

    return abel_ang, abel_th, abel_amp


############################## My Box Cutter ##################################
### Three versions
### Input is an FFT file in which you want to select a +- 1 order term
### Use as quad, indices =  box_cutter(fft, r, x is optional, z is optional, indices is optional)

### This box_cutter will pad to match the size of the original array, it is otherwise identical to box_cutter.
def box_cutter_pad(fft, r, x=None, z=None, indices=None):
    shape = fft.shape
    output = np.zeros(fft.shape, dtype=np.complex128)

    tmp, indices = box_cutter(fft, r, x, z, indices)

    output[fft.shape[0] // 2 - tmp.shape[0] // 2:fft.shape[0] // 2 + tmp.shape[0] // 2,
    fft.shape[1] // 2 - tmp.shape[1] // 2:fft.shape[1] // 2 + tmp.shape[1] // 2] = tmp

    return output, indices


def box_cutter_pad_ellips(fft, r, x=None, z=None, indices=None, edge_x=80, edge_z=80, dx=0, dz=0, e_type="tukey", e_w=0.1, peakswithfitting=False, shp=None):
    """
    A function that combines the box cutter and the ellipsoid functions (in succession).
    :param fft: An array to be cut
    :param r: Which quad to focus on. Choose between "quad1" and "quad2" (see box_cutter function for more)
    :return: The cut array, indices of peaks
    """        
    tmp, indices = box_cutter(fft, r, x, z, indices, peakswithfitting=peakswithfitting)
    tmp = ellipsoid(tmp, width=e_w, type=e_type)
    
    if shp == None:
        shape = fft.shape
        
    elif shp == "tight":
        shape = tmp.shape
        
    else: 
        shape = shp
    
    output = np.zeros(shape, dtype=np.complex128)
    output[shape[0] // 2 - tmp.shape[0] // 2:shape[0] // 2 + tmp.shape[0] // 2,
    shape[1] // 2 - tmp.shape[1] // 2:shape[1] // 2 + tmp.shape[1] // 2] = tmp
    return output, indices


def box_cutter(fft, r, x=None, z=None, indices=None, peakswithfitting=False):
    # First determine which cut is smaller - it might be important. 
    # print("Using the new box cutter")
    all_distx = []
    all_distz = []
    for q_temp in ["quad1", "quad2"]:
        lenx = fft.shape[0] // 2
        lenz = fft.shape[1] // 2
        
        if q_temp == 'quad1': 
            quad = fft[:lenx, :lenz]
        elif q_temp == 'quad2': 
            quad = fft[:lenx, lenz:]

        if indices is None:
            """ Hopefully just to estimate - then we fit. """
            indices1 = np.where(quad == quad[lenx//5:4*lenx//5, lenz//5:4*lenz//5].max())
            indices1 = [int(indices1[0]), int(indices1[1])]

        else:
            print("Indices provided: ")
            indices1 = indices     

        # Additional cut to remove 0th order, can be set at will
        xnumber = x or 0
        znumber = z or 0

        # Determining the distances
        if indices1[0] <= lenx // 2.:
            distx = indices1[0] - xnumber
        elif indices1[0] > lenx // 2.:
            distx = lenx - indices1[0] - xnumber

        if indices1[1] <= lenz // 2.:
            distz = indices1[1] - znumber
        elif indices1[1] > lenz // 2.:
            distz = lenz - indices1[1] - znumber
            
        else:
            print("Error! Setting box size to 10x10.")
            distx = 10
            distz = 10
        
        all_distx.append(abs(distx))
        all_distz.append(abs(distz))

    lenx = fft.shape[0] // 2
    lenz = fft.shape[1] // 2
    # print(f"------ {r=} ------") 
    if r == 'quad1':
        quad = fft[:lenx, :lenz]
        
    elif r == 'quad2':
        quad = fft[:lenx, lenz:]

    elif r == 'quad3':
        quad = fft[lenx:, :lenz]
        
    elif r == 'quad4':
        quad = fft[lenx:, lenz:]
        
#         elif r == 'quad3':
#             quad = fft[lenx:, :lenz]

#         elif r == 'quad4':
#             quad = fft[lenx:, lenz:]

    # Find Peaks in a selected quadrant - Perhaps only take the central half of it? 
    # To make sure we avoid the first order peaks. This is finding the peaks with just 
    # finding the max array of the array in the center 
    # Let's also do the fitting, and get the peak that way. 

    if indices is None:
        """ Hopefully just to estimate - then we fit. """
        indices1 = np.where(fft == quad[lenx//5:4*lenx//5, lenz//5:4*lenz//5].max())
        # print(indices1)
        indices1 = [int(indices1[0]), int(indices1[1])]


    else:
        # print("Indices provided: ")
        indices1 = indices     


    if peakswithfitting:
        try: 
            def two_d_gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
                (x, y) = xdata_tuple
                xo = float(xo)
                yo = float(yo)    
                g = offset + amplitude * np.exp(-(((x-xo)**2/(2*sigma_x**2)) + ((y-yo)**2/(2*sigma_y**2))))
                return g.ravel()

            # --- this is the part we need to add 
            # Now we need to 2D fit our quad1 --  FITTING 
            aoi = abs(quad)
            x_ar = np.linspace(0, aoi.shape[1]-1, aoi.shape[1])
            y_ar = np.linspace(0, aoi.shape[0]-1, aoi.shape[0])
            x_ar, y_ar = np.meshgrid(x_ar, y_ar)
            initial_guess = (1e7, indices1[1], indices1[0], 1, 1, 2e4)
            popt, _ = curve_fit(two_d_gaussian, (x_ar, y_ar), aoi.ravel(), p0=initial_guess, maxfev=10000)
            print("Fitting worked")
            print(f"Old indices: {indices1}")
            indices1 = [int(popt[2]), int(popt[1])]
            print(f"New indices: {indices1}")
            # ------------------------------------------------

        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) 
            print("Fitting didn't work bro.") 
            print("Returning to max values of the arrays.")
            print(f"Old indices: {indices1}")


    # Additional cut to remove 0th order, can be set at will
    xnumber = x or 0
    znumber = z or 0

    
    distx = abs(min(all_distx))
    distz = abs(min(all_distz))
    print(distx, distz)
    
    distx_new = min(distx, int(fft.shape[0]/fft.shape[1]*distz))
    distz_new = min(distz, int(fft.shape[1]/fft.shape[0]*distx))
    print(distx_new, distz_new)

    # Ensure they are the same ratio as the image for FFT reasons
    print(np.array(fft[indices1[0] - distx_new:indices1[0] + distx_new, indices1[1] - distz_new:indices1[1] + distz_new]).shape)
    
    return fft[indices1[0] - distx_new:indices1[0] + distx_new, indices1[1] - distz_new:indices1[1] + distz_new], indices1
    

def box_cutter_lessless_old(fft, r, x=None, z=None, indices=None, edge_x=80, edge_z=80, dx=0, dz=0, peakswithfitting=False):
    lenx = fft.shape[0] // 2
    lenz = fft.shape[1] // 2
    
    if r == 'quad1':
        quad = fft[:lenx, :lenz]
        
    elif r == 'quad2':
        quad = fft[:lenx, lenz:]
        
    elif r == 'quad3':
        quad = fft[lenx:, :lenz]
        
    elif r == 'quad4':
        quad = fft[lenx:, lenz:]
        


    # Find Peaks in a selected quadrant - Perhaps only take the central half of it? 
    # To make sure we avoid the first order peaks. This is finding the peaks with just 
    # finding the max array of the array in the center 
    # Let's also do the fitting, and get the peak that way. 

    if indices is None:
        """ Hopefully just to estimate - then we fit. """
        indices1 = np.where(quad == quad[lenx//5:4*lenx//5, lenz//5:4*lenz//5].max())
        indices1 = [int(indices1[0]), int(indices1[1])]


    else:
        print("Indices provided: ")
        indices1 = indices     

        
    if peakswithfitting:
        try: 

            def two_d_gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
                (x, y) = xdata_tuple
                xo = float(xo)
                yo = float(yo)    
                g = offset + amplitude * np.exp(-(((x-xo)**2/(2*sigma_x**2)) + ((y-yo)**2/(2*sigma_y**2))))
                return g.ravel()

            # --- this is the part we need to add 
            # Now we need to 2D fit our quad1 --  FITTING 
            aoi = abs(quad)
            x_ar = np.linspace(0, aoi.shape[1]-1, aoi.shape[1])
            y_ar = np.linspace(0, aoi.shape[0]-1, aoi.shape[0])
            x_ar, y_ar = np.meshgrid(x_ar, y_ar)
            initial_guess = (1e7, indices1[1], indices1[0], 1, 1, 2e4)
            popt, _ = curve_fit(two_d_gaussian, (x_ar, y_ar), aoi.ravel(), p0=initial_guess, maxfev=10000)
            print("Fitting worked")
            print(f"Old indices: {indices1}")
            indices1 = [int(popt[2]), int(popt[1])]
            print(f"New indices: {indices1}")
            # ------------------------------------------------

        except Exception as error:
            # handle the exception
            print("An exception occurred:", error) 
            print("Fitting didn't work bro.") 
            print("Returning to max values of the arrays.")
            print(f"Old indices: {indices1}")
            

    # Additional cut to remove 0th order, can be set at will
    xnumber = x or 0
    znumber = z or 0

    # Determining the distances
    if indices1[0] <= lenx // 2.:
        distx = indices1[0] - xnumber
    elif indices1[0] > lenx // 2.:
        distx = lenx - indices1[0] - xnumber

    if indices1[1] <= lenz // 2.:
        distz = indices1[1] - znumber
    elif indices1[1] > lenz // 2.:
        distz = lenz - indices1[1] - znumber

    else:
        print("Error! Setting box size to 10x10.")
        distx = 10
        distz = 10
    
    
    return quad[indices1[0] - distx:indices1[0] + distx, indices1[1] - distz:indices1[1] + distz], indices1
    





def box_cutter_lessOld(fft, r, x=None, z=None, indices=None,edge_x=80,edge_z=80,dx=0,dz=0):
    """
    A function that cuts out a large box over a half of the FFT space.
    The sections of the main array:
    ---------------
      3   |   4
    ---------------
      1   |   2
    ---------------
    :param fft: An array to be cut
    :param r: Which quad to cut to - the output half, if you will.
    :param x: Additional cut to remove the 0th order (default=None)
    :param z: Additional cut to remove the 0th order (default=None)
    :param indices: Indices of peaks (default=None, will attempt to find them automatically)
    :param edge_x: X edge in which to find the peak (default=80)
    :param edge_z: Z edge in which to find the peak (default=80)
    :param dx: Slight delta in x (default=0)
    :param dz: Slight delta in z (default=0)
    :return: The cut array, indices of peaks
    """
    lenx = fft.shape[0]
    lenz = fft.shape[1] // 2

#     if r == 'quad1':
#         quad = fft[:, 0:fft.shape[1] // 2]
#         quad_copy0 = fft[0:fft.shape[0] // 2, 0:fft.shape[1] // 2]

#     if r == 'quad2':
#         quad = fft[:, fft.shape[1] // 2:fft.shape[1]]
#         quad_copy0 = fft[0:fft.shape[0] // 2, fft.shape[1] // 2:fft.shape[1]]
        
#     ### Added these two to check if the processing works the way we expect it. 
#     if r == 'quad3':
#         quad = fft[:, fft.shape[1] // 2:fft.shape[1]]
#         quad_copy0 = fft[0:fft.shape[0] // 2, fft.shape[1] // 2:fft.shape[1]]
        
#     if r == 'quad4':
#         quad = fft[:, fft.shape[1] // 2:fft.shape[1]]
#         quad_copy0 = fft[0:fft.shape[0] // 2, fft.shape[1] // 2:fft.shape[1]]

    lenx = fft.shape[0] // 2
    lenz = fft.shape[1] // 2

    if r == 'quad1':
        quad = fft[:lenx, :lenz]
        quad_copy0 = fft[:lenx, :lenz].copy()
    elif r == 'quad2':
        quad = fft[:lenx, lenz:]
        quad_copy0 = fft[:lenx, lenz:].copy()
    elif r == 'quad3':
        quad = fft[lenx:, :lenz]
        quad_copy0 = fft[lenx:, :lenz].copy()
    elif r == 'quad4':
        quad = fft[lenx:, lenz:]
        quad_copy0 = fft[lenx:, lenz:].copy()
    else:
        raise ValueError("Invalid quadrant. Please choose 'quad1', 'quad2', 'quad3', or 'quad4'.")


    # Finding the peak in the selected quad
    if indices is None:
        indices1 = np.where(quad_copy0 == quad_copy0.max())
        print("Position of peaks: ", indices1)
        indices1 = [int(indices1[0]), int(indices1[1])]
    else:
        indices1 = indices

    # Correct if the 0th order is selected - wrong peak position.
    if indices is None:
        if indices1[0] not in range(edge_x, lenx - edge_x) or indices1[1] not in range(edge_z, lenz - edge_z):
            print("Error! You have zeroth order issues! Finding a new peak.")
            quad_copy = quad_copy0.copy()
            if r == "quad1":
                quad_copy[0:edge_x + dx, :] = 1E3
                quad_copy[:, 0:edge_z + dz] = 1E3

            elif r == "quad2":
                quad_copy[0:edge_x + dx, :] = 1E3
                quad_copy[:, lenz - (edge_z + dz):lenz] = 1E3

            # Plot of the blacked out area
            indices1 = np.where(quad_copy == quad_copy.max())
            print("New peaks: ", indices1)
            if len(indices1[0]) != 1:
                print("Multiple peaks detected. Only choosing the last ones.")
                indices1 = [int(indices1[0][0]), int(indices1[1][0])]
            print("New indices: ", indices1)
            indices1 = [int(indices1[0]), int(indices1[1])]
            print("Found the peak! It is at: ", indices1)
        # Otherwise you are happy.
        else:
            print("Found the peak! It is at", indices1)
    else:
        indices1 = indices

    # We will now create the box. The quad can be divided up in pieces:
    # -------------
    #   B,C | B,D
    # -------------
    #   A,C | A,D
    # -------------

    # Additional cut to remove 0th order, can be set at will
    xnumber = x or 0
    znumber = z or 0

    # Determining the distances
    if indices1[0] <= lenx // 2.:
        distx = indices1[0] - xnumber
    elif indices1[0] > lenx // 2.:
        distx = lenx - indices1[0] - xnumber

    if indices1[1] <= lenz // 2.:
        distz = indices1[1] - znumber
    elif indices1[1] > lenz // 2.:
        distz = lenz - indices1[1] - znumber

    else:
        print("Error! Setting box size to 10x10.")
        distx = 10
        distz = 10

    return quad[indices1[0] - distx:indices1[0] + distx, indices1[1] - distz:indices1[1] + distz], indices1


################### My OLD Box Cutter #####################
### This is the old box cutter; a small box in the quad only
def old_box_cutter(fft, r, x=None, z=None, indices=None):
    lenx = fft.shape[0] / 2
    lenz = fft.shape[1] / 2

    # ---------------
    #   3   |   4
    # ---------------
    #   1   |   2
    # ---------------

    ### some settings
    if r == 'quad1':
        quad = fft[0:fft.shape[0] / 2, 0:fft.shape[1] / 2]

    if r == 'quad2':
        quad = fft[0:fft.shape[0] / 2, fft.shape[1] / 2:fft.shape[1]]

    if r == 'quad3':
        quad = fft[fft.shape[0] / 2:fft.shape[0], 0:fft.shape[1] / 2]

    if r == 'quad4':
        quad = fft[fft.shape[0] / 2:fft.shape[0], fft.shape[1] / 2:fft.shape[1]]

    ### finding the peak
    if indices is None:
        indices1 = np.where(quad == quad.max())
        indices1 = [int(indices1[0]), int(indices1[1])]
    else:
        indices1 = indices

    ### If you have a zeroth order, you will have wrong peak position.
    if indices1[0] not in range(80, lenx - 80) or indices1[1] not in range(80, lenz - 80):
        print("Error! You have zeroth order issues! Finding a new peak.")

        quad_copy = quad.copy()

        if r == "quad1":
            quad_copy[0:100, :] = 1E3
            quad_copy[:, 0:100] = 1E3

        elif r == "quad2":
            quad_copy[0:100, :] = 1E3
            quad_copy[:, lenz - 100:lenz] = 1E3

        ### plot of the blacked out area

        indices1 = np.where(quad_copy == quad_copy.max())
        indices1 = [int(indices1[0]), int(indices1[1])]

        print("Found the peak! It is at", indices1)

        if 0:
            plt.figure()
            plt.imshow(np.log(np.abs(quad_copy)) * 2, cmap='bwr', origin='lower')
            plt.show()

    ### But else you are happy.
    else:
        print("Found the peak! It is at", indices1)

    ### We will now create the box. The quad can be divided up in pieces:
    # -------------
    #   B,C | B,D
    # -------------
    #   A,C | A,D
    # -------------

    ### Additional cut to remove 0th order, can be set at will
    xnumber = x or 0
    znumber = z or 0

    ### determining the distances.
    if indices1[0] <= lenx / 2.:
        distx = indices1[0] - xnumber

    elif indices1[0] > lenx / 2.:
        distx = lenx - indices1[0] - xnumber

    if indices1[1] <= lenz / 2.:
        distz = indices1[1] - znumber

    elif indices1[1] > lenz / 2.:
        distz = lenz - indices1[1] - znumber


    else:
        print("Error! Setting box size to 10x10.")
        distx = 10
        distz = 10

    return quad[indices1[0] - distx:indices1[0] + distx, indices1[1] - distz:indices1[1] + distz], indices1


###
### Hieronder op basis van HI.py
###

###
### Turns typical three images atoms, flat and dark into
### amplitude and argument arrays
###

def get_amp_ang(atoms, flat, dark, refocus=0.):
    atoms = squaroid(atoms - dark, width=0.1)
    flat = squaroid(flat - dark, width=0.1)

    ######################### FFT  ##############################
    fft_atoms = np.fft.fft2(atoms)
    fft_flat = np.fft.fft2(flat)

    ################## Creating the IFFT's ##################
    ### We create the cutouts for the data and the background
    ### You can pass additional cuts in x,z direction, but this is optional; the default value is 0
    ### Note that because you later cut everything in the same size, this may affect the rest too
    quad1, q1peak = box_cutter_pad_ellips(fft_atoms, "quad2", 100, 100)
    flatq1, f1peak = box_cutter_pad_ellips(fft_flat, "quad2", indices=q1peak)

    if 1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.log(np.abs(fft_atoms)))
        ax2.imshow(np.log(np.abs(quad1)))
        plt.show()

        np.save("fig1_atom", atoms)
        np.save("fig1_fft_atoms", np.log(np.abs(fft_atoms)) * 2)
        np.save("fig1_quad1", np.log(np.abs(quad1)) * 2)

    ### Cutting the quads in the same sizes.
    quad1cut, flatq1cut = sizecomp(quad1, flatq1)

    ### Now we FFT shift
    fft1 = np.fft.fftshift(quad1cut)
    flatfft1 = np.fft.fftshift(flatq1cut)

    ## Refocussing
    fft_kx = np.fft.fftfreq(fft1.shape[1], d=pix_size)
    fft_ky = np.fft.fftfreq(fft1.shape[0], d=pix_size)

    fft_k2 = fft_kx[None, :] ** 2 + fft_ky[:, None] ** 2

    refocus_exp = np.exp(-1j * fft_k2 * refocus / (2 * kvec))

    fft1 = fft1 * refocus_exp
    flatfft1 = flatfft1 * refocus_exp

    ### The IFFT
    inv1 = np.fft.ifft2(fft1) / np.fft.ifft2(flatfft1)
    inv1 = inv1[50:-50, 50:-50]

    ################### New pixel sizes ###########################
    ### We set the pixel sizes before we make any cut.

    pix_x = pix_size * (1. * atoms.shape[0] / inv1.shape[0])
    pix_z = pix_size * (1. * atoms.shape[1] / inv1.shape[1])

    # print "Pixel size was %E, now %E, %E m" %(pix_size, pix_x, pix_z)

    ##################################################################
    # parlist = [pix_x,pix_z, polariz, tof]

    ################## To phase and amplitude ########################
    # inv1 = inv1[1:,:-1]

    #### phase
    ang1 = -np.angle(inv1)
    ang1 = unwrapper(ang1)

    normfactor = ang1[:300, :300].mean()
    ang1 = ang1 - normfactor

    ### amplitude
    amp1 = np.abs(inv1) ** 2

    normfactor2 = amp1[:300, :300].mean()
    amp1 = amp1 / normfactor2

    amp1 = -np.log(amp1)

    return amp1, ang1, [pix_x, pix_z]


def fit_bimodal_ang(ang2, parlist, center=[50, 75]):
    ### 0 bg, 1 amp, 2 angle, 3 x_cent, 4 z_cent,
    ### 5 tf_A, 6 tf_Rx,7 tf_Rz, 8 g_A, 9 gw_x, 10 gw_z,
    ### 11 x_grad, 12 z_grad

    # parlist = [0,0,0,0] # These are optional and not required for this model

    ### Phase
    init_guess = [0., 1., 0., center[0], center[1], 15., 10., 10., 2., 10., 10., 0., 0.]

    output1ang, fit1ang = fitfunc(ang2, phasebimodalmod, init_guess, parlist)

    output1ang = fix_angle(output1ang, 2, [6, 9], [7, 10])

    return output1ang, fit1ang


def BECcenter(pic):
    gauspic = ndimage.filters.gaussian_filter(pic[5:-5, 5:-5], 4)

    indices1 = np.where(gauspic == gauspic.max())
    indices1 = [int(indices1[0]) + 5, int(indices1[1]) + 5]

    return indices1


