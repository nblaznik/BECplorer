# ------------------------------------------------------------------------------------------------------------------
# SUPPLEMENTARY FUNCTIONS FOR TOP IMAGING (ABSORPTION IMAGING)

# Original version by Sanne and Jasper, modified and edited by Nejc
# Last checked in March 2021
# ------------------------------------------------------------------------------------------------------------------



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
from skimage.restoration import unwrap_phase
# import hartreefockfit as hff   # Jasper's hartreefock fit


# --------------------------------------------------- CONSTANTS ------------------------------------------------------
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
pix_size = 6.5E-6 / 3.
kvec = 2. * np.pi / (589E-9)

# Magnetic Trap
# fx   = 91.9
# fz   = 15.2

# Decompressed trap 2.5
fx = 72.6
fz = 15.1

# Decompressed trap 1.5
# fx   = 44.6
# fz   = 15.4


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- FUNCTIONS -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# --------------------------- FIT FUNCTIONS --------------------------------------


def gauss1D(x, amp, sig, mu):
    """ Simple 1D Gaussian function, taking usual arguments. """
    return np.abs(amp) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def gauss2D(x, amp, sig, mu):
    """ Simple 2D Gaussian function, taking usual arguments, now double per variable. """
    return gauss1D(x[0], amp, sig[0], mu[0]) * gauss1D(x[1], 1., sig[1], mu[1])


def thomasfermi(x, amp, r, mid):
    """ A 2D Thomas Fermi function, a function of amplitude, radia and means (both x and y). """
    value = 1 - ((x[0] - mid[0]) / r[0]) ** 2 - ((x[1] - mid[1]) / r[1]) ** 2
    value[value < 0] = 0
    return np.abs(amp) * np.power(value, 3. / 2.)


def rotvars(rot, mid, x):
    """ The function rotating the variables by angle 'rot' around 'mid'. """
    x0 = np.cos(rot) * (x[0] - mid[0]) - np.sin(rot) * (x[1] - mid[1])
    x1 = np.cos(rot) * (x[1] - mid[1]) + np.sin(rot) * (x[0] - mid[0])
    return [x0, x1]


def gaussmod(B, x):
    """
    Modify the 2D Gaussian function - take the exponential of minus the Gaussian, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the thermal cloud.
    B array:
        B[0] = initial 'level' of the Gaussian, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = Gaussian parameters (amp, sig_x, sig_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return 0 * B[0] + B[1] * np.exp(-gauss2D(xrot, B[5], [B[6], B[7]], [0., 0.]))  # B[1]*


def tfmod(B, x):
    """
    Modify the 2D Thomas-Fermi function - take the exponential of minus the TF, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the condensate.
        B[0] = initial 'level' of the TF, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return 0. * B[0] + B[1] * np.exp(-thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]))


def bimodalmod(B, x):
    """
    Combine the Gaussian and TF model into a single bimodal function used to fit and model the distribution
    of the atoms in a superfluid - thermal cloud + the condensate. The variables for both of the fits are contained
    within the B array.
        B[0] = initial 'level' of the bimodal model, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
        B[8, 9, 10] = Gaussian parameters (amp, sig_x, sig_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return 0 * B[0] + B[1] * np.exp(-(thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]) + gauss2D(xrot, B[8], [B[9], B[10]], [0., 0.])))


# ------------------------------------- FUNCTION TO UNWRAP MULTIPLE TIMES --------------------------------------------
def unwrapper(data):
    """
    Function to unwrap the data multiple times, if necessary.
    :param data: The input data to be unwrapped.
    :return: Unwrapped data.
    """
    # return np.unwrap(np.unwrap(np.unwrap(np.unwrap(data,axis=0),axis=1),axis=0),axis=1)
    # return unwrap_phase(np.unwrap(np.unwrap(np.unwrap(np.unwrap(data,axis=0),axis=1),axis=0),axis=1))
    # return np.unwrap(np.unwrap(np.unwrap(data,axis=0),axis=1),axis=0)
    # return np.unwrap(np.unwrap(data,axis=1),axis=0)
    # return np.unwrap(data,axis=1)
    return unwrap_phase(data)  # ,wrap_around=(True, False))


# ----------------------------------------- FITTING PROCEDURE FUNCTION ------------------------------------------------
def fitfunc(data, model, guess, parlist):
    """
    Return the output parameters and the fit result based on the input data, specified
    model, guess parameters, and various fixed parameter list. Fitted using the ODR package.
    :param data: The input data to be fitted (2D array)
    :param model: The model used to fit the data
    :param guess: The guess values used as a starting point for fitting, depending on the model.
    :param parlist: The list of 'fixed' parameters [pix_x, pix_z, polarisation, time of flight]
    :return: Output.beta parameters
    :return: fit results - reshaped model
    """
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]

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


# def hffit(data,guess, mu, temp, parlist):
# if 0:
#     ### 0 bg, 1 ang, 2 x cent, 3 z cent, 4 mu, 5 T, 6 omega x, 7 omega z
#     physres = physresults(guess, parlist)
#     hffguess = [physres[0], physres[2], physres[3], physres[4], mu, temp, 2. * np.pi * fx, 2. * np.pi * fz, 1.1]
#
#     ### Grid
#     hff_r, hff_z = gridmaker(data)
#
#     ### Detract gradient and scale with polariz
#     data2 = (data - guess[11] * hff_r - guess[12] * hff_z - guess[0]) / parlist[2]
#
#     ### Scaling grid to meters
#     hff_r = parlist[0] * hff_r
#     hff_z = parlist[1] * hff_z
#     hff_fitvars = np.array([hff_r, hff_z]).reshape(2, -1)
#
#     ### fit!
#     hff_model = hff.new_model_omgfit
#     hff_fitmod = odr.Model(hff_model)
#
#     hff_myData = odr.Data(hff_fitvars, data2.flatten())
#     hff_myODR = odr.ODR(hff_myData, hff_fitmod, beta0=hffguess)
#
#     hff_myODR.set_job(fit_type=2)
#     hff_output = hff_myODR.run()
#
#     hff_output.pprint()
#
#     ### the fit function
#     hff_tf, hff_th = hff.new_model_omgfit(hff_output.beta, hff_fitvars, for_plot=True)
#     hff_tf = hff_tf.reshape(hff_r.shape[0], hff_z.shape[1])
#     hff_th = hff_th.reshape(hff_r.shape[0], hff_z.shape[1])
#
#     print(hff_output.beta[8])
#
#     # return data2, hff_output.beta, hff_tf, hff_th


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


# ------------------------------- Make a grid ---------------------------------


def gridmaker(data):
    """
    Creates a grid from the size of data.
    """
    x = np.arange(data.shape[0])
    z = np.arange(data.shape[1])
    xv, zv = np.meshgrid(x, z, indexing='ij')

    return xv, zv


# ------------------------------- Cutout around BEC ---------------------------------


def BECcutout(ang1, amp1, numx=50, numz=75, cut=False, showFig=False):
    """
    Cutout around the BEC image. First find the max value, then create a square around
    it of dimensions 50x75.
    :param ang1: The input array of angles
    :param amp1: The input array of amplitudes
    :param numx: The x shape of the BEC cutout
    :param numz: the z shape of the BEC cutout
    :param cut: Whether to cut out a portion of the image, default = False
    :param showFig: Whether to show figure of the parameters, default = False
    :return ang2:
    """

    gausang = ndimage.filters.gaussian_filter(ang1, sigma=4)

    if cut:
        gausang = gausang[300:700, 10:500]
        ang1 = ang1[300:700, 10:500]
        amp1 = amp1[300:700, 10:500]

    indices1 = np.where(gausang == gausang.max())
    indices1 = [int(indices1[0]), int(indices1[1])]

    ang2 = ang1[indices1[0] - numx:indices1[0] + numx, indices1[1] - numz:indices1[1] + numz]
    amp2 = amp1[indices1[0] - numx:indices1[0] + numx, indices1[1] - numz:indices1[1] + numz]

    if showFig:
        figx, ((axx1, axx2), (axx3, axx4)) = plt.subplots(2, 2)
        axx1.imshow(gausang, cmap='bwr', interpolation='none', origin="lower")
        # axx2.imshow(np.log(gausamp), cmap='bwr', interpolation='none',origin="lower")
        axx3.imshow(ang2, cmap='bwr', interpolation='none', origin="lower")
        axx4.imshow(np.log(np.abs(amp2)) * 2, cmap='bwr', interpolation='none', origin="lower")
        plt.show()

    return ang2, amp2


# ------------------------------- List Cutter ---------------------------------


def sizecomp(list1, list2):
    """
    A function which cuts two 2D arrays into same sized arrays, but
    clipping away the edges on both sides from the bigger one.

    :param list1: first list
    :param list2: second list
    :return: returns two cut lists of the same size.
    """

    corr0 = (list1.shape[0] - list2.shape[0]) // 2
    corr1 = (list1.shape[1] - list2.shape[1]) // 2

    # Along x-axis
    if corr0 > 0:
        list1 = list1[corr0:list1.shape[0] - corr0, :]
    elif corr0 < 0:
        corr0 = np.abs(corr0)
        list2 = list2[corr0:list2.shape[0] - corr0, :]

    # Along z-axis
    if corr1 > 0:
        list1 = list1[:, corr1:list1.shape[1] - corr1]
    elif corr1 < 0:
        corr1 = np.abs(corr1)
        list2 = list2[:, corr1:list2.shape[1] - corr1]

    # print("Cutting the arrays to shapes:")
    # print(list1.shape, list2.shape)
    return list1, list2





def listcut(list1, list2, list3, list4):
    """
    Function to automatically cut 4 lists to the same size, by clipping
    away from the edges of the bigger ones.

    Parameters: four lists to be cut to the same size.
    Returns: four lists cut to the same size.
    """

    # print("Original sizes:", list1.shape, list4.shape, list3.shape, list4.shape)

    list1, list2 = sizecomp(list1, list2)
    list3, list4 = sizecomp(list3, list4)
    # print("First cut:", list1.shape, list2.shape, list3.shape, list4.shape)

    list1, list3 = sizecomp(list1, list3)
    list2, list4 = sizecomp(list2, list4)
    # print("Final cut:", list1.shape, list1.shape, list1.shape, list1.shape)

    return list1, list2, list3, list4



# ------------------------------- Box Cutters ---------------------------------


### This is the old box cutter; a small box in the quad only
def old_box_cutter(fft, r, x=None, z=None, indices=None):
    """
    This is the (older version of the) box cutter. This one cuts the FTT array into four quadrants,
    and finds the peak in the selected quadrant (if not specified in the function itself).

    The quadrant structure of the array is as follows:
    ---------------
    |  3   |   4  |
    ---------------
    |  1   |   2  |
    ---------------

    :param fft: The input array we would like to 'cut'
    :param r: Specifiying which quadrant we would like to cut to. Allowed variables "quad1", "quad2", "quad3" and "quad4"
    :param x:
    :param z:
    :param indices: Indices of the peak within the quadrant 'r'. By default set to None and is set to the max value.
    :return

    """
    lenx = fft.shape[0] // 2
    lenz = fft.shape[1] // 2

    # Cut the correct quadrant
    if r == 'quad1':
        quad = fft[0:fft.shape[0] // 2, 0:fft.shape[1] // 2]

    if r == 'quad2':
        quad = fft[0:fft.shape[0] // 2, fft.shape[1] // 2:fft.shape[1]]

    if r == 'quad3':
        quad = fft[fft.shape[0] // 2:fft.shape[0], 0:fft.shape[1] // 2]

    if r == 'quad4':
        quad = fft[fft.shape[0] // 2:fft.shape[0], fft.shape[1] // 2:fft.shape[1]]

    # Find the peak if not specified
    if indices is None:
        indices1 = np.where(quad == quad.max())
        indices1 = [int(indices1[0]), int(indices1[1])]
    else:
        indices1 = indices

    # Check if you have the zeroth order - then you will have wrong peak position.
    if indices1[0] not in list(range(80, lenx - 80)) or indices1[1] not in list(range(80, lenz - 80)):
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

    # Otherwise, you are happy.
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


############################## Abel transform ##############################
### input: the image, its fit parameters, a choice of mu (mux, muz, ...),
### and the pixel size.

def AbelT(ang, amp, outputlist, muchoice, parlist):
    pix_x = parlist[0]
    pix_z = parlist[1]
    polariz = parlist[2]
    tof = parlist[3]

    angle = outputlist[2] * 180. / np.pi

    ang, amp = BECcutout(ang, amp)

    # print "Angle = %f" %angle

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

    # output1pic, fit1pic = fitfunc(ang1, phasebimodalmod, extra_guess, parlist)
    output1pic, fit1pic = fitfunc(gauss, phasebimodalmod, extra_guess, parlist)

    xcenter = int(output1pic[3]) + 1
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

### This version cuts out a large box over a half of the FFT space
def box_cutter(fft, r, x=None, z=None, indices=None, marg_x=80, marg_z=80):
    lenx = fft.shape[0]
    lenz = fft.shape[1] // 2

    # ---------------
    #   3   |   4
    # ---------------
    #   1   |   2
    # ---------------

    ### some settings
    if r == 'quad1':
        quad = fft[:, 0:(fft.shape[1] // 2)]
        quad_copy0 = fft[0:fft.shape[0] // 2, 0:fft.shape[1] // 2]

    if r == 'quad2':
        quad = fft[:, fft.shape[1] // 2:fft.shape[1]]
        quad_copy0 = fft[0:fft.shape[0] // 2, fft.shape[1] // 2:fft.shape[1]]

    ### finding the peak in the quarter quad
    if indices is None:
        indices1 = np.where(quad_copy0 == quad_copy0.max())
        indices1 = [int(indices1[0]), int(indices1[1])]
    else:
        indices1 = indices

    ### If you have a zeroth order, you will have wrong peak position.
    if indices1[0] not in list(range(marg_x, lenx - marg_x)) or indices1[1] not in list(range(marg_z, lenz - marg_z)):
        print("Error! You have zeroth order issues! Finding a new peak.")

        quad_copy = quad_copy0.copy()

        if r == "quad1":
            quad_copy[0:marg_x, :] = 1E3
            quad_copy[:, 0:marg_z] = 1E3

        elif r == "quad2":
            quad_copy[0:marg_x, :] = 1E3
            quad_copy[:, lenz - marg_z:lenz] = 1E3

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

    tmp = quad[indices1[0] - distx:indices1[0] + distx, indices1[1] - distz:indices1[1] + distz]

    if 0:
        plt.figure()
        plt.suptitle("QuadReturned")
        plt.imshow(np.log(np.abs(tmp)))
        plt.show()

    return tmp, indices1


""" 
# Old stuff, commented out, but keep just in case. 

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



"""

