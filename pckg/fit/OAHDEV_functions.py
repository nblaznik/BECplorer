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

# Import own fit functions
import OAH_functions as f1
from scipy.optimize import curve_fit
from fitfunctions import gaussmod, tfmod, bimodalmod, tfaxialmod, gaussmod_OAH, tfmod_OAH, bimodalmod_OAH
from scipy.ndimage.interpolation import rotate
from scipy.special import zeta


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


# Initial values
initial_fit_vals = {
    "offset": 0.,
    "amp_ov": 1.,
    "ang": 0,
    "center": (50, 850),
    "tfa": 1,
    "tfw": (20, 50),
    "ga": 1,
    "gw": (200, 200),
    "j_guess": 40 ,
    "axamp": 1,
    "x_shift": 0.0,
    "squeeze_par": 1.0,
}



##################################################################################################
###############################   GENERAL PROCESSING FUNCTIONS ################################### 
##################################################################################################
##################################################################################################

def openFiles(date, shot, num, multi_loop=2):
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    # Opening files
    if multi_loop != 0:
        ff = ["{:}.fits".format(n) for n in range(multi_loop + 2)]
        atoms = [pyfits.open(path + image + fn)[0].data.astype(float)[:] for fn in ff[:-2]]
        flat = pyfits.open(path + image + ff[-2])[0].data.astype(float)[0][xmin:xmax, zmin:zmax]
        dark = pyfits.open(path + image + ff[-1])[0].data.astype(float).mean(axis=0)[xmin:xmax, zmin:zmax]
        atoms = np.concatenate(atoms)
        atoms = atoms[num][xmin:xmax, zmin:zmax]
    else: 
        atoms = pyfits.open(path + image + '0.fits')[0].data.astype(float)[num][xmin:xmax, zmin:zmax]
        flat = pyfits.open(path + image + '1.fits')[0].data.astype(float)[0][xmin:xmax, zmin:zmax]
        dark = pyfits.open(path + image + '2.fits')[0].data.astype(float).mean(axis=0)[xmin:xmax, zmin:zmax]
        
    return atoms, flat, dark

def makeAPic(date, shot, num=0, multi_loop=2):
    atom, flat, dark = openFiles(date, shot, num, multi_loop)
    # --------------------------------------- CREATE THE PICTURE ARRAY ------------------------------------------
    pic = (atom - dark) / (flat - dark)
    # pic = atom / flat
    pic[pic < 0.0001] = 0.0001
    mask = flat - dark < 25.
    pic = np.ma.array(pic, mask=mask)
    return pic
    
def first_nonzero(arr, axis):
    """ A function that finds the non-zero elements of the ellipsoid array. """
    invalid_val=arr.shape[axis]
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def get_parameter(date, seq, paramname):
    """
    Get the value of the parameter form the parameters.param file. Very often used, might be better to import
    it from the pcgk/fits file. But it might be better to include those functions here.
    """
    date = str(date)
    run_id = str(seq).zfill(4)
    path = "/storage/data/"
    param = "N/A"
#     try:
    with open(path + date + '/' + run_id + '/parameters.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        for row in csvreader:
            if row[0] == paramname:
                param = float(row[1])
#     except:
#         param = "N/A"
    return param

def get_fit_parameter(date, seq, paramname, mode='gauss'):
    """
    Get the value of the parameter form the parameters.param file. Very often used, might be better to import
    it from the pcgk/fits file. But it might be better to include those functions here.
    Mode can be "gauss", "bimodal", "tf"
    """
    date = str(date)
    run_id = str(seq).zfill(4)
    path = "/storage/data/"
    param = "N/A"
#     try:
    with open(path + date + '/' + run_id + '/fit_{:}.param'.format(mode)) as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        for row in csvreader:
            if row[0] == paramname:
                param = float(row[1])
#     except:
#         param = "N/A"
    return param

def get_comment(date, shot):
    file1 = open(f'/storage/BECViewer/comments/{date}_{str(shot).zfill(4)}.txt', 'r')
    return float(file1.read())

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
        title.set_text("Coordinates Recorded: \n X = {:} - {:}, Y = {:} - {:} \nPress Enter to save.".format(round(xtemp1),round(xtemp2),round(ytemp1),round(ytemp2)))
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



##################################################################################################
###################################   FFT PROCESSING FUNCTIONS ################################### 
##################################################################################################
##################################################################################################

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
        return a * x**2 + b*x + c
    

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


def pre_fft(date, shot, num, multi_loop=0):
    """ Do all the processing up and including to taking the FFT."""
    atoms, flat, dark = openFiles(date, shot, num, multi_loop=multi_loop)
    atoms_copy, flat_copy, atoms, flat = cutSquaroid(atoms, flat, dark)
    fft_atoms, fft_flat, fft_atoms_copy, fft_flat_copy = takeFFT(atoms, flat, atoms_copy, flat_copy)
    return atoms, flat, dark, fft_atoms, fft_flat


def rest_fft(quad1, flatq1, dz_focus=0.000, fullfield=False):
    """ 
    To be used after the window with the correct peak has been cut. Very useful when optimizing and 
    playing around with the FFT peaks cutting. 
    """
    
    # All the rest to generate image
    quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

    # Rest of the processing
    # ------------------------------------------------ FFT SHIFT ----------------------------------------------------
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
#     focus = np.exp(-1j * dz_focus * np.sqrt(k0 **2 - fft_k2))
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
    if fullfield:
        return inv1, fft1, flatfft1
    else: 
        return ang1, amp

def post_fft(fft_atoms, fft_flat, dz_focus, quad_cut="quad1", el_x=10, el_z=100, edge_x=10, edge_z=80, e_w=0.1, fullfield=False, shp=None):
    """ All the processing after taking the FFT. Output the ang and amp. """
    # All the rest to generate image
    quad1, q1peak = f1.box_cutter_pad_ellips(fft_atoms, quad_cut, el_x, el_z, edge_x=edge_x, edge_z=edge_z, e_w=e_w, shp=shp)
    flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat, quad_cut, el_x, el_z, edge_x=edge_x, edge_z=edge_z, e_w=e_w, indices=q1peak, shp=shp)
    
    return rest_fft(quad1, flatq1, dz_focus=dz_focus, fullfield=fullfield)



def fft_analysis(date, shot, num, dz_focus, quad_cut="quad1", multi_loop=0, el_x=10, el_z=100, edge_x=10, edge_z=80, e_w=0.1, shp=None):
    """ 
    Do the entire thing - for a date, shot, num, for a particular quad cut, and particluar focus. 
    Also output the intermediate steps. 
    """
    atoms, flat, dark, fft_atoms, fft_flat = pre_fft(date, shot, num, multi_loop=multi_loop)
    ang1, amp = post_fft(fft_atoms, fft_flat, dz_focus, quad_cut, shp=shp)
    return atoms, flat, dark, fft_atoms, fft_flat, ang1, amp




##################################################################################################
###################################### FITTING FUNCTIONS ######################################### 
##################################################################################################
    ##################################################################################################


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
    
    print(init_guess)
    
    
    # Normalize everything between 1 and 0
    if normalize:
#         pic = pic[10:-10, 10:-10]
        pic = pic + abs(pic.min()) + 0.00001
        avg_bg = pic.mean()
        pic = pic/avg_bg

    if invert:
        # Normalize such that the background is at 1, the peak is at zero.         
        pic= - pic + 0.5

        
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

        if mode == "doubletf":
            corr_guess = np.append(corr_guess[:8], [center2[0], center2[1], tfa2, tfw2[0], tfw2[1]])
            bin_scaling = np.append(bin_scaling[:8], bin_scaling[3:8])
            rng_offset = np.append(rng_offset[:8], rng_offset[3:8])
            par_names = np.append(par_names[:8], par_names[3:8])
            to_physical = np.append(to_physical[:8], to_physical[3:8])
            odrmodel = odr.Model(double_tf_OAH)  # Store information for the gaussian fitting model

            
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
        
        if mode == "doubletf":
            fitresult = double_tf_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = double_tf_OAH(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

        if mode == "tf":
            fitresult = tfmod_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitguess = tfmod_OAH(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = []
            fitresulttf = []

        if mode == "bimodal":
            fitresult = bimodalmod_OAH(odrout.beta, fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresulttf = tfmod_OAH(odrout.beta[:8], fitvars).reshape(pic.shape[0], pic.shape[1])
            fitresultgauss = gaussmod_OAH(np.append(odrout.beta[:5], odrout.beta[-3:]), fitvars).reshape(pic.shape[0],pic.shape[1])
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
            fitresultgauss = gaussmod(np.append(odrout.beta[:5], odrout.beta[-3:]), fitvars).reshape(pic.shape[0],pic.shape[1])
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



def FFT_window_cut_compare(date, shot, num=0, quad_cut="quad1", dz_focus=0.000, pad=1000):
    """ 
        Function that produces images based on different paddings to the FFT ellipsoid cut. 
    """
    atoms, flat, dark, fft_atoms, fft_flat = pre_fft(date, shot, num)
    
    # MAKE 3 IMAGES - for different window sizes
    # Keep the first one to the original size
    quad1, q1peak = f1.box_cutter_pad_ellips(fft_atoms, quad_cut, 10, 100, edge_x=10, edge_z=80, e_w=0.1)
    flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat, quad_cut, indices=q1peak)

    # Cut the second one to the size of the ellipsoid.
    pad_in = 0
    xbox = first_nonzero(quad1, axis=0).min() - pad_in
    ybox = first_nonzero(quad1, axis=1).min() - pad_in
    quad2 = quad1[xbox:-xbox, ybox:-ybox]
    flatq2 = flatq1[xbox:-xbox, ybox:-ybox]

    # Pad the third one 
    pad_out = pad     # 1000 -> 21s; 2000 -> 40s; 3000 -> 1m 15s; 4000 -> 1m 44s
    quad3 = np.pad(quad1, (pad_out), "constant", constant_values=(0.+0.j,0.+0.j))
    flatq3 = np.pad(flatq1, (pad_out), "constant", constant_values=(0.+0.j,0.+0.j))

    # Rest of FFT
    ang1, amp1 = rest_fft(quad1, flatq1, dz_focus=dz_focus)
    ang2, amp2 = rest_fft(quad2, flatq2, dz_focus=dz_focus)
    ang3, amp3 = rest_fft(quad3, flatq3, dz_focus=dz_focus)

    clear_output()
    print("Processing FFT processed.")
    quads = (quad1, quad2, quad3)
    angs = (ang1, ang2, ang3)
    amps = (amp1, amp2, amp3)
    
    return quads, angs, amps

