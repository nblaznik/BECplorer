# #######################################################################################
#  _____                  _                                          _                  #
# |_   _|                | |                                        | |                 #
#   | | _ __  _ __  _   _| |_   _ __   __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___   #
#   | || '_ \| '_ \| | | | __| | '_ \ / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|  #
#  _| || | | | |_) | |_| | |_  | |_) | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \  #
#  \___/_| |_| .__/ \__,_|\__| | .__/ \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/  #
#            | |               | |                                                      #
#            |_|               |_|                                                      #
# #######################################################################################

m_seq=[
    # [date, shot, dz_focus, mode, quad, NUM, bin, cut, intial_guess_parameters, time_between_shots]
    [20220517, 38, 0.000, 'gauss', 'quad1', 1, [4, 4], [1, -1, 1, -1], [0., [1100., 1120.], 1., [50., 50.], 1., [80., 600.]], 130],  # N = 0
    [20220517, 39, 0.000, 'gauss', 'quad1', 1, [4, 4], [700, -700, 600, -600], [0., [1100., 1120.], 1., [50., 50.], 1., [80., 600.]], 130],  # N = 1
    [20220517, 40, 0.000, 'gauss', 'quad1', 1, [4, 4], [1, -1, 1, -1], [0., [1100., 1120.], 1., [50., 50.], 1., [80., 600.]], 130],  # N = 2
    [20220517, 41, 0.000, 'gauss', 'quad1', 1, [4, 4], [700, -700, 1, -1],  [0., [1100., 1120.], 1., [50., 50.], 1., [80., 600.]], 130],  # N = 3
    [20220518, 4, 0.000, 'gauss', 'quad1', 1, [4, 4], [700, -700, 1, -1], [0., [1100., 1120.], 1., [50., 50.], 1., [80., 600.]], 130],  # N = 4
    [20220518, 9, 0.000, 'tf', 'quad1', 1, [1, 1], [1050, -950, 900, -900], [0., [1100., 1120.], 1., [10., 60.], 1., [10., 60.]], 130],  # N = 5, nice condensate
    [20220518, 13, 0.000, 'tf', 'quad1', 1, [1, 1], [1050, -950, 900, -900], [0., [1100., 1120.], 1., [10., 60.], 1., [10., 60.]], 130],  # N = 6, nice condensate
    [20220518, 15, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 700, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 130],  # N = 7, nice thermal cloud, 10 images
    [20220518, 16, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 650, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 1080],  # N = 8, nice thermal cloud, 25 images
    [20220518, 17, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 650, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 1080],  # N = 9, nice thermal cloud, 25 images
    [20220518, 21, 0.000, 'gauss', 'quad1', 1, [1, 1], [1000, -950, 650, -850], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 1000],  # N = 10, through cooling, 25 images
    [20220518, 22, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 1000],  # N = 11, through cooling, 25 images
    [20220518, 23, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 1000],  # N = 12, through cooling (T), after 5s RF2, 25 images
    [20220518, 24, 0.000, 'tf', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 180],  # N = 13, through cooling (C), after 15s RF2, 10 images
    [20220518, 25, 0.000, 'tf', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 180],  # N = 14, through cooling, after 15s RF2, 10 images
    [20220518, 26, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 1000],  # N = 15, through cooling (T/C), after 5s RF2, 10 images
    [20220518, 27, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 100],  # N = 16, through cooling, after 5s RF2, 25 images
    [20220518, 29, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 600],  # N = 17, same as above- ignore 15
    [20220518, 30, 0.000, 'gauss', 'quad1', 1, [4, 4], [1000, -950, 550, -750], [0., [1100., 1120.], 1., [10., 60.], 1., [80., 600.]], 600],  # N = 18, same as above
]

# To select a particular run (with the best cut and binning options, guess etc.)
N = 17
ignore = [15]
# #######################################################################################
#                                       END                                             #
# #######################################################################################


import os
import numpy as np
import csv
import scipy.odr as odr

try:
    from ..fit.fitfunctions import *
    from ..fit import OAH_refocus as OAH_refocus
except:
    from fitfunctions import *
    import OAH_refocus as OAH_refocus

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ################################# CONSTANTS #################################
detuning = 0  # change to 350 for OAH
wavelength = 589e-9
pixelsize = 6.5E-6 / 2.63
prefactor = float((1 + 4 * (float(detuning) ** 2)) * 2 * np.pi / (3 * (float(wavelength) ** 2)) * 18. / 5.)
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
fx = 91.9
fz = 15.2
linew = 9.7e-3
det_1 = 35e3
ang = .05
lamb0 = 589.1e-3
det_0 = det_1 + 15.8e-3
det_2 = det_1 - 34.4e-3
sigm_l = 3 * wavelength ** 2 / (2 * np.pi)
k0 = 2 * np.pi / wavelength
polarizability = 2j/3 * sigm_l / k0 * (1 / (1 - 2j * det_0 / linew) + 1 / (1 - 2j * det_1 / linew) + 1 / (1 - 2j * det_2 / linew))
print(polarizability)

# prefactor = prefactor * np.pi * polarizability / e0 / wavelength

# ########################### VARIABLES #################################
date = m_seq[N][0]  # Date of measurement
shot = m_seq[N][1]  # Shot number (38 - 42, 44-46)
dz_focus = m_seq[N][2]  # Defocus factor
mode = m_seq[N][3]  # gauss, tf or bimodal
quad = m_seq[N][4]  # Quad in the fourier space - depends on ref beam
NUM = m_seq[N][5]  # Number in a sequence
bin = m_seq[N][6]  # Binning (binx, binz)
cut = m_seq[N][7]  # Cutting (xmin, xmax, zmin, zmax)
ang, center, tfa, tfw, ga, gw = m_seq[N][8]  # Initial fit guess:
time_between_shots = m_seq[N][9]  # Time between shots

path = '/storage/data/' + str(date) + '/'
image = str(shot).zfill(4) + '/'
full_path = path + image

# STEP 1: convert the interference pattern into density profile
def OAHprocess(date, shot, dz_focus):
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    full_path = path + image
    # If the preprocessing was done before, take that file
    if os.path.exists(full_path + "pics_foc_ss_{:}.fits".format(dz_focus)):  # If single shot:
        print("Analysis previously completed")
        output = pyfits.open(full_path + "pics_foc_ss_{:}.fits".format(dz_focus))[0].data.astype(float)[:]
    elif os.path.exists(full_path + "pics_foc_{:}.fits".format(dz_focus)):  # If multi shot
        print("Analysis previously completed")
        output = pyfits.open(full_path + "pics_foc_{:}.fits".format(dz_focus))[0].data.astype(float)[:]
    else:
        # Import 0.fits file to get the number of shots
        atom_all = pyfits.open(full_path + '0.fits')[0].data.astype(float)[:]
        output = []
        # If a single shot only
        if atom_all.shape[0] == 1:
            ang = OAH_refocus.HI_refocus(date, shot, 0, dz_focus, quad=quad, plot=False)
            output.append(
                ang)  # We do this, so that we can call the 0th element of the group, as we do with multiple images.
            prihdr = pyfits.open(full_path + "0.fits")[0].header  # Get the header
            hdu = pyfits.PrimaryHDU(output, header=prihdr)
            hdu.writeto(full_path + "pics_foc_ss_{:}.fits".format(dz_focus))

        else:
            for it in range(atom_all.shape[0]):
                os.system('clear')
                ang = OAH_refocus.HI_refocus(date, shot, it, dz_focus, quad=quad, plot=False)
                output.append(ang)
            prihdr = pyfits.open(full_path + "0.fits")[0].header  # Get the header
            hdu = pyfits.PrimaryHDU(output, header=prihdr)
            hdu.writeto(full_path + "pics_foc_{:}.fits".format(dz_focus))  # Save it now, for next time
    return output

# STEP 2: prep the picture and optimize for quick fitting
def picPrep(output, NUM, cut=cut, bin=bin):
    xmin, xmax, zmin, zmax = cut
    xbin, zbin = bin
    mask = output[NUM] == 0
    pic = np.ma.array(output[NUM], mask=mask)
    # Cut pic:
    pic = pic[xmin:xmax, zmin:zmax]
    # Bin pic:
    if pic.shape[0] % xbin != 0:
        pic = pic[:-(pic.shape[0] % xbin), :]
    if pic.shape[1] % zbin != 0:
        pic = pic[:, :-(pic.shape[1] % zbin)]
    pic = pic.reshape(pic.shape[0] // xbin, xbin, pic.shape[1] // zbin, zbin).mean(axis=3).mean(axis=1)

    # Normalize pic
    pic = -pic + 1
    return pic

# STEP3: fit each of those images with a gaussian and extract the particle numbers and temperatures.
def fitPic(pic, mode=mode, date=date, shot=shot, bin=bin, cut=cut):
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    full_path = path + image
    x = np.arange(pic.shape[0])
    y = np.arange(pic.shape[1])
    xv, yv = np.meshgrid(x, y, indexing='ij')
    fitvars = np.array([xv, yv]).reshape(2, -1)

    par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'tfamp', 'tfxw', 'tfyw', 'gamp', 'gxw', 'gyw']
    init_guess = np.array([0., 1., ang, center[0], center[1], tfa, tfw[0], tfw[1], ga, gw[0], gw[1]])
    bin_scaling = np.array([1., 1., 1., bin[0], bin[1], 1., bin[0], bin[1], 1., bin[0], bin[1]])
    rng_offset = np.array([0., 0., 0., cut[0], cut[2], 0., 0., 0., 0., 0., 0.])
    to_physical = np.array(
        [1., 1., 1., pixelsize, pixelsize, prefactor, pixelsize, pixelsize, prefactor, pixelsize, pixelsize])
    corr_guess = (init_guess - rng_offset) / bin_scaling

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
        odrmodel = odr.Model(bimodalmod)
        # Store information for the bimodal fitting model


    # Run the ODR Fit procedure.
    odrdata = odr.Data(fitvars[:, ~pic.mask.flatten()], pic.flatten()[~pic.mask.flatten()])
    odrobj = odr.ODR(odrdata, odrmodel, beta0=corr_guess)
    odrobj.set_job(2)  # Ordinary least-sqaures fitting
    odrout = odrobj.run()
    odrout.pprint()

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
        fitresultgauss = gaussmod(np.append(odrout.beta[:5], odrout.beta[-3:]), fitvars).reshape(pic.shape[0],
                                                                                                 pic.shape[1])
        fitguess = bimodalmod(corr_guess, fitvars).reshape(pic.shape[0], pic.shape[1])

    print("The shape of the pic file: \n{:}\n".format(pic.shape))
    print("The guess parameters: \n{:}\n".format(corr_guess))
    print("Odrout: \n{:}\n".format(odrout.beta))

    # As the entire output, except for the angle, has to be positive,
    # we take the absolute value of the entire list, then put the angle back in.
    ang_temp = odrout.beta[2]
    odrout.beta = np.abs(odrout.beta)
    odrout.beta[2] = ang_temp % np.pi

    # Converts the fit results to absolute pixel values in the unbinned image.
    fit_results = odrout.beta * bin_scaling + rng_offset
    phys_results = fit_results * to_physical

    with open(full_path + 'parameters.param', 'r') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        for row in csvreader:
            if row[0] == "tof":
                tof = float(row[1]) / 1000.
            if row[0] == "rftime":
                rftime = float(row[1])

    ncount = -np.log(pic.flatten()).sum() * prefactor * pixelsize ** 2 * bin[0] * bin[1]
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
    return fitguess, fitresult, ntotal, tx, tz

# STEP 5: Plot all
def plotAll(preview_plane=True):
    """ If preview plane then display the fits as well. """
    # Plots
    output = OAHprocess(date, shot, dz_focus)  # OAH analysis
    pic_num = pyfits.open(full_path + '0.fits')[0].data.astype(float)[:].shape[0]
    if preview_plane:
        fig, ax = plt.subplots(pic_num, 1, figsize=(15, 15))

    pnums = []
    txs = []
    tzs = []
    time_l = []
    for i in range(pic_num):
        pic = picPrep(output, i, cut=cut, bin=bin)  # Optimize the pic
        fitguess, fitresult, ntotal, tx, tz = fitPic(pic, mode, date=date, shot=shot)
        levels = np.linspace(pic.min(), pic.max(), 3)
        if preview_plane:
            ax[i].imshow(pic, vmin=pic.min(), vmax=pic.max())
            ax[i].contour(fitresult, levels, cmap='gray',  vmin=pic.min(), vmax=pic.max())
            ax[i].set_xticks([])

        if i not in ignore:  #To enable ignoring poor fits
            pnums.append(ntotal)
            txs.append(tx)
            tzs.append(tz)
            time_l.append(i*time_between_shots)

    fig2 = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
    gs.update(wspace=0.25, hspace=0.)  # set the spacing between axes.

    ax1 = fig2.add_subplot(gs[0, 0])
    ax2 = fig2.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig2.add_subplot(gs[:, 1])
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.plot(time_l, pnums)
    ax2.plot(time_l, txs, label="Temperature-x")
    ax2.plot(time_l, tzs, label="Temperature-z")
    ax3.scatter(np.log(txs), np.log(pnums), label="Temp-x")
    ax3.scatter(np.log(tzs), np.log(pnums), label="Temp-z")
    ax3.legend()

    ax1.set_ylabel("Particle Number")
    ax2.set_ylabel("Temparature [K]")
    ax2.set_xlabel("Time [ms]")
    ax3.set_xlabel("Log (T)")
    ax3.set_ylabel("Log (N)")

    ax1.ticklabel_format(axis='both', style='', scilimits=(0, 0))
    ax2.ticklabel_format(axis='both', style='', scilimits=(0, 0))

    plt.show()



plotAll()