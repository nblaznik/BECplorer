import astropy.io.fits as pyfits
import numpy as np
import os
import time
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.odr as odr
import glob
import csv
import matplotlib

"""
    VARYING ONLY TOF, WHILE KEEPING WAIT TIME CONSTANT AT 762 MS.
    Might means we are imaging the same moment in the oscillations everytime,
    but with different expansion rates. 
"""

shotsday2 = [x for x in range(116, 131, 1)]

def get_parameter(date, shot, paramname):
    """
    Get the value of the parameter form the parameters.param file. Very often used, might be better to import
    it from the pcgk/fits file. But it might be better to include those functions here.
    """
    date = str(date)
    shot = str(shot).zfill(4)
    param = "N/A"
    path = '/storage/data/' + date + '/' + shot
    try:
        with open(path + '/parameters.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            for row in csvreader:
                if row[0] == paramname:
                    param = float(row[1])
    except:
        param = "N/A"
    return param

def createpic(date, shot):
    shot = str(shot).zfill(4)
    input_folder = '/storage/data/' + date + '/' + shot + '/'
    atom = pyfits.open(input_folder + '0.fits')[0].data.astype(float)[0]  # .mean(axis=0)
    flat = pyfits.open(input_folder + '1.fits')[0].data.astype(float)[0]  # .mean(axis=0)
    dark = pyfits.open(input_folder + '2.fits')[0].data.astype(float)[0]  # .mean(axis=0)
    pic = (atom - dark) / (flat-dark)
    pic = pic[200:-200, 200:-200]
    return pic



fig, ax = plt.subplots(7, 2, figsize=(7, 9), sharex=True)
plt.subplots_adjust(hspace=0., wspace=0.)
for i in range(7):
    shot = shotsday2[i]
    ax[i][0].imshow(createpic("20221205", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
    ax[i][0].set_ylim(800, 1150)
    ax[i][0].set_xlim(600, 1100)
    tof = get_parameter("20221205", shot, "tof")
    ax[i][0].text(0.01, 0.1, "TOF = {:} ms".format(tof), horizontalalignment='left', verticalalignment='center', transform=ax[i][0].transAxes)

for i in range(7):
    shot = shotsday2[i+7]
    ax[i][1].imshow(createpic("20221205", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
    ax[i][1].set_ylim(800, 1150)
    ax[i][1].set_xlim(600, 1100)
    tof = get_parameter("20221205", shot, "tof")
    ax[i][1].text(0.01, 0.1, "TOF = {:} ms".format(tof), horizontalalignment='left', verticalalignment='center', transform=ax[i][1].transAxes)

for axes in ax:
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
plt.show()
# plt.tight_layout()
