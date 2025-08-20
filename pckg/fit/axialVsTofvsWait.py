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
    VARYING BOTH TOF AND WAIT TIME, SUCH THAT THE TOTAL TIME IS THE SAME. 
    Might means we are imaging the same moment in the oscillations everytime,
    but with different expansion rates. 
"""


shotsday1 = [x for x in range(231, 238, 1)]
shotsday2 = [x for x in range(71, 91, 1)]

cutsx = [0, 0, 0, 0, 0, 0, 0, 0]
cutsy = [400, 400, 400, 400, 400, 400, 400, 400]


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


fig, ax = plt.subplots(len(shotsday1), 1, figsize=(7, 9), sharex=True)
plt.subplots_adjust(hspace=0., wspace=0.)
for i in range(len(shotsday1)):
    shot = shotsday1[i]
    ax[i].imshow(createpic("20221202", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
    ax[i].set_ylim(800, 1150)
    ax[i].set_xlim(600, 1100)
    tof = get_parameter("20221202", shot, "tof")
    ax[i].text(0.1, 0.1, "TOF = {:} ms".format(tof), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
plt.show()

fig, ax2 = plt.subplots(7, 1, figsize=(7,9), sharex=True)
plt.subplots_adjust(hspace=0., wspace=0.)
for i in range(7):
    shot = shotsday2[i]
    ax2[i].imshow(createpic("20221205", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
    ax2[i].set_ylim(800, 1150)
    ax2[i].set_xlim(600, 1100)
    tof = get_parameter("20221205", shot, "tof")
    ax2[i].text(0.1, 0.1, "TOF = {:} ms".format(tof), horizontalalignment='center', verticalalignment='center', transform=ax2[i].transAxes)
for a in ax2:
    a.set_xticks([])
    a.set_yticks([])
plt.show()

fig, ax2 = plt.subplots(7, 1, figsize=(7, 9), sharex=True)#, sharey=True)
plt.subplots_adjust(hspace=0., wspace=0.)
for i in range(7):
    shot = shotsday2[i+7]
    ax2[i].imshow(createpic("20221205", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
    ax2[i].set_ylim(800, 1150)
    ax2[i].set_xlim(600, 1100)
    tof = get_parameter("20221205", shot, "tof")
    ax2[i].text(0.1, 0.1, "TOF = {:} ms".format(tof), horizontalalignment='center', verticalalignment='center', transform=ax2[i].transAxes)
for a in ax2:
    a.set_xticks([])
    a.set_yticks([])
plt.show()

fig, ax2 = plt.subplots(7, 1, figsize=(7,9), sharex=True)
plt.subplots_adjust(hspace=0., wspace=0.)
for i in range(6):
    shot = shotsday2[i+14]
    ax2[i].imshow(createpic("20221205", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
    ax2[i].set_ylim(800, 1150)
    ax2[i].set_xlim(600, 1100)
    tof = get_parameter("20221205", shot, "tof")
    ax2[i].text(0.1, 0.1, "TOF = {:} ms".format(tof), horizontalalignment='center', verticalalignment='center', transform=ax2[i].transAxes)

shot = 230
ax2[6].imshow(createpic("20221202", shot), vmin=-1, vmax=1.3, cmap='bone', aspect='auto')
ax2[6].set_ylim(800, 1150)
ax2[6].set_xlim(600, 1100)
tof = get_parameter("20221205", shot, "tof")
ax2[6].text(0.1, 0.1, "TOF = 35 ms", horizontalalignment='center', verticalalignment='center', transform=ax2[6].transAxes)

for a in ax2:
    a.set_xticks([])
    a.set_yticks([])
plt.show()
# plt.tight_layout()
