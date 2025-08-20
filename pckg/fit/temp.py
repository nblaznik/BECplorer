
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import mplcursors
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.mplot3d import Axes3D

import scipy.odr as odr
import os
from os.path import exists
import sys
import csv
import astropy.io.fits as pyfits
import pathlib
import datetime
import sip
import numpy as np
from PIL import Image
from collections import Counter
import re
import shutil
import fileinput
import webbrowser
import pathlib

import PyQt5
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtWidgets import *
from typing import List
from PyQt5.QtCore import QModelIndex
from PyQt5.QtCore import pyqtSlot
import pyqtgraph as pg
from pyqtgraph import PlotWidget


date = 20231214
shot = 154

date = 20231215
shot = 1
#

input_folder = f"/storage/data/{date}/{str(shot).zfill(4)}/"
atom = pyfits.open(input_folder + '0.fits')[0].data.astype(float)[0]  # .mean(axis=0)
flat = pyfits.open(input_folder + '1.fits')[0].data.astype(float)[0]  # .mean(axis=0)
dark = pyfits.open(input_folder + '2.fits')[0].data.astype(float)[0]  # .mean(axis=0)

fig, ax = plt.subplots(2,2, sharex=True,sharey=True)
fig.suptitle(f"{date}-{shot}")
ax[0][0].set_title("Atoms")
ax[0][1].set_title("Flat")
ax[1][0].set_title("Dark")
ax[1][1].set_title("PIC")
ax[0][0].imshow(atom)
ax[0][1].imshow(flat)
ax[1][0].imshow(dark)
ax[1][1].imshow((atom-dark)/(flat-dark), interpolation="none", vmin=0, vmax=1.3)

plt.show()