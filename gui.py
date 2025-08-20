"""
# Nejc's Cool Plotting App
# This app was developed in April / May 2021, for the needs of the speed up, and as a fast and a reliable tool of
# analysis to be used in the BEC Lab.

# The functionallity so far includes:
#   - A data viewer and browser, together with previewing the parameters for each run, and previous analysis
#   - Live updating of the data available,
#   - Plotting of any old or new data - together with options of binning, cropping, changing the colorbar range or style
#   - Fitting of any old or new data - for now, implementations of plain viewer, gaussian fits, (Thomas Fermi fit,
#     bimodal fit, OAH - still to be implemented)
#   - Taking linecuts of the data and the fits
#   - Ability of live plotting and fitting the data measured in the lab (based on tracking a .fits file appearing)
#   - Live modifying of the 'constants' (such as the pixel size, detuning, trapping frequencies).
#   - Saving and uploading variables
#   - Saving and loading comments for each run
#   - A nice and neat way of keeping log of what is being done, and saving of the images (perhaps not even necessary)
#   - Save variables every time the app is closed.
#   - Easter Egg, check the gallery. 

"""

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

from pckg.fit.fitfunctions import gaussmod, tfmod, bimodalmod
from pckg.fit import OAH_functions as f1
from pckg.constants import kB, m, hb, asc, mu0, e0
from pckg.fit.OAH_refocus import HI_refocus, preprocessHI_refocus, preprocessHI_refocus_multi
from pckg.fit.coolingOAH import *

_translate = QtCore.QCoreApplication.translate

matplotlib.use("Qt5Agg")


class MplCanvas(FigureCanvas):
    """ Matplotlib Widget. Used for generating the main plot. """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        gs = fig.add_gridspec(3, 2, width_ratios=(7, 2), height_ratios=(7, 1, 1),
                              top=0.9, bottom=0.1, left=0.1, right=0.9,
                              wspace=0.05, hspace=0.05)

        self.ax = fig.add_subplot(gs[0, 0])
        self.ax2 = fig.add_subplot(gs[1:, 0], sharex=self.ax)
        self.ax3 = fig.add_subplot(gs[0, 1], sharey=self.ax)
        self.ax4 = fig.add_subplot(gs[1, 1], sharex=self.ax3, sharey=self.ax2)
        self.ax_in1 = inset_axes(self.ax, width="45%", height="45%", loc=2)
        self.ax_in2 = inset_axes(self.ax, width="45%", height="45%", loc=1)
        self.ax_in3 = inset_axes(self.ax, width="45%", height="45%", loc=8)
        # self.ax_in4 = inset_axes(self.ax, width="45%", height="45%", loc=4)
        super(MplCanvas, self).__init__(fig)
        # fig.tight_layout()


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OAH Analyis - Enter Parameters")
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.inputs = []

        # Focus spinBox
        self.focus_spin = QDoubleSpinBox(self)
        self.focus_spin.setDecimals(3)
        step_type = QAbstractSpinBox.AdaptiveDecimalStepType
        self.focus_spin.setStepType(step_type)
        self.focus_spin.setMinimum(-1)
        self.focus_spin.setMaximum(1)

        # Bin spinBox
        self.bin_spin = QSpinBox(self)
        self.bin_spin.setStepType(1)
        self.bin_spin.setMinimum(1)
        self.bin_spin.setMaximum(30)
        self.bin_spin.setValue(4)

        # Fit mode combobox
        self.combo_box_fitmode = QComboBox(self)
        self.combo_box_fitmode.addItems(["gauss", "tf", "bimodal"])

        # Fit mode quad
        self.combo_box_quad = QComboBox(self)
        self.combo_box_quad.addItems(["quad1", "quad2"])

        # Time_measurement spinBox
        self.t_spin = QSpinBox(self)
        self.t_spin.setStepType(1)
        self.t_spin.setMinimum(1)
        self.t_spin.setMaximum(9999)

        # Ignore numbers
        self.ignorenums = QLineEdit(self)
        self.ignorenums.setText("100")

        # Fitting parameters
        self.cutAreaX = QLineEdit(self)
        self.cutAreaY = QLineEdit(self)
        self.loadButton = QPushButton('Get')

        try:
            coords = np.load("/storage/BECViewer/variables/fittingparamsOAH.npy", allow_pickle=True)
        except:
            coords = np.load("/storage/BECViewer/variables/fittingparams.npy", allow_pickle=True)

        self.splitLayout = QtGui.QGridLayout()
        self.splitLayout.addWidget(self.cutAreaX, 0, 0)
        self.splitLayout.addWidget(self.cutAreaY, 0, 1)
        self.splitLayout.addWidget(self.loadButton, 0, 2)

        self.cutAreaX.setText("{:}, {:}".format(coords[0][0], coords[0][1]))
        self.cutAreaY.setText("{:}, {:}".format(coords[0][2], coords[0][3]))

        # Inputs
        self.inputs.append(self.focus_spin)
        layout.addRow("Focus", self.inputs[-1])
        self.inputs.append(self.combo_box_fitmode)
        layout.addRow("Fit Mode", self.inputs[-1])
        self.inputs.append(self.combo_box_quad)
        layout.addRow("Quad to Cut", self.inputs[-1])
        self.inputs.append(self.bin_spin)
        layout.addRow("Bin", self.inputs[-1])
        self.inputs.append(self.t_spin)
        layout.addRow("Time between images [ms]", self.inputs[-1])
        self.inputs.append(self.ignorenums)
        layout.addRow("Enter numbers to ignore", self.inputs[-1])
        self.inputs.append(self.splitLayout)
        layout.addRow("Coordinates", self.inputs[-1])

        layout.addWidget(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        dz_focus = self.inputs[0].value()
        fit_mode_string = self.inputs[1].currentText()
        quad_string = self.inputs[2].currentText()
        bin_val = self.inputs[3].value()
        time_ms = self.inputs[4].value()
        ignore = [int(num) for num in self.ignorenums.text().split(", ")]
        areaX = self.cutAreaX
        areaY = self.cutAreaY

        return dz_focus, fit_mode_string, quad_string, bin_val, time_ms, ignore, areaX, areaY


class BEC_LIVE_PLOT_APP(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Initial Configuration & Full Screen
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('resources/main.ui', self)
        self.resize(888, 600)
        icon = QtGui.QIcon()
        self.setWindowTitle("BECplorer")
        self.dir_path = self.menuFile.addAction('&Open', self.fileOpen, QtCore.Qt.CTRL + QtCore.Qt.Key_O)  # Menu Open
        self.menuFile.addAction('&Quit', self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)  # Menu Quit
        icon.addPixmap(QtGui.QPixmap("resources/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.threadpool = QtCore.QThreadPool()
        self.showMaximized()  # Start maximized
        self.tabWidget.setCurrentIndex(0)  # Start on the Data Loader Tab
        self.spinBox_updateTime.valueChanged.connect(self.liveUpdate)  # Change the update time
        self.checkBox_autoUpdate.clicked.connect(self.quickLiveUpdate)  # Change the update time
        self.info_bttn.stateChanged.connect(self.displayInfo)  # Display the information about the scan run
        self.copy_bttn.hide()  # Currently have no use for it
        self.saveas_bttn.hide()  # Currently have no use for it

        # Data Loader Tab
        self.checkBox_tablemod.clicked.connect(self.showHide)
        self.pushButton_wildcard.clicked.connect(self.wildcardupdate)  # Generate the wildcard multi
        self.updateDateList()  # Start with the dates loaded
        self.load_date_bttn.clicked.connect(self.updateDateList)  # When click on Update Date, update the date list
        self.listWidget.itemActivated.connect(self.updateRunList)  # When click on the date, update run list
        self.tableWidget.itemChanged.connect(self.saveComment)  # Save comment on change
        self.initialParams()  # Set the initial params for cutting and binning
        self.analyze_bttn.clicked.connect(self.updatePlot)  # When clicked on analyze it generates the plot
        self.pbar.hide()  # Hide the progress bar initially
        self.lineEdit_tof.hide()
        self.lineEdit_thold.hide()
        self.lineEdit_RF.hide()
        self.lineEdit_RF3.hide()
        self.lineEdit_wildcard.hide()
        self.pushButton_wildcard.hide()
        self.checkBox_bbot.clicked.connect(self.bbot)
        self.movie = QMovie("/home/bec_lab/python/BECViewer/resources/signal-wave.gif")
        # self.tableWidget.cellDoubleClicked.connect(self.cell_double_clicked) # Maybe might be useful for something somewhere in the future.


        # Analysis Tab
        self.setCmap()  # Generate a color map interface
        self.hideFrame()  # Start with the frame hidden
        self.textEdit_comment.hide()  # Start with the comment box hidden
        self.buttonBox_saveComment.hide()  # Start with the save button hidden
        #self.pushButton_logBook2.hide()  # Start with the logbook button hidden
        self.menu_checkbttn.clicked.connect(self.frameDisplay)  # Display/hide the main edit frame on the analysis tab
        self.plot_bttn.clicked.connect(self.updatePlot)  # When clicked, update the plot using the cmap vars
        self.radioButton_bin_comb.toggled.connect(self.radioButtonBin)  # Toggle the binning options
        self.radioButton_bin_indiv.toggled.connect(self.radioButtonBin)  # Toggle the binning options
        self.radioButton_nobin.toggled.connect(self.radioButtonBin)  # Toggle the binning options
        self.horizontalSlider.valueChanged[int].connect(self.slider)  # Slider function for navigating images
        self.pushButton_prev.clicked.connect(self.prevImage)  # Go to previous image
        self.pushButton_next.clicked.connect(self.nextImage)  # Go to next image
        self.shortcut_prev = QShortcut(QKeySequence("Alt+,"), self)  # Shortcut for previous image
        self.shortcut_next = QShortcut(QKeySequence("Alt+."), self)  # Shortcut for next image
        self.shortcut_viewer = QShortcut(QKeySequence("Alt+V"), self)  # Shortcut for viewer mode
        self.shortcut_gauss = QShortcut(QKeySequence("Alt+G"), self)  # Shortcut for gaussian mode
        self.shortcut_hideframe = QShortcut(QKeySequence("Alt+F"), self)  # Shortcut for hide/show main analysis frame
        self.shortcut_savecomment = QShortcut(QKeySequence("Alt+S"), self)  # Shortcut for hide/show main analysis frame
        self.shortcut_prev.activated.connect(self.prevImage)  # When previous image shortcut activated
        self.shortcut_next.activated.connect(self.nextImage)  # When next image shortcut activated
        self.shortcut_viewer.activated.connect(self.viewer)  # Select Viewer Mode
        self.shortcut_gauss.activated.connect(self.gauss)  # Select Gauss mode
        self.shortcut_hideframe.activated.connect(self.frameDisplay)  # Hide Frame when clicked on collapse button
        self.shortcut_savecomment.activated.connect(self.saveComment_analyse)  # Hide Frame when clicked on collapse button
        self.pushButton_default.clicked.connect(self.defaultInitialGuess)  # Set default initial guess for fitting
        self.pushButton_default_cut.clicked.connect(self.initialParams)  # Set default initial guess for fitting
        self.checkBox_comment.stateChanged.connect(self.showComment)  # Show comment button
        self.buttonBox_saveComment.clicked.connect(self.saveComment_analyse)  # Save comment
        #self.pushButton_logBook2.clicked.connect(self.logBook) # Open Log Book
        self.pushButton_OAHcmap.clicked.connect(self.setCmapOAH)  # Set cmap for OAH
        self.pushButton_TIcmap.clicked.connect(self.setCmap)  # Set cmap for TI
        self.pushButton_fromimg.clicked.connect(self.fittingParamsFromImage)  # Get fitting parameters from image selection.
        self.pushButton_fromimg_cut.clicked.connect(self.cutParamsFromImage)  # Get fitting parameters from image selection.
        self.spinBox_multiimage.hide()          ## Hide OAH stuff
        self.checkBox_multi_img.hide()
        self.spinBox_num.hide()
        self.checkBox_singleNum.hide()
        self.checkBox_OAHFit.hide()
        self.radioButton_oah1.hide()
        self.radioButton_oah2.hide()
        self.radioButton_ang.hide()
        self.radioButton_amp.hide()
        self.checkBox_singleNum.stateChanged.connect(self.updateMaxVal)
        self.comboBox_mode.currentTextChanged.connect(self.OAH_mode_display)

        # Variables Tab
        self.paramsLoadStart()  # Load parameters
        self.pushButton_update_vars.clicked.connect(self.setVariables)  # Set the currently set variables
        self.pushButton_upload_vars.clicked.connect(self.paramsLoad)  # Set the currently set variables
        self.pushButton_save_vars.clicked.connect(self.paramsSave)  # Save the currently set variables

        # Further Analysis
        # No big use, use external tools, so disable it for now.
        self.tabWidget.setTabVisible(3, False)
        self.pushButton_clear.clicked.connect(self.clearFurtherAnalysis)  # Clear further analysis
        self.pushButton_oldAnalyse.clicked.connect(self.oldAnalysis)  # Load old analysis script and execute it
        self.pushButton_focusCompare.clicked.connect(self.HI_refocus_preview)  # Load old HI refocus script
        self.pushButton_toFits.clicked.connect(self.makeAFits)  # Make a fits file from a collection
        self.pushButton_FA_loaddates.clicked.connect(self.setDatesFA)  # Load dates in further analysis
        self.pushButton_FA_loadruns.clicked.connect(self.setRunsFA)  # Load runs in further analysis

        # Gallery
        self.setDatesGallery()
        self.pushButton_fetchDates.clicked.connect(self.setDatesGallery)
        self.pushButton_makeGallery.clicked.connect(self.galleryDisplay)

        # Search
        # It's lagging so let's hide it for now.
        # self.tabWidget.setTabVisible(5, False)
        self.searchButton.clicked.connect(self.searchFunction)
        self.searchLineEdit.returnPressed.connect(self.searchFunction)
        self.searchListWidget.doubleClicked.connect(self.takeMeToComment)  # When click on the date, update run list

        # Easter Egg
        self.sc = 0  # Secret Counter
        self.sc2 = 0  # Secret Counter 2
        self.pushButton_secret.clicked.connect(self.secretFunction)
        self.copy_bttn.clicked.connect(self.secretFunction2)
        self.saveas_bttn.clicked.connect(self.secretFunction3)


    # --------------------------------------------
    # -------------- Menu Functions --------------
    # --------------------------------------------s

    def fileOpen(self):
        # Doesn't quite work, I can't open directories, only files. Odd.
        self.dir_path = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Directory"))
        print(self.dir_path)
        return self.dir_path

    # def closeEvent(self, event):   ## A bit annoying so I disabled it.
    #     reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
    #                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #     if reply == QMessageBox.Yes:
    #         event.accept()
    #         """ Auto save the parameters specified on exit to /storage/BECViewer/autosave. """
    #         path = '/storage/BECViewer/autosave'
    #         date = datetime.datetime.now()
    #         datestring = "vars_autosave_" + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(
    #             2) + "_" + str(
    #             date.hour) + str(date.minute) + ".param"
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         with open(path + "/" + datestring, 'w') as file:
    #             csvwriter = csv.writer(file, delimiter=';', quotechar='"')
    #             csvwriter.writerow([0, self.label_radfreq.text(), self.lineEdit_radfreq.text()])
    #             csvwriter.writerow([1, self.label_axfreq.text(), self.lineEdit_axfreq.text()])
    #             csvwriter.writerow([2, self.label_wavelength.text(), self.lineEdit_wavelength.text()])
    #             csvwriter.writerow([3, self.label_detuning.text(), self.lineEdit_detuning.text()])
    #             csvwriter.writerow([4, self.label_pixelsize.text(), self.lineEdit_pixelsize.text()])
    #         self.close()
    #     else:
    #         event.ignore()

    def paramsSave(self):
        """ Saves the parameters specified on the Variable tab - for future loading. """
        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', "/storage/BECViewer/variables/", filter=".csv")
        with open(name[0], 'w') as file:
            csvwriter = csv.writer(file, delimiter=';', quotechar='"')
            csvwriter.writerow([0, self.label_radfreq.text(), self.lineEdit_radfreq.text()])
            csvwriter.writerow([1, self.label_axfreq.text(), self.lineEdit_axfreq.text()])
            csvwriter.writerow([2, self.label_wavelength.text(), self.lineEdit_wavelength.text()])
            csvwriter.writerow([3, self.label_detuning.text(), self.lineEdit_detuning.text()])
            csvwriter.writerow([4, self.label_pixelsize.text(), self.lineEdit_pixelsize.text()])

    def paramsLoad(self):
        """ Loading variables. """
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        vars = []
        with open(name[0]) as varsfile:
            csvreader = csv.reader(varsfile, delimiter=';')
            for row in csvreader:
                vars.append(row[2])
        self.lineEdit_radfreq.setText(vars[0])
        self.lineEdit_axfreq.setText(vars[1])
        self.lineEdit_wavelength.setText(vars[2])
        self.lineEdit_detuning.setText(vars[3])
        self.lineEdit_pixelsize.setText(vars[4])
        self.setVariables()

    def paramsLoadStart(self):
        """ Automatically upload the most recent one on start. """
        autosavefiles = []
        for asfile in os.listdir("/storage/BECViewer/autosave/"):
            if asfile.startswith("vars_autosave_"):
                autosavefiles.append(asfile)
        autosavefiles.sort(reverse=True)
        vars = []
        with open("/storage/BECViewer/autosave/" + autosavefiles[0]) as varsfile:
            csvreader = csv.reader(varsfile, delimiter=';')
            for row in csvreader:
                vars.append(row[2])
        self.lineEdit_radfreq.setText(vars[0])
        self.lineEdit_axfreq.setText(vars[1])
        self.lineEdit_wavelength.setText(vars[2])
        self.lineEdit_detuning.setText(vars[3])
        self.lineEdit_pixelsize.setText(vars[4])
        self.setVariables()

    def fittingParamsFromImage(self):
        self.paramsFromImg(mode="fit", setParams=True)

    def cutParamsFromImage(self):
        self.paramsFromImg(mode="cut", setParams=True)

    def getCutParamsOAH(self):
        date = BEC_LIVE_PLOT_APP.tableWidget.selectedItems()[0].text()
        run = BEC_LIVE_PLOT_APP.tableWidget.selectedItems()[1].text()
        os.system('python3 pckg/fit/coords_from_image.py {:} {:} -M OAHcooling'.format(date, run))
        coords = np.load("/storage/BECViewer/variables/fittingparamsOAH.npy", allow_pickle=True)
        # coords = [[x1, x2, y1, y2], [centerx, centery], [wx, wy]]
        InputDialog.cutAreaX.setText("{:}, {:}".format(coords[0][0], coords[0][1]))
        InputDialog.cutAreaY.setText("{:}, {:}".format(coords[0][2], coords[0][3]))
        InputDialog.center.setText("{:}, {:}".format(coords[1][0], coords[1][1]))

    def paramsFromImg(self, mode, setParams=False):
        date = self.tableWidget.selectedItems()[0].text()
        run = self.tableWidget.selectedItems()[1].text()
        os.system('python3 pckg/fit/coords_from_image.py {:} {:} -M normal'.format(date, run))
        coords = np.load("/storage/BECViewer/variables/fittingparams.npy", allow_pickle=True)

        # coords = [[x1, x2, y1, y2], [centerx, centery], [wx, wy]]
        print(coords)

        if mode == "fit":
            if setParams:
                self.lineEdit_center.setText("{:}, {:}".format(str(int(coords[1][0])), str(int(coords[1][1]))))
                self.lineEdit_angle.setText("0")
                self.lineEdit_ga.setText("1")
                self.lineEdit_gw.setText("{:}, {:}".format(str(int(coords[2][0])), str(int(coords[2][1]))))
                self.lineEdit_tfa.setText("1")
                self.lineEdit_tfw.setText("{:}, {:}".format(str(int(coords[2][0])), str(int(coords[2][1]))))
            else:
                return coords
            
        if mode == "cut":
            if setParams:
                self.lineEdit_cut_xi.setText("{:}".format(str(int(coords[0][0]))))
                self.lineEdit_cut_xf.setText("{:}".format(str(int(coords[0][1])-int(coords[0][0]))))
                self.lineEdit_cut_zi.setText("{:}".format(str(int(coords[0][2]))))
                self.lineEdit_cut_zf.setText("{:}".format(str(int(coords[0][3])-int(coords[0][2]))))

            else:
                return coords

    # --------------------------------------------
    # --------------- Data Loader ----------------
    # --------------------------------------------

    def updateDateList(self):
        """ Generate the directory list of all the dates and sort it. """
        if isinstance(self.dir_path, str):
            self.path = self.dir_path
        else:
            self.path = "/storage/data/"
        self.dirList = [x for x in os.listdir(self.path) if len(x) == 8]  # Only detect folders with length of 8 (YYYYMMDD)
        self.listWidget.clear()
        self.listWidget.addItems(self.dirList)
        self.listWidget.sortItems(QtCore.Qt.DescendingOrder)  # In descending order

    def wildcardupdate(self):
        date = self.listWidget.currentItem().text()
        self.updateRunList(date)

    def updateRunList(self, item):
        """ Generate the directory table of all the runs for a set date with parameters. """
        if isinstance(item, str):
            date = item
        else:
            date = item.text()
        # wildcard = self.comboBox_wildcard.currentText()
        path = "/storage/data/" + str(date) + "/"
        self.dirList = [x for x in os.listdir(path) if os.path.isdir(path + x)]

        if self.checkBox_favourites.isChecked():
            # Get numbers where comments were marked with exclamation mark(s)
            favs = self.getFavourites(date, self.dirList)
            self.dirList = favs


        self.tableWidget.setRowCount(0)
        self.Loading = False
        self.tableWidget.setSortingEnabled(False)
        self.wc_string = self.lineEdit_wildcard.text()
        self.RF_string = self.lineEdit_RF.text()
        self.RF3_string = self.lineEdit_RF3.text()
        self.thold_string = self.lineEdit_thold.text()
        self.tof_string = self.lineEdit_tof.text()

        # self.tableWidget.setHorizontalHeaderLabels(['Date', 'Run_ID', 'Analysis Done', 'tof', 'thold', 'RF', 'RF3', wildcard, 'Particle Number', 'Comments'])
        for run in self.dirList:
            rowPosition = self.tableWidget.rowCount()
            self.tableWidget.insertRow(rowPosition)
            analys = str(self.get_analysis(date, run))
            rftime = str(self.get_parameter(date, run, self.RF_string))
            rftime3 = str(self.get_parameter(date, run, self.RF3_string))
            thold = str(self.get_parameter(date, run, self.thold_string))
            tof = str(self.get_parameter(date, run, self.tof_string))
            wildcard = str(self.get_parameter(date, run, self.wc_string))
            ntherm, clr = self.get_n(date, run)
            comment = str(self.get_comment(date, run))
            # QtWidgets.QTableWidgetItem(str(run)).setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(rowPosition, 0, QtWidgets.QTableWidgetItem(date))  # Date
            self.tableWidget.setItem(rowPosition, 1, QtWidgets.QTableWidgetItem(str(run)))  # Run ID
            self.tableWidget.setItem(rowPosition, 2, QtWidgets.QTableWidgetItem(analys))  # Previous analysis
            self.tableWidget.setItem(rowPosition, 3, QtWidgets.QTableWidgetItem(tof))  # tof
            self.tableWidget.setItem(rowPosition, 4, QtWidgets.QTableWidgetItem(thold))  # thold
            self.tableWidget.setItem(rowPosition, 5, QtWidgets.QTableWidgetItem(rftime))  # rftime
            self.tableWidget.setItem(rowPosition, 6, QtWidgets.QTableWidgetItem(rftime3))  # rftime3
            self.tableWidget.setItem(rowPosition, 7, QtWidgets.QTableWidgetItem(wildcard))  # wildcard (default t_light)
            self.tableWidget.setItem(rowPosition, 8, QtWidgets.QTableWidgetItem(ntherm))  # ntherm
            self.tableWidget.setItem(rowPosition, 9, QtWidgets.QTableWidgetItem(comment))  # comment
            self.tableWidget.item(rowPosition, 8).setBackground(clr)
            header = self.tableWidget.horizontalHeader()
            for i in range(10):
                header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.setSortingEnabled(True)
        self.Loading = True
        self.tableWidget.sortItems(1, QtCore.Qt.DescendingOrder)
        self.tableWidget.setHorizontalHeaderLabels(['Date', 'Run_ID', 'Analysis', self.tof_string, self.thold_string, self.RF_string, self.RF3_string, self.wc_string, 'Particle Number', 'Comments'])

    def getFavourites(self, date, r_list):
        runlist = sorted(r_list, reverse=True)
        favs = []
        for run in runlist:
            """ Fetch the comment, if there is one """
            if os.path.exists("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run)):
                with open("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run), "r") as commentFile:
                    comment = commentFile.read()
                    if len(comment) > 0 and comment[0] == "!":
                        favs.append(run)
        return favs

    def logBook(self):
        """ Still in dev. """
        # date = self.listWidget.selectedItems()[0].text()
        webbrowser.open('../LogBrowser/index.html')

    def contextMenuEvent(self, event):
        """
        Right-Click on the menu of runs. To plot the number of particles vs one of the variables.
        Or, autofits a set of data, and gets parameters. Can take a while. Program freezes in-between.
        """
        self.menu = QtGui.QMenu(self)
        findMenu = self.menu.addMenu("Particle vs ...")
        tempMenu = self.menu.addMenu("Temperature vs ...")
        groupAnalyse = self.menu.addMenu("Group analysis")
        color = self.menu.addMenu("Mark With Color")
        FixRF = self.menu.addMenu("Fix RF")
        effMenu = self.menu.addMenu("Efficiency Tools")

        runaction = QtGui.QAction('Run', self)
        rfaction = QtGui.QAction(self.RF_string, self)
        tholdaction = QtGui.QAction(self.thold_string, self)
        tofaction = QtGui.QAction(self.tof_string, self)
        varaction = QtGui.QAction(self.wc_string, self)
        timeaction = QtGui.QAction('Time', self)

        runactionT = QtGui.QAction('Run', self)
        rfactionT = QtGui.QAction(self.RF_string, self)
        tholdactionT = QtGui.QAction(self.thold_string, self)
        tofactionT = QtGui.QAction(self.tof_string, self)
        varactionT = QtGui.QAction(self.wc_string, self)

        gaussanalysis = QtGui.QAction('Gaussian Fits', self)
        bimodalanalysis = QtGui.QAction('Bimodal Fits', self)

        markblue = QtGui.QAction('Blue', self)
        markred = QtGui.QAction('Red', self)
        markgreen = QtGui.QAction('Green', self)

        liveplot = QtGui.QAction('Live', self)
        fixPnum = QtGui.QAction('Empty ...', self)
        ignore = QtGui.QAction('Ignore', self)
        unignore = QtGui.QAction('Unignore all', self)

        loglog = QtGui.QAction('Log-Log Plot', self)
        OAHloglog = QtGui.QAction('OAH Log-Log Plot', self)
        TCanalysis = QtGui.QAction('TIME CRYSTAL analysis', self)

        RF1only = QtGui.QAction('RF1 Only', self)
        RF1RF2 = QtGui.QAction('RF1 + RF2', self)
        Reset = QtGui.QAction('Reset', self)

        findMenu.addAction(runaction)
        findMenu.addAction(rfaction)
        findMenu.addAction(tholdaction)
        findMenu.addAction(tofaction)
        findMenu.addAction(varaction)
        findMenu.addAction(timeaction)

        tempMenu.addAction(runactionT)
        tempMenu.addAction(rfactionT)
        tempMenu.addAction(tholdactionT)
        tempMenu.addAction(tofactionT)
        tempMenu.addAction(varactionT)

        groupAnalyse.addAction(gaussanalysis)
        groupAnalyse.addAction(bimodalanalysis)

        color.addAction(markblue)
        color.addAction(markred)
        color.addAction(markgreen)

        FixRF.addAction(RF1only)
        FixRF.addAction(RF1RF2)
        FixRF.addAction(Reset)

        effMenu.addAction(loglog)
        effMenu.addAction(OAHloglog)

        self.menu.addAction(liveplot)
        self.menu.addAction(ignore)
        self.menu.addAction(unignore)
        self.menu.addAction(fixPnum)

        runaction.triggered.connect(lambda: self.runVsNum(event))
        rfaction.triggered.connect(lambda: self.RFvsNum(event))
        tholdaction.triggered.connect(lambda: self.TholdvsNum(event))
        tofaction.triggered.connect(lambda: self.TofvsNum(event))
        varaction.triggered.connect(lambda: self.VarvsNum(event))
        timeaction.triggered.connect(lambda: self.TimevsNum(event))

        runactionT.triggered.connect(lambda: self.runVsTemp(event))
        rfactionT.triggered.connect(lambda: self.RFvsTemp(event))
        tholdactionT.triggered.connect(lambda: self.TholdvsTemp(event))
        tofactionT.triggered.connect(lambda: self.TofvsTemp(event))
        varactionT.triggered.connect(lambda: self.VarvsTemp(event))

        gaussanalysis.triggered.connect(lambda: self.groupGaussAnalysis(event))
        bimodalanalysis.triggered.connect(lambda: self.groupBimodalAnalysis(event))
        markblue.triggered.connect(lambda: self.markblue(event))
        markred.triggered.connect(lambda: self.markred(event))
        markgreen.triggered.connect(lambda: self.markgreen(event))

        RF1only.triggered.connect(lambda: self.fixRF(event, "RF1only"))
        RF1RF2.triggered.connect(lambda: self.fixRF(event, "RF1RF2"))
        Reset.triggered.connect(lambda: self.fixRF(event, "Reset"))


        liveplot.triggered.connect(lambda: self.livePlot(event))
        fixPnum.triggered.connect(lambda: self.fixPnum(event))
        ignore.triggered.connect(lambda: self.ignore(event))
        unignore.triggered.connect(lambda: self.unignore(event))
        loglog.triggered.connect(lambda: self.loglog(event))
        OAHloglog.triggered.connect(lambda: self.OAHloglog(event))
        TCanalysis.triggered.connect(lambda: self.timeCrystalAnalysis(event))

        # add other required actions
        self.menu.popup(QtGui.QCursor.pos())

    def runVsNum(self, event):
        indexes = self.tableWidget.selectionModel().selectedRows() # For the colors here.
        selected = self.tableWidget.selectedItems() # For the elements here

        # Check for ignore
        date_selection = self.tableWidget.item(0, 0).text()
        if exists("/storage/BECViewer/ignore/{:}".format(date_selection)):
            ignore = []
            with open("/storage/BECViewer/ignore/{:}".format(date_selection), "r") as f:
                for line in f:
                    ignore.append(int(line.strip()))
            ignore = sorted(list(dict.fromkeys(ignore)))  # eliminate duplicates
            print(ignore)
            indexes = [x for x in indexes if int(self.tableWidget.item(x.row(), 1).text()) not in ignore]
            selected = [x for x in selected if int(self.tableWidget.item(x.row(), 1).text()) not in ignore]

        # Check for colors
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        # Get numbers and runs here.
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        nums = [float(i.split()[0]) for i in elements[8::10]] 
        # runs, nums = zip(*sorted(zip(runs, nums)))
        runs, nums, colors = zip(*sorted(zip(runs, nums, colors)))
        plt.plot(runs, nums, c='r', ls='--', alpha=0.3)
        plt.scatter(runs, nums, c=colors)
        plt.title("Number of particles over runs.")
        plt.xlabel('RUN')
        plt.ylabel('Particle Number')
        plt.show()

    def RFvsNum(self, event):
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())
        RF12 = [float(i) for i in elements[5::10]]
        RF3 = []
        for i in elements[6::10]:
            if i != "N/A":
                RF3.append(float(i))
            else:
                RF3.append(0)
        RF = RF12 + RF3
        nums = [float(i.split()[0]) for i in elements[8::10]]
        RF, nums = zip(*sorted(zip(RF, nums)))
        plt.plot(RF, nums, c='r', ls='--', alpha=0.3)
        plt.scatter(RF, nums)
        plt.title("Number of particles vs {:}".format(self.RF_string))
        plt.xlabel(self.RF_string)
        plt.ylabel('Particle Number')
        plt.show()

    def fixRF(self, event, mode):
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        date = elements[0::10][0]
        for i in runs:
            if mode == "RF1only":
                rf1 = self.get_parameter(date, i, 'rftime1')
                rftotal = self.get_parameter(date, i, 'rftime')
                print(rf1, rftotal)
                if rf1 != rftotal:
                    print(rftotal, "Not the same as", rf1)
                    src = "/storage/data/{:}/{:}/parameters.param".format(date, str(i).zfill(4))
                    dst = "/storage/data/{:}/{:}/parameters_mod.param".format(date, str(i).zfill(4))
                    # Copy parameters.param to parameters_old.param
                    shutil.copyfile(src, dst)
                    # Add RF2 = 0, RF3 = 0 to the end (would that work)
                    with open(dst, 'a') as paramfile:
                        paramfile.write("rftime,{:}\n".format(rf1))
                        paramfile.write("rftime2,0\n")
                        paramfile.write("rftime3,0")

                print("Saved")


            elif mode == "RF1RF2":
                rf1 = int(self.get_parameter(date, i, 'rftime1'))
                rf2 = int(self.get_parameter(date, i, 'rftime2'))
                rftotal = int(self.get_parameter(date, i, 'rftime'))
                print(rf1, rf2, rftotal)
                if (rf1 + rf2) != rftotal:
                    src = "/storage/data/{:}/{:}/parameters.param".format(date, str(i).zfill(4))
                    dst = "/storage/data/{:}/{:}/parameters_mod.param".format(date, str(i).zfill(4))
                    # Copy parameters.param to parameters_old.param
                    shutil.copyfile(src, dst)
                    rf1 = self.get_parameter(date, i, 'rftime1')
                    with open(dst, 'a') as paramfile:
                        paramfile.write("rftime,{:}\n".format(rf1 + rf2))
                        paramfile.write("rftime3,0")

            elif mode == "Reset":
                os.remove("/storage/data/{:}/{:}/parameters_mod.param".format(date, str(i).zfill(4)))

    def showHide(self, state):
        if state:
            self.lineEdit_tof.show()
            self.lineEdit_thold.show()
            self.lineEdit_RF.show()
            self.lineEdit_RF3.show()
            self.lineEdit_wildcard.show()
            self.pushButton_wildcard.show()

        else:
            self.lineEdit_tof.hide()
            self.lineEdit_thold.hide()
            self.lineEdit_RF.hide()
            self.lineEdit_RF3.hide()
            self.lineEdit_wildcard.hide()
            self.pushButton_wildcard.hide()

    def TholdvsNum(self, event):
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())
        thold = [float(i) for i in elements[4::10]]
        nums = [float(i.split()[0]) for i in elements[8::10]]
        thold, nums = zip(*sorted(zip(thold, nums)))
        plt.plot(thold, nums, c='r', ls='--', alpha=0.3)
        plt.scatter(thold, nums)
        plt.title("Number of particles vs {:}".format(self.thold_string))
        plt.xlabel(self.thold_string)
        plt.ylabel('Particle Number')
        plt.show()

    def TofvsNum(self, event):
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())
        tof = [float(i) for i in elements[3::10]]
        nums = [float(i.split()[0]) for i in elements[8::10]]
        tof, nums = zip(*sorted(zip(tof, nums)))
        plt.plot(tof, nums, c='r', ls='--', alpha=0.3)
        plt.scatter(tof, nums)
        plt.title("Number of particles vs {:}".format(self.tof_string))
        plt.xlabel(self.tof_string)
        plt.ylabel('Particle Number')
        plt.show()

    def VarvsNum(self, event):
        # Check for colors
        indexes = self.tableWidget.selectionModel().selectedRows() # For the colors here.
        selected = self.tableWidget.selectedItems() # For the elements here

        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())
        var = [float(i) for i in elements[7::10]]
        nums = [float(i.split()[0]) for i in elements[8::10]]
        var, nums, colors = zip(*sorted(zip(var, nums, colors)))
        # plt.plot(var, nums, c='r', ls='--', alpha=0.3)
        plt.scatter(var, nums, c=colors)
        plt.title("Number of particles vs {:}".format(self.wc_string))
        plt.xlabel(self.wc_string)
        plt.ylabel('Particle Number')
        plt.show()

    def TimevsNum(self, event):

        indexes = self.tableWidget.selectionModel().selectedRows() # For the colors here.
        selected = self.tableWidget.selectedItems() # For the elements here
        ignore = []

        # Check for ignore
        date_selection = self.tableWidget.item(0, 0).text()
        if exists("/storage/BECViewer/ignore/{:}".format(date_selection)):
            with open("/storage/BECViewer/ignore/{:}".format(date_selection), "r") as f:
                for line in f:
                    ignore.append(int(line.strip()))
            ignore = sorted(list(dict.fromkeys(ignore)))  # eliminate duplicates
            print(ignore)
            indexes = [x for x in indexes if int(self.tableWidget.item(x.row(), 1).text()) not in ignore]
            selected = [x for x in selected if int(self.tableWidget.item(x.row(), 1).text()) not in ignore]

        # Check for colors
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        # Get numbers and runs here.
        elements = []
        for wid in selected:
            elements.append(wid.text())


        runs = [str(i) for i in elements[1::10]]
        date = int(elements[0::10][0])
        nums = [float(i.split()[0]) for i in elements[8::10]]

        time_creations = [pathlib.Path(f"/storage/data/{date}/{str(run).zfill(4)}/.runfinished").stat().st_ctime for run in runs if run not in ignore]
        dts = [datetime.datetime.fromtimestamp(tc) for tc in time_creations]


        # runs, nums = zip(*sorted(zip(runs, nums)))
        dts, nums, colors = zip(*sorted(zip(dts, nums, colors)))
        plt.plot(dts, nums, c='r', ls='--', alpha=0.3)
        plt.scatter(dts, nums, c=colors)
        plt.title("Number of particles over time.")
        plt.xlabel('Time')
        plt.ylabel('Particle Number')
        plt.show()

    def runVsTemp(self, event):
        # Get colors here.
        indexes = self.tableWidget.selectionModel().selectedRows()
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        # Get numbers and runs here.
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        dates = [int(i) for i in elements[0::10]]
        tempx = []
        tempz = []
        for run in runs:
            tempx.append(self.get_temp(dates[0], run)[0])
            tempz.append(self.get_temp(dates[0], run)[1])

        runs, tempx, tempz, colors = zip(*sorted(zip(runs, tempx, tempz, colors)))
        plt.plot(runs, tempx, c='r', ls='--', alpha=0.3, label="Temperature-x")
        plt.scatter(runs, tempx, c=colors)
        plt.plot(runs, tempz, c='b', ls=':', alpha=0.3, label="Temperature-z")
        plt.scatter(runs, tempz, c=colors)
        plt.title("Temperature over runs.")
        plt.xlabel('RUN')
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.show()

    def RFvsTemp(self, event):
        # Get colors here.
        indexes = self.tableWidget.selectionModel().selectedRows()
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        dates = [int(i) for i in elements[0::10]]
        tempx = []
        tempz = []
        for run in runs:
            tempx.append(self.get_temp(dates[0], run)[0])
            tempz.append(self.get_temp(dates[0], run)[1])

        RF12 = [float(i) for i in elements[5::10]]
        RF3 = []
        for i in elements[6::10]:
            if i != "N/A":
                RF3.append(float(i))
            else:
                RF3.append(0)
        RF = RF12 + RF3

        RF, tempx, tempz, colors = zip(*sorted(zip(RF, tempx, tempz, colors)))

        plt.plot(RF, tempx, c='r', ls='--', alpha=0.3, label="Temperature-x")
        plt.scatter(RF, tempx, c=colors)
        plt.plot(RF, tempz, c='b', ls=':', alpha=0.3, label="Temperature-z")
        plt.scatter(RF, tempz, c=colors)

        plt.title("Temperature vs {:}".format(self.RF_string))
        plt.xlabel(self.RF_string)
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.show()

    def TholdvsTemp(self, event):
        # Get colors here.
        indexes = self.tableWidget.selectionModel().selectedRows()
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        dates = [int(i) for i in elements[0::10]]
        tempx = []
        tempz = []
        for run in runs:
            tempx.append(self.get_temp(dates[0], run)[0])
            tempz.append(self.get_temp(dates[0], run)[1])

        thold = [float(i) for i in elements[4::10]]

        thold, tempx, tempz, colors = zip(*sorted(zip(thold, tempx, tempz, colors)))

        plt.plot(thold, tempx, c='r', ls='--', alpha=0.3, label="Temperature-x")
        plt.scatter(thold, tempx, c=colors)
        plt.plot(thold, tempz, c='b', ls=':', alpha=0.3, label="Temperature-z")
        plt.scatter(thold, tempz, c=colors)

        plt.title("Number of particles vs {:}".format(self.thold_string))
        plt.xlabel(self.thold_string)
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.show()

    def TofvsTemp(self, event):
        # Get colors here.
        indexes = self.tableWidget.selectionModel().selectedRows()
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        dates = [int(i) for i in elements[0::10]]
        tempx = []
        tempz = []
        for run in runs:
            tempx.append(self.get_temp(dates[0], run)[0])
            tempz.append(self.get_temp(dates[0], run)[1])

        tof = [float(i) for i in elements[3::10]]

        tof, tempx, tempz, colors = zip(*sorted(zip(tof, tempx, tempz, colors)))

        plt.plot(tof, tempx, c='r', ls='--', alpha=0.3, label="Temperature-x")
        plt.scatter(tof, tempx, c=colors)
        plt.plot(tof, tempz, c='b', ls=':', alpha=0.3, label="Temperature-x")
        plt.scatter(tof, tempz, c=colors)

        plt.title("Temperature vs {:}".format(self.tof_string))
        plt.xlabel(self.tof_string)
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.show()

    def VarvsTemp(self, event):
        # Get colors here.
        indexes = self.tableWidget.selectionModel().selectedRows()
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        dates = [int(i) for i in elements[0::10]]
        tempx = []
        tempz = []
        for run in runs:
            tempx.append(self.get_temp(dates[0], run)[0])
            tempz.append(self.get_temp(dates[0], run)[1])

        var = [float(i) for i in elements[7::10]]

        var, tempx, tempz, colors = zip(*sorted(zip(var, tempx, tempz, colors)))

        plt.plot(var, tempx, c='r', ls='--', alpha=0.3, label="Temperature-x")
        plt.scatter(var, tempx, c=colors)
        plt.plot(var, tempz, c='b', ls=':', alpha=0.3, label="Temperature-x")
        plt.scatter(var, tempz, c=colors)

        plt.title("Temperature vs {:}".format(self.wc_string))
        plt.xlabel(self.wc_string)
        plt.ylabel('Temperature [K]')
        plt.legend()
        plt.show()

    def groupGaussAnalysis(self, event):
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())
        runs = elements[1::10]  # The runs to be analysed
        date = elements[0]  # Date to be analysed
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.label_info2.setText("Analysing now: 0 %")
        count = 1
        for run in runs:
            perc = (1 - (len(runs) - count) / len(runs)) * 100
            os.system('python3 ../analyze.py {:} {:} -M gauss --generatepic False'.format(date, run))
            self.label_info2.setText("Analysing now: {:.2f} %".format(perc))
            app.processEvents()
            count += 1
        QtWidgets.QApplication.restoreOverrideCursor()

    def groupBimodalAnalysis(self, event):
        selected = self.tableWidget.selectedItems()
        elements = []
        for wid in selected:
            elements.append(wid.text())
        runs = elements[1::10]  # The runs to be analysed
        date = elements[0]  # Date to be analysed
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.label_info2.setText("Analysing now: 0 %")
        count = 1
        for run in runs:
            perc = (1 - (len(runs) - count) / len(runs)) * 100
            os.system('python3 ../analyze.py {:} {:} -M bimodal --generatepic False'.format(date, run))
            self.label_info2.setText("Analysing now: {:.2f} %".format(perc))
            app.processEvents()
            count += 1
        QtWidgets.QApplication.restoreOverrideCursor()

    def markblue(self, event):
        indexes = self.tableWidget.selectionModel().selectedRows()
        for rowIndex in indexes:
            for j in range(self.tableWidget.columnCount()):
                self.tableWidget.item(rowIndex.row(), j).setBackground(QtGui.QColor(156, 167, 249))

    def markred(self, event):
        indexes = self.tableWidget.selectionModel().selectedRows()
        for rowIndex in indexes:
            for j in range(self.tableWidget.columnCount()):
                self.tableWidget.item(rowIndex.row(), j).setBackground(QtGui.QColor(249, 161, 156))
    
    def markgreen(self, event):
        indexes = self.tableWidget.selectionModel().selectedRows()
        for rowIndex in indexes:
            for j in range(self.tableWidget.columnCount()):
                self.tableWidget.item(rowIndex.row(), j).setBackground(QtGui.QColor(156, 249, 161))

    def livePlot(self, event):
        try:
            elements = [wid.text() for wid in self.tableWidget.selectedItems()]
            startrun = elements[1::10][0]
            startdate = elements[0::10][0]
            print(startrun)
            command = f"python3 ~/python/BECViewer/pckg/live_plot.py -D {startdate} -R {startrun}"
        except:
            command = f"python3 ~/python/BECViewer/pckg/live_plot.py"

        os.system("gnome-terminal -e 'bash -c \"" + command + ";bash\"'")

    def fixPnum(self, event):
        """
            When the fit of an empty image gives something silly and large, this sets the number to zero. Only works with a gaussian fit.
        """
        indexes = self.tableWidget.selectionModel().selectedRows()
        date_selection = self.tableWidget.item(0, 0).text()
        for rowIndex in indexes:
            for mode in ["gauss", "bimodal", "tf"]:
                filename = f"/storage/data/{date_selection}/{str(self.tableWidget.item(rowIndex.row(), 1).text()).zfill(4)}/fit_{mode}.param"
                if os.path.exists(filename):
                    for line in fileinput.input(filename, inplace=True):
                        if line[:6] == "ntherm":
                            print('ntherm,0\n', end='')  # for Python 3
                        elif line[:3] == "ntf":
                            print('ntf,0\n', end='')  # for Python 3
                        elif line[:6] == "ntotal":
                            print('ntotal,0\n', end='')  # for Python 3
                        elif line[:2] == "tx":
                            print('tx,N/A\n', end='')  # for Python 3
                        elif line[:2] == "tz":
                            print('tx,N/A\n', end='')  # for Python 3
                        else:
                            print(line, end='')

    def ignore(self, event):
        indexes = self.tableWidget.selectionModel().selectedRows()
        date_selection = self.tableWidget.item(0, 0).text()
        with open("/storage/BECViewer/ignore/{:}".format(date_selection), "a") as f:
            for rowIndex in indexes:
                f.write(str(self.tableWidget.item(rowIndex.row(), 1).text()) + "\n")

    def unignore(self, event):
        date_selection = self.tableWidget.item(0, 0).text()
        if exists("/storage/BECViewer/ignore/{:}".format(date_selection)):
            os.remove("/storage/BECViewer/ignore/{:}".format(date_selection))
            print("File deleted successfully.")
        else:
            print("No ignores found.")

    def loglog(self, event):
        indexes = self.tableWidget.selectionModel().selectedRows()  # For the colors here.
        selected = self.tableWidget.selectedItems()  # For the elements here

        # Check for ignore
        date_selection = self.tableWidget.item(0, 0).text()
        if exists("/storage/BECViewer/ignore/{:}".format(date_selection)):
            ignore = []
            with open("/storage/BECViewer/ignore/{:}".format(date_selection), "r") as f:
                for line in f:
                    ignore.append(int(line.strip()))
            ignore = sorted(list(dict.fromkeys(ignore)))  # eliminate duplicates
            print(ignore)
            indexes = [x for x in indexes if int(self.tableWidget.item(x.row(), 1).text()) not in ignore]
            selected = [x for x in selected if int(self.tableWidget.item(x.row(), 1).text()) not in ignore]

        # Check for colors
        colors = []
        for rowIndex in indexes:
            colors.append(self.tableWidget.item(rowIndex.row(), 0).background().color().name())

        # Get numbers and runs here.
        elements = []
        for wid in selected:
            elements.append(wid.text())

        runs = [int(i) for i in elements[1::10]]
        dates = [int(i) for i in elements[0::10]]
        nums = [float(i.split()[0]) for i in elements[8::10]]
        tempx = []
        tempz = []
        for run in runs:
            tempx.append(self.get_temp(dates[0], run)[0])
            tempz.append(self.get_temp(dates[0], run)[1])

        nums, tempx, tempz, colors = zip(*sorted(zip(nums, tempx, tempz, colors)))
        nums = np.log(nums)
        tempx = np.log(tempx)
        tempz = np.log(tempz)

        plt.plot(tempx, nums, c='r', ls='--', alpha=0.3, label="Temperature-x")
        plt.scatter(tempx, nums, c=colors)
        plt.plot(tempz, nums, c='b', ls=':', alpha=0.3, label="Temperature-z")
        plt.scatter(tempz, nums, c=colors)
        plt.title("Log-log Plot.")
        plt.xlabel('Log(T)')
        plt.ylabel('Log(N)')
        plt.legend()
        plt.grid()
        plt.show()

    def OAHloglog(self, event):
        dialog = InputDialog()
        if dialog.exec():
            dz_focus, fit_mode_string, quad_string, bin_val, time_ms, ignore, areaX, areaY = dialog.getInputs()
            self.setVariables()
            try:
                self.date = self.tableWidget.selectedItems()[0].text()
                self.run = self.tableWidget.selectedItems()[1].text()
            except Exception as e:
                print(e, " - Trying to plot, but no data selected.")
                return
            self.pbar.show()

            cut_coords = self.paramsFromImg(setParams=False)
            print([0, cut_coords[1], 1., [cut_coords[1][0]-cut_coords[0][0], cut_coords[1][1]-cut_coords[0][1]], 1., cut_coords[2]])

            coolingOAH_main([int(self.date), int(self.run), float(dz_focus), str(fit_mode_string), str(quad_string), 1 , [int(bin_val), int(bin_val)],
                             cut_coords[0],
                             # [60, -10, 1100, 1400],
                             [0, cut_coords[1], 1., [cut_coords[1][0]-cut_coords[0][0], cut_coords[1][1]-cut_coords[0][1]], 1., cut_coords[2]],
                             int(time_ms)],
                            self.pbar, ignore)
            #coolingOAH_main([date, shot, focus, fit_mode, quad_string, NUM , [xbin, ybin], [xmin, xmax, ymin, ymax],
            #                [offset, [centerx, centery], gauss_amp, [gausswx, gausswy], tf_amp, [tfwx, tfwy] time_ms], self.pbar, ignore)

            self.pbar.hide()

    def timeCrystalAnalysis(self, event):
        try:
            date = self.tableWidget.selectedItems()[0].text()
            run = self.tableWidget.selectedItems()[1].text()
            os.system('python3 pckg/fit/time_crystal_analysis.py {:} {:}'.format(date, run))

        except Exception as e:
            print(e, " - Trying to plot, but no data selected.")
            return

    def cell_double_clicked(self, row, column):
        # Check if the double-clicked cell is in the specific column, say column 1
        if column == 1:
            print(row, column)
            # Set the background color of the cell
            self.tableWidget.item(row, column).setBackground(QColor(255, 255, 0))

    def bbot(self, state):
        if state:
            if not os.path.exists("/home/bec_lab/python/BECViewer/resources/.run_bbot"):
                f = open("/home/bec_lab/python/BECViewer/resources/.run_bbot", 'x')
                print("Starting the BecBot")

            # Make the .run_bbot file in the resources folder.
            self.label_loading.show()
            self.label_loading.setMovie(self.movie)
            self.movie.start()
            command = f"python3 -u /home/bec_lab/python/becbot.py"
            os.system("gnome-terminal -e 'bash -c \"" + command + ";bash\"'")
        else:
            if os.path.exists("/home/bec_lab/python/BECViewer/resources/.run_bbot"):
                # delete the .run_bbot file
                print("Exiting the BBOT")
                os.remove("/home/bec_lab/python/BECViewer/resources/.run_bbot")
                self.movie.stop()
                self.label_loading.hide()
            else:
                self.movie.stop()
                self.label_loading.hide()

    # --------------------------------------------
    # ----------- Plotting Functions -------------
    # --------------------------------------------

    def setCmap(self):
        # Generate a combox with a list of cmaps that can be used, and set it to default value of 'bwr'
        self.themes = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
                       'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys',
                       'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
                       'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                       'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
                       'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1',
                       'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
                       'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
                       'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
                       'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper',
                       'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
                       'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
                       'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                       'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv',
                       'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral',
                       'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism',
                       'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                       'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
                       'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
                       'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
        self.comboBox_themes.addItems(self.themes)
        index = self.comboBox_themes.findText('afmhot_r', QtCore.Qt.MatchFixedString)
        self.comboBox_themes.setCurrentIndex(index)  # Set initial to 'bwr'
        self.lineEdit_vmax.setText("1.3")
        self.lineEdit_vmin.setText("0")

    def setCmapOAH(self):
        index = self.comboBox_themes.findText('afmhot_r', QtCore.Qt.MatchFixedString)
        self.comboBox_themes.setCurrentIndex(index)  # Set initial to 'bwr'
        self.lineEdit_vmax.setText("2")
        self.lineEdit_vmin.setText("-2")

    def clearPlot(self):
        # Clear the output of the plot
        try:
            self.verticalLayout_18.removeWidget(self.toolbar)
            self.verticalLayout_18.removeWidget(self.canvas)
            sip.delete(self.toolbar)
            sip.delete(self.canvas)
            self.toolbar = None
            self.canvas = None
        except Exception as e:
            print(e)
            pass

    def updatePlot(self, switch=True):
        # Some supplementary functions for plotting and interacting with the plot.
        def updatePlot_linecuts(axes, indx_h, indx_v, alpha):
            """ A function to call on mouseclick, to update the linecuts """
            while len(axes.lines) > 1:
                axes.lines[-1].remove()
                axes.lines[-1].remove()
            drawLineCuts(axes, indx_v, indx_h, alpha)

        def drawLineCuts(axes, indx_v, indx_h, alpha):
            """ A function that draws the linecuts upon selection. """
            self.canvas.ax2.clear()
            self.canvas.ax3.clear()
            axes.axvline(indx_h, alpha=alpha)
            axes.axhline(indx_v, alpha=alpha)
            y_data = np.arange(pic.shape[0])
            x_data = np.arange(pic.shape[1])
            linecut_h = self.picLineCut(pic, indx_v, 'horizontal')
            linecut_v = self.picLineCut(pic, indx_h, 'vertical')
            self.canvas.ax2.scatter(x_data, linecut_h, c='k', marker=".")
            self.canvas.ax3.scatter(linecut_v, y_data, c='k', marker=".")
            self.canvas.ax3.set_xlim(linecut_v.max(), linecut_v.min())
            if self.analysisMode not in ["Viewer", "Raw Images", "FFT Plane"]:
                linecut_h = self.picLineCut(fitresults, indx_v, 'horizontal')
                linecut_v = self.picLineCut(fitresults, indx_h, 'vertical')
                self.canvas.ax2.plot(x_data, linecut_h, c='r', alpha=0.7, label="Combined")
                self.canvas.ax3.plot(linecut_v, y_data, c='r', alpha=0.7, label="Combined")
                if self.analysisMode == "Bi-modal Fit":
                    linecut_h = self.picLineCut(fitresultgauss, indx_v, 'horizontal')
                    linecut_v = self.picLineCut(fitresultgauss, indx_h, 'vertical')
                    linecut_h2 = self.picLineCut(fitresulttf, indx_v, 'horizontal')
                    linecut_v2 = self.picLineCut(fitresulttf, indx_h, 'vertical')
                    self.canvas.ax2.plot(x_data, linecut_h, c='C4', alpha=0.6, ls="--", label="Gaussian")
                    self.canvas.ax3.plot(linecut_v, y_data, c='C4', alpha=0.6, ls="--", label="Gaussian")
                    self.canvas.ax2.plot(x_data, linecut_h2, c='C2', alpha=0.6, ls="--", label="Thomas-Fermi")
                    self.canvas.ax3.plot(linecut_v2, y_data, c='C2', alpha=0.6, ls="--", label="Thomas-Fermi")
                    self.canvas.ax2.legend()
            self.canvas.ax4.set_visible(False)

        # Load the variables, the selected date and run sequence
        self.setVariables()
        try:
            self.date = self.tableWidget.selectedItems()[0].text()
            self.run = self.tableWidget.selectedItems()[1].text()
        except Exception as e:
            print(e, " - Trying to plot, but no data selected.")
            return

        # If selected successfully, automatically go to the analysis tab, and modify the label to current
        # run and date indices, set the next/prev buttons, display default Temp/Chem/num text, and the comment.
        # self.label_info.setText(" RUN ID : {:} / {:}".format(self.date, self.run))
        if self.info_bttn.isChecked():
            self.displayInfo(state=QtCore.Qt.Checked)
        else:
            self.label_info.setText(" RUN ID : {:} / {:}".format(self.date, self.run))

        self.prevImage(onlyButtons=True)
        self.nextImage(onlyButtons=True)
        # self.prevImage(onlyButtons=True)
        self.label_info.setFont(QtGui.QFont('monospace', 10))
        #if switch:
        self.tabWidget.setCurrentIndex(1)
        self.label_chem.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-weight:600;\">Chemical "
                                           "Potential:</span></p></body></html>"))
        self.label_pnumber.setText(_translate("MainWindow",
                                              "<html><head/><body><p><span style=\" font-weight:600;\">Particle "
                                              "Number:</span></p></body></html>"))
        self.label_temp.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-weight:600;\">Temperature: "
                                           "</span></p></body></html>"))

        self.textEdit_comment.setText(self.get_comment(self.date, self.run))

        # Delete the navigation bar and the plot if drawn, and then draw and append them to the layout widget.
        self.clearPlot()
        self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout_18.addWidget(self.canvas)
        self.verticalLayout_18.addWidget(self.toolbar)
        self.canvas.ax_in1.set_visible(False)
        self.canvas.ax_in2.set_visible(False)
        self.canvas.ax_in3.set_visible(False)

        # Select plot variables
        self.cmap = str(self.comboBox_themes.currentText())
        self.vmax = float(self.lineEdit_vmax.text())
        self.vmin = float(self.lineEdit_vmin.text())
        self.xmin = int(self.lineEdit_cut_xi.text())
        self.xmax = int(self.lineEdit_cut_xf.text())
        self.zmin = int(self.lineEdit_cut_zi.text())
        self.zmax = int(self.lineEdit_cut_zf.text())
        self.canvas.ax.yaxis.tick_right()
        self.canvas.ax3.yaxis.tick_right()
        plt.setp(self.canvas.ax.get_yticklabels(), visible=False)
        plt.setp(self.canvas.ax.get_xticklabels(), visible=False)
        alpha = 0.01
        if self.checkBox_aspect.isChecked():
            aspect = 1
        else:
            aspect = "auto"

        # Choose the analysis mode
        self.analysisMode = self.comboBox_mode.currentText()

        if self.analysisMode == "Viewer" or "Off-Axis Holography":
            out1 = self.get_OAH_analysis_foci(self.date, self.run)
            # Set the labels for OAH
            self.label_temp.setText(_translate("MainWindow", f"{out1}"))
            self.label_chem.setText(_translate("MainWindow", ""))
            self.label_pnumber.setText(_translate("MainWindow", ""))

        # Binning
        if self.analysisMode != "Viewer" and self.analysisMode != "Off-Axis Holography":
            # For all fitting stuff, we bin using the comb
            self.radioButton_nobin.setChecked(True)
            self.xbin = int(self.spinBox_bin_comb.value())
            self.zbin = int(self.spinBox_bin_comb.value())
        # Else, dependent on the selection. By default, start with no bin.
        elif self.rbStatus == 0:
            self.xbin = 1
            self.zbin = 1
        elif self.rbStatus == 1:
            self.xbin = int(self.spinBox_bin_comb.value())
            self.zbin = int(self.spinBox_bin_comb.value())
        elif self.rbStatus == 2:
            self.xbin = int(self.spinBox_bin_x.value())
            self.zbin = int(self.spinBox_bin_z.value())

        # Generate a pic file, cut it, bin it, and get the peaks positions form the pic array
        if self.checkBox_OAHFit.isChecked():
            path = '/storage/data/' + str(self.date) + '/'
            image = str(self.run).zfill(4) + '/'
            full_path = path + image
            dz_focus = self.spinBoxFocus.value()
            # Get the quad
            if self.radioButton_oah1.isChecked():
                self.quad = "quad1"
            elif self.radioButton_oah2.isChecked():
                self.quad = "quad2"
            else:
                print("Error - no quad selected.")
                return
            if os.path.exists(full_path + "pics_foc_ss_{:}_{:}.fits".format(self.quad, dz_focus)):
                print("FITTING OAH")
                pic = pyfits.open(full_path + "pics_foc_ss_{:}_{:}.fits".format(self.quad, dz_focus))[0].data.astype(float)[:][0]
                mask = pic == 0
                pic = np.ma.array(pic, mask=mask)
                pic= - pic + 0.5
            elif os.path.exists(full_path + "pics_foc_{:}_{:}.fits".format(self.quad, dz_focus)):
                print("FITTING OAH")
                pic = pyfits.open(full_path + "pics_foc_{:}_{:}.fits".format(self.quad, dz_focus))[0].data.astype(float)[:][0]
                mask = pic == 0
                pic = np.ma.array(pic, mask=mask)
                pic = - pic + 0.5
        else:
            pic = self.generateImage(self.date, self.run)
            print(type(pic))
        pic = pic[4:-4, 4:-4]
        # picnobin = pic.copy()
        pic = self.cutImage(pic, self.xmin, self.xmax, self.zmin, self.zmax)
        pic_nobin = pic.copy()
        pic = self.binImage(pic, self.xbin, self.zbin)
        indx = np.where(pic == pic.min())

        ################################# ANALYSIS ######################################
        if self.analysisMode != "Viewer" and self.analysisMode != "Raw Images" and self.analysisMode != "FFT Plane" :
            self.radioButton_nobin.setChecked(True)
            self.xbin = int(self.spinBox_bin_comb.value())
            self.zbin = int(self.spinBox_bin_comb.value())

            if self.analysisMode == "Gaussian Fit":
                # Gaussian
                fitresults, fitguess, fitresultgauss, fitresulttf = self.fitFunctions(pic, mode="gauss")
                self.label_mode.setText(_translate("Main Window:", "<b>Analysed using Gaussian Model.</b> <br/> <br/>"))
            elif self.analysisMode == "Thomas-Fermi Fit":
                # Thomas-Fermi
                fitresults, fitguess, fitresultgauss, fitresulttf = self.fitFunctions(pic, mode="tf")
                self.label_mode.setText(
                    _translate("Main Window:", "<b>Analysed using Thomas-Fermin Model.</b> <br/> <br/>"))
            elif self.analysisMode == "Bi-modal Fit":
                # Bimodal
                fitresults, fitguess, fitresultgauss, fitresulttf = self.fitFunctions(pic, mode="bimodal")
                self.label_mode.setText(_translate("Main Window:", "<b>Analysed using Bimodal Model.</b> <br/> <br/>"))

            elif self.analysisMode == "Off-Axis Holography":
                # OAH
                #self.setCmapOAH()
                self.label_mode.setText(_translate("Main Window:", "<b>Analysed with OAH.</b> <br/> <br/>"))

                dz_focus = self.spinBoxFocus.value()
                self.cmap = str(self.comboBox_themes.currentText())
                self.vmax = float(self.lineEdit_vmax.text())
                self.vmin = float(self.lineEdit_vmin.text())
                # Get the quad
                if self.radioButton_oah1.isChecked():
                    self.quad = "quad1"
                elif self.radioButton_oah2.isChecked():
                    self.quad = "quad2"
                else:
                    print("Error - no quad selected.")
                    return
                
                if self.radioButton_amp.isChecked():
                    self.OAH_out = "amp"
                elif self.radioButton_ang.isChecked():
                    self.OAH_out = "ang"
                else:
                    print("Error - no output mode.")
                    return
                
                # Pre-process images
                if self.checkBox_singleNum.isChecked():
                    self.num = int(self.spinBox_num.value())
                    
                else:
                    self.num = "single"

                if self.checkBox_multi_img.isChecked():
                    self.OAH_output = preprocessHI_refocus_multi(self.date, self.run, dz_focus, self.progressBar, self.quad, self.OAH_out)
                else:
                    self.OAH_output = preprocessHI_refocus(self.date, self.run, dz_focus, self.progressBar, self.num, self.quad, self.OAH_out)

                output = self.OAH_output[0]
                output = self.cutImage(output, self.xmin, self.xmax, self.zmin, self.zmax)
                im = self.canvas.ax.imshow(output, cmap=self.cmap,
                                           vmin=self.vmin, vmax=self.vmax,
                                           interpolation='none', origin="lower", aspect=aspect)
                plt.colorbar(im, ax=self.canvas.ax3, orientation="vertical")
                self.canvas.ax2.set_visible(False)
                self.canvas.ax3.set_visible(False)
                self.canvas.ax4.set_visible(False)
                self.canvas.ax_in1.remove()
                self.canvas.ax_in2.remove()
                self.canvas.ax_in3.remove()
                self.num_of_shots = self.get_numofshots(self.date, self.run)
                if self.path == "/storage/data/":
                    self.canvas.print_figure(
                        "/storage/data/{:}/{:}/{:}_{:}.png".format(self.date, self.run, "HI", self.run))
                return

            # Here need to transform to nonbined sizes.
            levels = np.linspace(self.vmin, self.vmax, 3)
            self.canvas.ax.contour(fitresults, levels, cmap='gray', vmin=self.vmin, vmax=self.vmax, origin="lower")
            indx = np.where(fitresults == fitresults.min())
        #################################################################################

        indx_ht = indx[1][0]
        indx_vt = indx[0][0]

        # For previewing the 3-fit-pannel
        if self.checkBox_fitresults.isChecked() and self.analysisMode not in ["Viewer", "Off-Axis Holography", "Raw Images", "FFT Plane"]:
            self.canvas.ax_in1.set_visible(True)
            self.canvas.ax_in2.set_visible(True)
            self.canvas.ax_in3.set_visible(True)
            self.canvas.ax.set_visible(False)
            im = self.canvas.ax_in1.imshow(pic, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                           origin="lower", aspect=aspect)
            self.canvas.ax_in2.imshow(fitresults, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                      origin="lower", aspect=aspect)
            self.canvas.ax_in3.imshow(fitguess, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                      origin="lower", aspect=aspect)
            plt.colorbar(im, ax=self.canvas.ax4, orientation="horizontal")
            cursor = mplcursors.cursor(im)
            if self.checkBox_lcut.isChecked():
                alpha = 0.4
            drawLineCuts(self.canvas.ax_in1, indx_vt, indx_ht, alpha)
            cursor.connect("add", lambda sel: updatePlot_linecuts(self.canvas.ax_in1, sel.target.index[1],
                                                                  sel.target.index[0]))

        # For previewing the 3-fit-pannel
        elif self.analysisMode == "Raw Images":
            atoms, flat, dark, pic = self.generateImage(self.date, self.run, raw=True)
            atoms = atoms[4:-4, 4:-4]
            flat = flat[4:-4, 4:-4]            
            dark = dark[4:-4, 4:-4]

            # picnobin = pic.copy()
            atoms = self.cutImage(atoms, self.xmin, self.xmax, self.zmin, self.zmax)
            flat = self.cutImage(flat, self.xmin, self.xmax, self.zmin, self.zmax)
            dark = self.cutImage(dark, self.xmin, self.xmax, self.zmin, self.zmax)
            
            atoms = self.binImage(atoms, self.xbin, self.zbin)
            flat = self.binImage(flat, self.xbin, self.zbin)
            dark = self.binImage(dark, self.xbin, self.zbin)
            indx = np.where(pic == pic.min())
            self.canvas.ax_in1.set_visible(True)
            self.canvas.ax_in2.set_visible(True)
            self.canvas.ax_in3.set_visible(True)
            self.canvas.ax.set_visible(False)
            im = self.canvas.ax_in1.imshow(atoms, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                           origin="lower", aspect=aspect)
            self.canvas.ax_in2.imshow(flat, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                      origin="lower", aspect=aspect)
            self.canvas.ax_in3.imshow(dark, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                      origin="lower", aspect=aspect)
            plt.colorbar(im, ax=self.canvas.ax4, orientation="horizontal")
            cursor = mplcursors.cursor(im)
            if self.checkBox_lcut.isChecked():
                alpha = 0.4
            drawLineCuts(self.canvas.ax_in1, indx_vt, indx_ht, alpha)
            cursor.connect("add", lambda sel: updatePlot_linecuts(self.canvas.ax_in1, sel.target.index[1], sel.target.index[0], alpha))

        # Or a single image
        else:
            if self.analysisMode == "FFT Plane":
                atoms, flat, dark, pic = self.generateImage(self.date, self.run, raw=True)
                self.canvas.ax_in1.remove()
                self.canvas.ax_in2.remove()
                self.canvas.ax_in3.remove()
                im = self.canvas.ax.imshow(np.log(abs(np.fft.fftshift(np.fft.fft2(atoms)))), cmap=plt.get_cmap('gist_heat'))
                plt.colorbar(im, ax=self.canvas.ax4, orientation="horizontal")
                cursor = mplcursors.cursor(im)
                if self.checkBox_lcut.isChecked():
                    alpha = 0.4
                drawLineCuts(self.canvas.ax, indx_vt, indx_ht, alpha)
                cursor.connect("add",
                            lambda sel: updatePlot_linecuts(self.canvas.ax, sel.target.index[1], sel.target.index[0],
                                                            alpha))
            else:
                self.canvas.ax_in1.remove()
                self.canvas.ax_in2.remove()
                self.canvas.ax_in3.remove()
                im = self.canvas.ax.imshow(pic, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                        origin="lower", aspect=aspect)
                plt.colorbar(im, ax=self.canvas.ax4, orientation="horizontal")
                cursor = mplcursors.cursor(im)
                if self.checkBox_lcut.isChecked():
                    alpha = 0.4
                drawLineCuts(self.canvas.ax, indx_vt, indx_ht, alpha)
                cursor.connect("add",
                            lambda sel: updatePlot_linecuts(self.canvas.ax, sel.target.index[1], sel.target.index[0],
                                                            alpha))

        # Save Figure - to note under "previous analysis" and to keep a .png file.
        if self.analysisMode == "Gaussian Fit":
            params = self.get_fit_parameter(self.date, self.run, "gauss")
            self.canvas.ax.set_title("$N_{count} = $" + "{:.2e}".format(params[16]) +
                                     "   $T_{x} = $" + "{:.2e}".format(params[20]) + " K \n" +
                                     "$N_{therm} = $" + "{:.2e}".format(params[17]) +
                                     "   $T_{z} = $" + "{:.2e}".format(params[21]) + " K \n"
                                     )
            if self.path == "/storage/data/":
                self.canvas.print_figure(
                    "/storage/data/{:}/{:}/{:}_{:}.png".format(self.date, self.run, "fit_g", self.run))
        elif self.analysisMode == "Thomas-Fermi Fit":
            params = self.get_fit_parameter(self.date, self.run, "tf")
            self.canvas.ax.set_title("$N_{count} = $" + "{:.2e}".format(params[16]) +
                                     "   $T_{x} = $" + "{:.2e}".format(params[20]) + " K \n" +
                                     "$N_{cond} = $" + "{:.2e}".format(params[18]) +
                                     "   $T_{z} = $" + "{:.2e}".format(params[21]) + " K \n"
                                     )
            if self.path == "/storage/data/":
                self.canvas.print_figure(
                    "/storage/data/{:}/{:}/{:}_{:}.png".format(self.date, self.run, "fit_tf", self.run))
        elif self.analysisMode == "Bi-modal Fit":
            params = self.get_fit_parameter(self.date, self.run, "bimodal")
            self.canvas.ax.set_title("$N_{therm} = $" + "{:.2e}".format(params[23]) +
                                     "   $T_{x} = $" + "{:.2e}".format(params[26]) + " K \n" +
                                     "$N_{cond} = $" + "{:.2e}".format(params[24]) +
                                     "   $T_{z} = $" + "{:.2e}".format(params[27]) + " K \n" +
                                     "Condensate Fraction = {:.2f}%".format(
                                         params[24] / (params[23] + params[24]) * 100)
                                     )
            if self.path == "/storage/data/":
                self.canvas.print_figure(
                    "/storage/data/{:}/{:}/{:}_{:}.png".format(self.date, self.run, "fit_bi", self.run))
        elif self.analysisMode == "Viewer":
            try:
                params = self.get_fit_parameter(self.date, self.run, "gauss")
                self.canvas.ax.set_title("$N_{therm} = $" + "{:} M".format(round(params[17]/1e6)), fontsize=20, weight='bold', pad=18, color='gray')
            except:
                pass
            self.num_of_shots = self.get_numofshots(self.date, self.run)
            if self.path == "/storage/data/":
                self.canvas.print_figure(
                    "/storage/data/{:}/{:}/{:}_{:}.png".format(self.date, self.run, "viewer", self.run))

        self.canvas.draw_idle()

        if not self.checkBox_autoFit.isChecked() and not self.analysisMode == "Raw Images":
            self.comboBox_mode.setCurrentText("Viewer")
            self.checkBox_fitresults.setChecked(False)

    def slider(self, value):
        # Some supplementary functions for plotting and interacting with the plot.
        def updatePlot_linecuts(axes, indx_h, indx_v, alpha):
            """ A function to call on mouseclick, to update the linecuts """
            while len(axes.lines) > 1:
                axes.lines[-1].remove()
                axes.lines[-1].remove()
            drawLineCuts(axes, indx_v, indx_h, alpha)

        def drawLineCuts(axes, indx_v, indx_h, alpha):
            """ A function that draws the linecuts upon selection. """
            self.canvas.ax2.clear()
            self.canvas.ax3.clear()
            axes.axvline(indx_h, alpha=alpha)
            axes.axhline(indx_v, alpha=alpha)
            y_data = np.arange(pic.shape[0])
            x_data = np.arange(pic.shape[1])
            linecut_h = self.picLineCut(pic, indx_v, 'horizontal')
            linecut_v = self.picLineCut(pic, indx_h, 'vertical')
            self.canvas.ax2.scatter(x_data, linecut_h, c='k', marker=".")
            self.canvas.ax3.scatter(linecut_v, y_data, c='k', marker=".")
            self.canvas.ax3.set_xlim(linecut_v.max(), linecut_v.min())
            # if self.analysisMode != "Viewer":
            #     linecut_h = self.picLineCut(fitresults, indx_v, 'horizontal')
            #     linecut_v = self.picLineCut(fitresults, indx_h, 'vertical')
            #     self.canvas.ax2.plot(x_data, linecut_h, c='r', alpha=0.7, label="Combined")
            #     self.canvas.ax3.plot(linecut_v, y_data, c='r', alpha=0.7, label="Combined")
            #     if self.analysisMode == "Bi-modal Fit":
            #         linecut_h = self.picLineCut(fitresultgauss, indx_v, 'horizontal')
            #         linecut_v = self.picLineCut(fitresultgauss, indx_h, 'vertical')
            #         linecut_h2 = self.picLineCut(fitresulttf, indx_v, 'horizontal')
            #         linecut_v2 = self.picLineCut(fitresulttf, indx_h, 'vertical')
            #         self.canvas.ax2.plot(x_data, linecut_h, c='C4', alpha=0.6, ls="--", label="Gaussian")
            #         self.canvas.ax3.plot(linecut_v, y_data, c='C4', alpha=0.6, ls="--", label="Gaussian")
            #         self.canvas.ax2.plot(x_data, linecut_h2, c='C2', alpha=0.6, ls="--", label="Thomas-Fermi")
            #         self.canvas.ax3.plot(linecut_v2, y_data, c='C2', alpha=0.6, ls="--", label="Thomas-Fermi")
            #         self.canvas.ax2.legend()
            self.canvas.ax4.set_visible(False)

        if self.analysisMode == "Off-Axis Holography":
            self.horizontalSlider.setMaximum(self.num_of_shots-1)
            if self.num_of_shots != 0:
                # OAH
                self.clearPlot()
                self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
                self.toolbar = NavigationToolbar(self.canvas, self)
                self.verticalLayout_18.addWidget(self.canvas)
                self.verticalLayout_18.addWidget(self.toolbar)
                self.canvas.ax_in1.set_visible(False)
                self.canvas.ax_in2.set_visible(False)
                self.canvas.ax_in3.set_visible(False)
                # Select plot variables
                self.cmap = str(self.comboBox_themes.currentText())
                self.vmax = float(self.lineEdit_vmax.text())
                self.vmin = float(self.lineEdit_vmin.text())
                self.xmin = int(self.lineEdit_cut_xi.text())
                self.xmax = int(self.lineEdit_cut_xf.text())
                self.zmin = int(self.lineEdit_cut_zi.text())
                self.zmax = int(self.lineEdit_cut_zf.text())
                self.canvas.ax.yaxis.tick_right()
                self.canvas.ax3.yaxis.tick_right()
                plt.setp(self.canvas.ax.get_yticklabels(), visible=False)
                plt.setp(self.canvas.ax.get_xticklabels(), visible=False)
                alpha = 0.2
                if self.checkBox_aspect.isChecked():
                    aspect = 1
                else:
                    aspect = "auto"

                # Pre-processing images and go through the array.
                output = self.OAH_output[value]
                output = self.cutImage(output, self.xmin, self.xmax, self.zmin, self.zmax)
                im = self.canvas.ax.imshow(output, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                           origin="lower", aspect=aspect)
                plt.colorbar(im, ax=self.canvas.ax3, orientation="vertical")
                # self.canvas.ax2.set_visible(False)
                # self.canvas.ax3.set_visible(False)
                # self.canvas.ax4.set_visible(False)
                self.label_info.setText(" RUN ID : {:} / {:} -- {:} / {:}".format(self.date, self.run, value, self.num_of_shots))
                self.label_info.setFont(QtGui.QFont('monospace', 10))

            indx = np.where(output == output.min())
            indx_ht = indx[1][0]
            indx_vt = indx[0][0]
            cursor = mplcursors.cursor(im)
            alpha = 0.2
            if self.checkBox_lcut.isChecked():
                alpha = 0.4
            # drawLineCuts(self.canvas.ax, indx_vt, indx_ht, alpha)
            cursor.connect("add",
                           lambda sel: updatePlot_linecuts(self.canvas.ax, sel.target.index[1], sel.target.index[0],
                                                           alpha))

        elif self.analysisMode == "Viewer":
            self.horizontalSlider.setMaximum(self.num_of_shots)
            if self.num_of_shots != 0:
                self.clearPlot()
                self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
                self.toolbar = NavigationToolbar(self.canvas, self)
                self.verticalLayout_18.addWidget(self.canvas)
                self.verticalLayout_18.addWidget(self.toolbar)
                self.canvas.ax_in1.set_visible(False)
                self.canvas.ax_in2.set_visible(False)
                self.canvas.ax_in3.set_visible(False)
                if self.checkBox_OAHFit.isChecked():
                    path = '/storage/data/' + str(self.date) + '/'
                    image = str(self.run).zfill(4) + '/'
                    full_path = path + image
                    dz_focus = self.spinBoxFocus.value()
                    if os.path.exists(full_path + "pics_foc_{:}.fits".format(dz_focus)):
                        print("FITTING OAH")
                        pic = pyfits.open(full_path + "pics_foc_{:}.fits".format(dz_focus))[0].data.astype(float)[:][value]
                        mask = pic == 0
                        pic = np.ma.array(pic, mask=mask)
                        pic= - pic  + 0.5

                # OAH
                else:
                    pic = self.generateImage(self.date, self.run, num=value)
                # Select plot variables
                self.cmap = str(self.comboBox_themes.currentText())
                self.vmax = float(self.lineEdit_vmax.text())
                self.vmin = float(self.lineEdit_vmin.text())
                self.xmin = int(self.lineEdit_cut_xi.text())
                self.xmax = int(self.lineEdit_cut_xf.text())
                self.zmin = int(self.lineEdit_cut_zi.text())
                self.zmax = int(self.lineEdit_cut_zf.text())
                self.canvas.ax.yaxis.tick_right()
                self.canvas.ax3.yaxis.tick_right()
                plt.setp(self.canvas.ax.get_yticklabels(), visible=False)
                plt.setp(self.canvas.ax.get_xticklabels(), visible=False)
                alpha = 0.2
                if self.checkBox_aspect.isChecked():
                    aspect = 1
                else:
                    aspect = "auto"
                im = self.canvas.ax.imshow(pic, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, interpolation='none',
                                           origin="lower", aspect=aspect)
                plt.colorbar(im, ax=self.canvas.ax3, orientation="vertical")
                # self.canvas.ax2.set_visible(False)
                # self.canvas.ax3.set_visible(False)
                # self.canvas.ax4.set_visible(False)
                self.label_info.setText(" RUN ID : {:} / {:} -- {:} / {:}".format(self.date, self.run, value, self.num_of_shots))
                self.label_info.setFont(QtGui.QFont('monospace', 10))

            indx = np.where(pic == pic.min())
            indx_ht = indx[1][0]
            indx_vt = indx[0][0]
            cursor = mplcursors.cursor(im)
            alpha = 0.2
            if self.checkBox_lcut.isChecked():
                alpha = 0.4
            drawLineCuts(self.canvas.ax, indx_vt, indx_ht, alpha)
            cursor.connect("add",
                           lambda sel: updatePlot_linecuts(self.canvas.ax, sel.target.index[1], sel.target.index[0],
                                                           alpha))

    def prevImage(self, onlyButtons=False):
        curRow = self.tableWidget.currentRow()
        nextRow = curRow + 1
        self.tableWidget.selectRow(nextRow)

        string_text = str(int(self.tableWidget.selectedItems()[1].text()) - 1).zfill(4)
        self.pushButton_prev.setText("<  " + string_text)
        self.pushButton_next.setText(str(int(self.tableWidget.selectedItems()[1].text()) + 1).zfill(4) + "  >")
        self.pushButton_next.setFont(QFont('Umpush', 11))
        if not onlyButtons:
            self.updatePlot()
            self.displayInfo()

    def nextImage(self, onlyButtons=False):
        curRow = self.tableWidget.currentRow()
        nextRow = curRow - 1
        self.tableWidget.selectRow(nextRow)
        self.pushButton_prev.setText("<  " + str(int(self.tableWidget.selectedItems()[1].text()) - 1).zfill(4))
        self.pushButton_next.setText(str(int(self.tableWidget.selectedItems()[1].text()) + 1).zfill(4) + "  >")
        self.pushButton_next.setFont(QFont('Umpush', 11))
        if not onlyButtons:
            self.updatePlot()
            self.displayInfo()

    def viewer(self):
        self.comboBox_mode.setCurrentText("Viewer")

    def gauss(self):
        self.comboBox_mode.setCurrentText("Gaussian Fit")

    def generateImage(self, date, run, num=0, raw=False):
        """ Generate a pic array for the image. """
        input_folder = '/storage/data/' + date + '/' + run + "/"
        # input_folder = '/storage/data/20201112/0074/'
        if self.checkBox_multi_img.isChecked():
            atoms1 = pyfits.open(input_folder + '0.fits')[0].data.astype(float)[:]  # .mean(axis=0)
            atoms2 = pyfits.open(input_folder + '1.fits')[0].data.astype(float)[:]  # .mean(axis=0)
            atom = np.concatenate([atoms1, atoms2])[num]
            flat = pyfits.open(input_folder + '2.fits')[0].data.astype(float)[0]  # .mean(axis=0)
            dark = pyfits.open(input_folder + '3.fits')[0].data.astype(float).mean(axis=0)
        else:
            atom = pyfits.open(input_folder + '0.fits')[0].data.astype(float)[num]  # .mean(axis=0)
            flat = pyfits.open(input_folder + '1.fits')[0].data.astype(float)[0]  # .mean(axis=0)
            dark = pyfits.open(input_folder + '2.fits')[0].data.astype(float).mean(axis=0)

        # --------------------------------------- CREATE THE PICTURE ARRAY ------------------------------------------
        atom_corr = atom - dark
        flat_corr = flat - dark 
        atom_corr[atom_corr < 0.0001] = 0.0001
        flat_corr[flat_corr < 0.0001] = 0.0001
        pic = atom_corr / flat_corr
        # pic = atom / flat
        pic[pic < 0.0001] = 0.0001
        mask = flat - dark < 25.
        pic = np.ma.array(pic, mask=mask)
        if raw: 
            return atom, flat, dark, pic
        return pic

    def binImage(self, pic, xbin, zbin):
        """ A function to bin the pic file based on the bin parameters. """
        # If pic file not a multiple of bin, cut from the edge so it is.
        if pic.shape[0] % xbin != 0:
            pic = pic[:-(pic.shape[0] % xbin), :]
        if pic.shape[1] % zbin != 0:
            pic = pic[:, :-(pic.shape[1] % zbin)]
        pic = pic.reshape(pic.shape[0] // xbin, xbin, pic.shape[1] // zbin, zbin).mean(axis=3).mean(axis=1)
        return pic

    def cutImage(self, pic, xmin, xmax, ymin, ymax):
        """ A funciton to quickly cut the pic file. """
        pic = pic[xmin:, ymin:]
        if xmax != 0:
            pic = pic[:xmax, :]
        if ymax != 0:
            pic = pic[:, :ymax]
        return pic

    def picLineCut(self, pic, indx, mode):
        if mode == 'vertical':
            pic_linecut = pic[:, indx]
        elif mode == 'horizontal':
            pic_linecut = pic[indx, :]
        else:
            print("Please select a valid mode: [vertical, horizontal].")
        return pic_linecut

    def radioButtonBin(self):
        """ Problematic because only on click it records what the value of the spinbox is. """
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == "No Binning":
                self.rbStatus = 0
            elif radioBtn.text() == "Bin-combined":
                self.rbStatus = 1
            elif radioBtn.text() == "Bin-x :":
                self.rbStatus = 2

    def initialParams(self):
        """
        A function to set the values of the initial parameters, when the plot is first plotted and generated:
            - bin: set to bin comb and select the comb radio button
            - cut: set to no cut
        """
        self.radioButton_nobin.setChecked(True)
        self.lineEdit_cut_xi.setText("0")
        self.lineEdit_cut_xf.setText("0")
        self.lineEdit_cut_zi.setText("0")
        self.lineEdit_cut_zf.setText("0")
        self.rbStatus = 0

    def OAH_mode_display(self, text):
        """ Display the counter when the multimage check button is checked. """
        if text == "Off-Axis Holography":
            self.checkBox_multi_img.show()
            self.spinBox_num.show()
            self.checkBox_singleNum.show()
            self.checkBox_OAHFit.show()
            self.radioButton_oah1.show()
            self.radioButton_oah2.show()
            self.radioButton_ang.show()
            self.radioButton_amp.show()
        else:
            self.checkBox_multi_img.hide()
            self.spinBox_num.hide()
            self.checkBox_singleNum.hide()
            self.checkBox_OAHFit.hide()
            self.radioButton_oah1.hide()
            self.radioButton_oah2.hide()
            self.radioButton_ang.hide()
            self.radioButton_amp.hide()


    def updateMaxVal(self):
        max_num = pyfits.open(f"/storage/data/{self.date}/{str(self.run).zfill(4)}/0.fits")[0].data.astype(float)[:].shape[0]
        self.spinBox_num.setMaximum(max_num-1)
        self.spinBox_num.setSuffix(f" / {max_num - 1}")


    # --------------------------------------------
    # ----------- Fitting Functions --------------
    # --------------------------------------------

    def defaultInitialGuess(self):
        """
        Set the default values of the initial guesses.
        """
        self.lineEdit_center.setText("1200, 1108")
        self.lineEdit_angle.setText("0")
        self.lineEdit_ga.setText("1")
        self.lineEdit_gw.setText("200, 200")
        self.lineEdit_tfa.setText("1")
        self.lineEdit_tfw.setText("50, 50")

    def setInitialGuess(self):
        """
        This will be obsolete once better fitting methods roll in. Globally set these initial guesses
        whenever the setInitialGuess() is called.
        """
        # Collect the data from the table. To get a tuple:
        # tuple(int(x.strip()) for x in lineEdit.split(','))
        center_str = self.lineEdit_center.text()
        gw_str = self.lineEdit_gw.text()
        tfw_str = self.lineEdit_tfw.text()

        center = tuple(int(x.strip()) for x in center_str.split(','))
        ang = int(self.lineEdit_angle.text())
        ga = int(self.lineEdit_ga.text())
        gw = tuple(int(x.strip()) for x in gw_str.split(','))
        tfa = int(self.lineEdit_tfa.text())
        tfw = tuple(int(x.strip()) for x in tfw_str.split(','))

        # center = [4 * 300., 4 * 277.]  # Center of the cloud
        # ang = 0.  # Angle with respect to the horizontal, in radians.
        # ga = 1.  # Gaussian amplitude.
        # gw = [200., 200.]  # Gaussian widths of the cloud.
        # tfa = 1.  # Thomas-Fermi amplitude.
        # tfw = [50., 50.]  # Thomas-Fermi radius.
        return ang, center, tfa, tfw, ga, gw

    def fitFunctions(self, pic, mode):
        """ Takes a pic array as an input, and outputs all necessary to draw fits. """
        # self.tableWidget.selectedItems()[4].text()
        date_run = self.tableWidget.selectedItems()[0].text()
        run_run = self.tableWidget.selectedItems()[1].text()
        date = datetime.datetime.now()
        datestring = str(date.year) + "/" + str(date.month).zfill(2) + "/" + str(date.day).zfill(2)
        time = str(date.hour) + ":" + str(date.minute)
        print("\n" + "=" * 50)
        print("{:}, {:}\n:".format(datestring, time))
        print("Beginning analysis on {:} - {:}\n".format(date_run, run_run))
        # Generate empty arrays of the pic size which we will feed into the fitting procedure.
        # Create a 'fitvars' array of x and y coordinates
        x = np.arange(pic.shape[0])
        y = np.arange(pic.shape[1])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        fitvars = np.array([xv, yv]).reshape(2, -1)

        ang, center, tfa, tfw, ga, gw = self.setInitialGuess()  # Call to define the variables (init_guess variables)

        par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'tfamp', 'tfxw', 'tfyw', 'gamp', 'gxw', 'gyw']
        bin_scaling = np.array([1., 1., 1., self.xbin, self.zbin, 1., self.xbin, self.zbin, 1., self.xbin, self.zbin])
        rng_offset = np.array([0., 0., 0., self.xmin, self.zmin, 0., 0., 0., 0., 0., 0.])
        init_guess = np.array([0., 1., ang, center[0], center[1], tfa, tfw[0], tfw[1], ga, gw[0], gw[1]])
        to_physical = np.array(
            [1., 1., 1., self.pixelsize, self.pixelsize, self.prefactor, self.pixelsize, self.pixelsize, self.prefactor,
             self.pixelsize, self.pixelsize])
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
            odrmodel = odr.Model(bimodalmod)  # Store information for the bimodal fitting model

        self.progressBar.setValue(29)

        # Run the ODR Fit procedure.
        odrdata = odr.Data(fitvars[:, ~pic.mask.flatten()], pic.flatten()[~pic.mask.flatten()])
        odrobj = odr.ODR(odrdata, odrmodel, beta0=corr_guess)
        odrobj.set_job(2)  # Ordinary least-sqaures fitting
        odrout = odrobj.run()
        odrout.pprint()

        self.progressBar.setValue(59)

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

        self.progressBar.setValue(70)

        # As the entire output, except for the angle, has to be positive,
        # we take the absolute value of the entire list, then put the angle back in.
        ang_temp = odrout.beta[2]
        odrout.beta = np.abs(odrout.beta)
        odrout.beta[2] = ang_temp % np.pi

        # Converts the fit results to absolute pixel values in the unbinned image.
        fit_results = odrout.beta * bin_scaling + rng_offset
        phys_results = fit_results * to_physical

        # tof = float(self.tableWidget.selectedItems()[4].text()) / 1000  # fml I think I had the wrong number for the first two years of my phd, and was calculating pnums using thold.
        tof = float(self.tableWidget.selectedItems()[3].text()) / 1000
        ncount = -np.log(pic.flatten()).sum() * self.prefactor * self.pixelsize ** 2 * self.xbin * self.zbin #just integrate the entire pic
        ntherm = 0
        ntf = 0
        tx = 0
        tz = 0
        mux = 0
        muz = 0
        mun = 0

        if mode == "gauss":
            ntherm = 2 * np.pi * phys_results[5] * phys_results[6] * phys_results[7]
            tx = 1 / kB * m / 1 * (self.fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * self.fx * np.pi * 2) ** 2)
            tz = 1 / kB * m / 1 * (self.fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * self.fz * np.pi * 2) ** 2)
            mux = m / 1 * (self.fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * self.fx * np.pi * 2) ** 2)
            muz = m / 1 * (self.fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * self.fz * np.pi * 2) ** 2)
            mun = 1.47708846953 * np.power(
                ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * self.fx ** 2 * self.fz, 1. / 3.)))),
                2. / 5.) * hb * np.power(8 * np.pi ** 3 * self.fx ** 2 * self.fz, 1. / 3.)
        if mode == "tf":
            ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]  # 2/5 = 8/15 / (4/3)
            tx = 1 / kB * m / 1 * (self.fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * self.fx * np.pi * 2) ** 2)
            tz = 1 / kB * m / 1 * (self.fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * self.fz * np.pi * 2) ** 2)
            mux = m / 1 * (self.fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * self.fx * np.pi * 2) ** 2)
            muz = m / 1 * (self.fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * self.fz * np.pi * 2) ** 2)
            mun = 1.47708846953 * np.power(
                ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * self.fx ** 2 * self.fz, 1. / 3.)))),
                2. / 5.) * hb * np.power(8 * np.pi ** 3 * self.fx ** 2 * self.fz, 1. / 3.)
        if mode == "bimodal":
            ntf = 2. * np.pi / 5. * phys_results[5] * phys_results[6] * phys_results[7]
            ntherm = 2 * np.pi * phys_results[8] * phys_results[9] * phys_results[10]
            tx = 1 / kB * m / 1 * (self.fx * np.pi * 2 * phys_results[9]) ** 2 / (1 + (tof * self.fx * np.pi * 2) ** 2)
            tz = 1 / kB * m / 1 * (self.fz * np.pi * 2 * phys_results[10]) ** 2 / (1 + (tof * self.fz * np.pi * 2) ** 2)
            mux = m / 1 * (self.fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * self.fx * np.pi * 2) ** 2)
            muz = m / 1 * (self.fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * self.fz * np.pi * 2) ** 2)
            mun = 1.47708846953 * np.power(
                ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * self.fx ** 2 * self.fz, 1. / 3.)))),
                2. / 5.) * hb * np.power(8 * np.pi ** 3 * self.fx ** 2 * self.fz, 1. / 3.)

        ntotal = ntherm + ntf

        self.progressBar.setValue(80)
        self.label_temp.setText(_translate("MainWindow",
                                           "<b>Temperature: </b><br/>") + "T_x = {:.2e} K <br/>T_z = {:.2e} K <br/><br/>".format(
            tx, tz))
        self.label_chem.setText(_translate("MainWindow",
                                           "<b>Chemical Potential:</b><br/>") + "\u03BC_x = {:.2e} <br/>\u03BC_z = {:.2e} <br/> \u03BC_n = {:.2e} <br/><br/>".format(
            mux, muz, mun))
        self.label_pnumber.setText(_translate("MainWindow",
                                              "<b>Particle Number:</b><br/>") +
                                   "Counted = {:.2e} <br/>Thermal Atoms = {:.2e} <br/>Condensed Atoms = {:.2e} <br/><br/>Total Number = {:.2e} ".format(
            ncount, ntherm, ntf, ntotal ))
        self.progressBar.setValue(90)

        # Write out the variables and parameters into fit.param
        with open("/storage/data/{:}/{:}/fit_{:}.param".format(date_run, run_run, mode), "w") as output_file:
            output_file.write("{:}\n".format("=" * 50))
            output_file.write("ANALYSIS MODE {:}\n".format(mode))
            for i in range(0, fit_results.shape[0]):
                output_file.write(par_names[i] + ",%1.5E\n" % fit_results[i])

            for i in range(0, fit_results.shape[0]):
                output_file.write(par_names[i] + "_p,%1.5E\n" % phys_results[i])

            output_file.write("ncount,%1.5E\n" % ncount)
            output_file.write("ntherm,%1.5E\n" % ntherm)
            output_file.write("ntf,%1.5E\n" % ntf)
            output_file.write("ntotal,%1.5E\n" % ntotal)
            output_file.write("tx,%1.5E\n" % tx)
            output_file.write("tz,%1.5E\n" % tz)
            output_file.write("mux,%1.5E\n" % mux)
            output_file.write("muz,%1.5E\n" % muz)
            output_file.write("mun,%1.5E\n" % mun)
        self.progressBar.setValue(100)

        return fitresult, fitguess, fitresultgauss, fitresulttf

    # def OAH_analysis(self, num, dz_focus):
    #     # Do stuff to the pic file
    #     date = self.tableWidget.selectedItems()[0].text()
    #     run = self.tableWidget.selectedItems()[1].text()
    #     output = HI_refocus(date, run, num, dz_focus, )
    #     return output


    # --------------------------------------------
    # ------------ Further Analysis --------------
    # --------------------------------------------

    def setDatesFA(self):
        # Generate a combox with a list of dates
        path = self.path
        self.dates = []
        self.dates = [x for x in os.listdir(path) if len(x) == 8]
        self.dates.sort(reverse=True)
        self.comboBox_dateFA.clear()
        self.comboBox_dateFA.addItems(self.dates)

    def setRunsFA(self):
        # Generate a combox with a list of runs
        path = self.path + str(self.comboBox_dateFA.currentText()) + "/"
        self.runs = []
        self.runs = [x for x in os.listdir(path) if len(x) == 4]
        self.runs.sort(reverse=True)
        self.comboBox_runFA.clear()
        self.comboBox_runFA.addItems(self.runs)

    def oldAnalysis(self):
        date = str(self.comboBox_dateFA.currentText())
        run = str(self.comboBox_runFA.currentText())
        mode = self.lineEdit_mode.text()
        os.system('python3 ../analyze.py {:} {:} -M {:}'.format(date, run, mode))

    def HI_refocus_preview(self):
        date = str(self.comboBox_dateFA.currentText())
        run = str(self.comboBox_runFA.currentText())
        num = self.lineEdit_num.text()
        refocus_str = self.lineEdit_focuslims.text()
        refocus = tuple(float(x.strip()) for x in refocus_str.split(','))
        os.system(
            'python3 ../analyze_OAH.py {:} {:} -M refocus_compare --shotnr {:} --focuslims {:} {:}'.format(date, run,
                                                                                                           num,
                                                                                                           refocus[0],
                                                                                                           refocus[1]))

    def makeAFits(self):
        date = str(self.comboBox_dateFA.currentText())
        run = str(self.comboBox_runFA.currentText())
        focus = self.lineEdit_focus.text()
        os.system('python3 ../analyze_OAH.py {:} {:} -M refocus_save '.format(date, run, focus))

    def makeAVideo(self):
        date = str(self.comboBox_dateFA.currentText())
        run = str(self.comboBox_runFA.currentText())
        focus = self.lineEdit_focus.text()
        # os.system('python3 ../analyze_OAH.py {:} {:} -M refocus_compare --shotnr {:} --focuslims {:} {:}'.format(date, run, num, refocus[0], refocus[1]))

    def clearFurtherAnalysis(self):
        self.comboBox_dateFA.clear()
        self.comboBox_runFA.clear()
        self.lineEdit_mode.setText("")
        self.lineEdit_num.setText("")
        self.lineEdit_focus.setText("")
        self.lineEdit_focuslims.setText("")
        self.lineEdit_runrange.setText("")

    # --------------------------------------------
    # ----------- Gallery Functions --------------
    # --------------------------------------------

    def setDatesGallery(self):
        # Generate a combox with a list of dates
        path = self.path
        self.dates = []
        self.dates = [x for x in os.listdir(path) if len(x) == 8]
        self.dates.sort(reverse=True)
        self.combogallery.clear()
        self.combogallery.addItems(self.dates)

    def galleryDisplay(self):
        date = str(self.combogallery.currentText())
        print(self.combogallery.currentText())
        print(date)
        os.system('gallery {:}'.format(date))

    # --------------------------------------------
    # ------------- Search Function --------------
    # --------------------------------------------

    def searchFunction(self):
        searchText = self.searchLineEdit.text()
        self.searchListWidget.clear()
        output = []
        for filename in sorted(os.listdir('/storage/BECViewer/comments/')):
            with open('/storage/BECViewer/comments/' + filename, "r") as file_one:
                print(f"Opening {filename}")
                patrn = searchText
                for line in file_one:
                    if re.search(patrn, line):
                        output.append("{:}:   {:}".format(filename[0:13], line))
        print("Done searching")
        output.sort(reverse=True) # to sort it by date
        self.searchListWidget.addItems(output)

    def takeMeToComment(self):
        date = self.searchListWidget.currentItem().text()[0:8]
        run = self.searchListWidget.currentItem().text()[9:13]
        self.updateRunList(date)
        # Select the correct row
        rowlist = []
        itemlist = []
        for row in range(self.tableWidget.rowCount()):
            _item = self.tableWidget.item(row, 1)
            if _item:
                item = self.tableWidget.item(row, 1).text()
                rowlist.append(row)
                itemlist.append(int(item))
        indx = (rowlist[itemlist.index(int(run))])
        self.tableWidget.selectRow(indx)
        self.tabWidget.setCurrentIndex(0)



    # --------------------------------------------
    # ------------ Secret Functions --------------
    # --------------------------------------------

    def secretFunction(self):
        self.sc += 1
        if self.sc > 7:
            self.copy_bttn.show()

    def secretFunction2(self):
        self.sc2 += 1
        if self.sc < 3:
            self.copy_bttn.setText("Again.")
        elif self.sc2 < 5:
            self.copy_bttn.setText("Harder.")
        elif self.sc2 < 7:
            self.copy_bttn.setText("HARDER.")
        elif self.sc2 < 20:
            self.copy_bttn.setText("Keep on going.")
        elif self.sc2 < 25:
            self.copy_bttn.setText("That's it.")
        else:
            self.saveas_bttn.show()

    def secretFunction3(self):
        import webbrowser
        self.copy_bttn.hide()
        self.saveas_bttn.hide()
        webbrowser.open('https://www.youtube.com/watch?v=xvFZjo5PgG0')

    # --------------------------------------------
    # ---- Widget structure display functions ----
    # --------------------------------------------

    def frameDisplay(self):
        """ Displays or hides the main analysis-specs frame on the second tab. """
        if self.analysis_frame.isHidden():
            self.showFrame()
        else:
            self.hideFrame()

    def showFrame(self):
        self.analysis_frame.show()
        self.menu_checkbttn.setText("<")
        self.analysis_side_label.setText("")

    def hideFrame(self):
        # Hide the frame
        self.analysis_frame.hide()
        self.menu_checkbttn.setText(">")
        self.analysis_side_label.setText("L\nI\nV\nE\n\nA\nN\nA\nL\nY\nS\nI\nS")
        self.analysis_side_label.setStyleSheet("font-size:16pt; font-weight:600; color:#a2afb3;")

    def displayInfo(self, state):
        """ Display info when the info check button is checked. """

        try:
            date = self.tableWidget.selectedItems()[0]
            run = self.tableWidget.selectedItems()[1]
            if state == QtCore.Qt.Checked:
                self.label_info.setText(self.getInfo(date.text(), run.text()))
                info2_text = self.getInfo2(date.text(), run.text()) + "\n" + self.getInfo_ODT(date.text(), run.text())
                self.label_info2.setText(info2_text)
                self.label_info.setFont(QtGui.QFont('monospace', 10))
                self.label_info2.setFont(QtGui.QFont('monospace', 10))
                self.hideFrame()
            else:
                self.label_info.setText(" RUN ID : {:} / {:}".format(date.text(), run.text()))
                self.label_info.setFont(QtGui.QFont('monospace', 10))
                self.label_info2.setText(" ")

        except:
            if state == QtCore.Qt.Checked:
                self.label_info.setText("Please select a valid entry.")
                self.label_info.setFont(QtGui.QFont('monospace', 10))
            else:
                self.label_info.setText(" ")
            pass

    def showComment(self, state):
        if state == QtCore.Qt.Checked:
            self.textEdit_comment.show()
            self.buttonBox_saveComment.show()
            #self.pushButton_logBook2.show()
        else:
            self.textEdit_comment.hide()
            self.buttonBox_saveComment.hide()
            #self.pushButton_logBook2.hide()

    # --------------------------------------------
    # --- Other analysis and processing files ----
    # --------------------------------------------

    def get_parameter(self, date, seq, paramname):
        """
        Get the value of the parameter form the parameters.param file. Very often used, might be better to import
        it from the pcgk/fits file. But it might be better to include those functions here.
        """
        date = str(date)
        run_id = str(seq).zfill(4)
        param = "N/A"
        try:
            with open(self.path + date + '/' + run_id + '/parameters_mod.param') as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                for row in csvreader:
                    if row[0] == paramname:
                        param = float(row[1])
        except:
            try:
                with open(self.path + date + '/' + run_id + '/parameters.param') as paramfile:
                    csvreader = csv.reader(paramfile, delimiter=',')
                    for row in csvreader:
                        if row[0] == paramname:
                            param = float(row[1])
            except:
                param = "N/A"
        return param

    def setVariables(self):
        """ Globally set these varialbes whenever the setVariables() is called. """
        self.fx = float(self.lineEdit_radfreq.text())
        self.fz = float(self.lineEdit_axfreq.text())
        self.wavelength = float(self.lineEdit_wavelength.text())
        self.detuning = float(self.lineEdit_detuning.text())
        self.pixelsize = float(self.lineEdit_pixelsize.text())
        self.prefactor = float(
            1*(1 + 4 * (float(self.detuning) ** 2)) * 2 * np.pi / (3 * (float(self.wavelength) ** 2)) * 18. / 5.)

    def get_fit_parameter(self, date, seq, mode):
        """
        Get the value of the fitted parameters, such as the number of atoms or temperature
        from the fit_mode.param file. Get all of them, and append the values in an array.
        """
        date = str(date)
        run_id = str(seq).zfill(4)
        mode = mode
        params = []
        with open(self.path + date + '/' + run_id + '/fit_' + mode + '.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                params.append(float(row[1]))
        return params

    def get_n(self, date, seq):
        """
        Get the number of thermal atoms. Made a separate one, to account for different modes.
        """
        date = str(date)
        run_id = str(seq).zfill(4)
        gausspath = self.path + date + '/' + run_id + '/fit_gauss.param'
        tfpath = self.path + date + '/' + run_id + '/fit_tf.param'
        bipath = self.path + date + '/' + run_id + '/fit_bimodal.param'

        if os.path.isfile(bipath):
            with open(bipath) as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    if row[0] == "ntotal":
                        ntots = float(row[1])
                    if row[0] == "ntf":
                        ntf = float(row[1])
                if ntots == 0:
                    ntherm = " --- empty --- "
                else:
                    ntherm = "{:0.3e} @ {:.1f}%".format(ntots, ntf / ntots * 100)
            clr = QtGui.QColor(252, 238, 212)

        elif os.path.isfile(gausspath):
            with open(gausspath) as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    if row[0] == "ntherm":
                        if float(row[1]) == 0:
                            ntherm = " --- empty --- "
                        else:
                            ntherm = "{:0.3e}".format(float(row[1]))
            clr = QtGui.QColor(214, 230, 255)

        elif os.path.isfile(tfpath):
            with open(tfpath) as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    if row[0] == "ntf":
                        ntherm = "{:0.3e}".format(float(row[1]))
            clr = QtGui.QColor(207, 250, 222)

        else:
            ntherm = "N/A"
            clr = QtGui.QColor(250, 250, 250)

        return ntherm, clr

    def get_temp(self, date, seq):
        """
        Get the temperature. Made a separate one, to account for different modes.
        """
        date = str(date)
        run_id = str(seq).zfill(4)
        gausspath = self.path + date + '/' + run_id + '/fit_gauss.param'
        tfpath = self.path + date + '/' + run_id + '/fit_tf.param'
        bipath = self.path + date + '/' + run_id + '/fit_bimodal.param'

        if os.path.isfile(bipath):
            with open(bipath) as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    if row[0] == "tx":
                        tx = float(row[1])
                    if row[0] == "tz":
                        tz = float(row[1])

        elif os.path.isfile(gausspath):
            with open(gausspath) as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    if row[0] == "tx":
                        tx = float(row[1])
                    if row[0] == "tz":
                        tz = float(row[1])

        elif os.path.isfile(tfpath):
            with open(tfpath) as paramfile:
                csvreader = csv.reader(paramfile, delimiter=',')
                next(csvreader)
                next(csvreader)
                for row in csvreader:
                    if row[0] == "tx":
                        tx = float(row[1])
                    if row[0] == "tz":
                        tz = float(row[1])

        else:
            tx = 0
            tz = 0

        return tx, tz

    def saveComment(self, item):
        """
        Save the comment, if the 'comment' item has been modified, but not
        while loading the full table.
        """
        if item.column() == 9 and self.Loading:
            date = self.tableWidget.item(item.row(), 0).text()
            run = self.tableWidget.item(item.row(), 1).text()
            with open("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run), "w") as text_file:
                text_file.write(item.text())

    def saveComment_analyse(self):
        """
        The save comment function for in the analysis mode.
        """
        date = self.date
        run = self.run
        with open("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run), "w") as text_file:
            text_file.write(self.textEdit_comment.toPlainText())

    def get_comment(self, date, run):
        """ Fetch the comment, if there is one """
        if os.path.exists("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run)):
            with open("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run), "r") as commentFile:
                comment = commentFile.read()
        else:
            comment = ""
        return comment

    def get_analysis(self, date, seq):
        """ Return the type of analysis previously done of a run. Simply reads the output images. """
        date = str(date)
        run_id = str(seq).zfill(4)
        path = self.path + date + "/" + run_id + "/"
        output = ""
        count = 0
        for file in os.listdir(path):
            if file.endswith(".png"):
                if file[-5].isdigit():  # new file saves include run number in the name for gallery view
                    if file[:-9] != 'viewer':
                        if count > 0:
                            output += ", "
                        output += file[:-9]
                        count += 1
                else:
                    if file[:-4] != 'viewer':
                        if count > 0:
                            output += ", "
                        output += file[:-4]
                        count += 1
        if len(output) == 0:
            output = "/"
        return output

    def get_OAH_analysis_foci(self, date, seq):
        """ Return the type of analysis previously done of a run. Simply reads the output images. """
        date = str(date)
        run_id = str(seq).zfill(4)
        path = self.path + date + "/" + run_id + "/"
        output_text = "\n"
        for file in os.listdir(path):
            if file[:8] == "pics_foc":
                output_text += f"| "
                for el in file.split('_')[2:]:
                    output_text += f" {el} |"
                output_text = output_text[:-7] + " |"
                output_text += "\n"
        return output_text

    def get_numofshots(self, date, seq):
        """ Return the number of images for a particular scan. Very heavy and slow for many runs. """
        if self.checkBox_multi_img.isChecked():
            path = '/storage/data/' + str(date) + '/'
            image = str(seq).zfill(4) + '/'
            full_path = path + image

            # If the preprocessing was done before, take that file
            fits_files = [f for f in os.listdir(path + image) if (f.endswith('.fits') and len(f[:-5]) == 1)]
            fits_files = np.sort(fits_files)

            multi_array = []
            for file in fits_files:
                multi_array.append(pyfits.open(path + image + file)[0].data.astype(float)[:])
            nr = len(np.concatenate(multi_array)) - 4
        else:
            date = str(date)
            run_id = str(seq).zfill(4)
            path = self.path + date + "/" + run_id + "/"
            nr = len(pyfits.open(path + '0.fits')[0].data.astype(float))
        return nr

    def getInfo(self, date, seq):
        """ Get information about each run from the param files. """
        date = str(date)
        run_id = str(seq).zfill(4)
        output = "Information for date {:}, run ID {:}:\n{:}\n".format(date, run_id, "-" * 84)
        with open(self.path + date + '/' + run_id + '/parameters.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            countr = 0
            for row in csvreader:
                output += '{:<15} | {:>10}   ||   '.format(row[0], round(float(row[1]), 8))
                countr += 1
                if countr % 3 == 0:
                    output += '\n'
            output += '\n'
            output += "-" * 84
        return output

    def getInfo2(self, date, seq):
        """ Get additional information about previous analysis.
            I thought it might be useful, but it's not really.
        """
        date = int(date)
        seq = int(seq)

        def find_closest_date(date):
            all_sf_dates = [int(a) for a in sorted(os.listdir("/storage/spinflip_log"), reverse=True)]
            all_dates = [d for d in all_sf_dates if d <= date]
            if len(all_dates) != 0:
                closest_date = max(all_dates)
                return closest_date
            else:
                return None

        def find_closest_run(run, date):
            closest_date = find_closest_date(date)

            if closest_date == None:
                return [None, None]

            elif date != closest_date:
                # If not the same date, then find the last one from the closest date
                all_sf_runs = []
                with open(f"/storage/spinflip_log/{closest_date}") as file:
                    for specs in file.read().split(f"{'-' * 71}\n\n\n")[:-1]:
                        all_sf_runs.append(int(specs[41:45]))
                    closest_run = max(all_sf_runs)

            elif date == closest_date:
                # If the same date, there are two options -> then find the last one before the run
                all_sf_runs = []
                with open(f"/storage/spinflip_log/{closest_date}") as file:
                    for specs in file.read().split(f"{'-' * 71}\n\n\n")[:-1]:
                        all_sf_runs.append(int(specs[41:45]))

                if min(all_sf_runs) >= run:
                    print("Need to go to the previous session")
                    closest_run, closest_date = find_closest_run(float('inf'), date - 1)
                else:
                    closest_run = max([r for r in all_sf_runs if r <= seq])

            return closest_run, closest_date

        def find_specs(run, date):
            closest_run, closest_date = find_closest_run(run, date)

            if closest_date == None:
                return "N/A"

            else:
                sp_str = ''
                with open(f"/storage/spinflip_log/{closest_date}") as file:
                    for specs in file.read().split(f"{'-' * 71}\n\n\n")[:-1]:
                        if int(specs[41:45]) == int(closest_run):
                            sp_str = specs
                return sp_str, closest_date, closest_run


        sp_str, closest_date, closest_run = find_specs(seq, date)
        sp_form = [s.split(" ") for s in sp_str.split("\n")]
        header = " ".join(sp_form[0])

        output = f"{'-' * 16} SPINFLIP Parameters set on {closest_date}-{str(closest_run).zfill(4)} {'-' * 16}\n"
        # output += f"{header}\n\n"
        for s in sp_form[1:-1]:
            first_string = " ".join(s[:-2])
            output += f"{first_string : <50}{s[-2] : >10} {s[-1]}\n"
        # output += f"{'-' * 63}"

        return output

    def getInfo_ODT(self, date, seq):
        """ Get additional information about the ODTtrap
        """
        date = int(date)
        seq = int(seq)

        def find_closest_date(date):
            all_sf_dates = [int(a[:-4]) for a in sorted(os.listdir("/storage/ODT_setuplog"), reverse=True)]
            all_dates = [d for d in all_sf_dates if d <= date]
            if len(all_dates) != 0:
                closest_date = max(all_dates)
                return closest_date
            else:
                return None

        def find_closest_run(run, date):
            closest_date = find_closest_date(date)

            if closest_date == None:
                return [None, None]

            elif date != closest_date:
                # If not the same date, then find the last one from the closest date
                all_sf_runs = []
                with open(f"/storage/ODT_setuplog/{closest_date}.txt") as file:
                    for specs in file.read().split("------------------------------------\n")[1:]:
                        all_sf_runs.append(int(specs.split(" ")[10]))
                    closest_run = max(all_sf_runs)

            elif date == closest_date:
                # If the same date, there are two options -> then find the last one before the run
                all_sf_runs = []
                with open(f"/storage/ODT_setuplog/{closest_date}.txt") as file:
                    for specs in file.read().split("------------------------------------\n")[1:]:
                        # print(specs.split(" ")[10])
                        all_sf_runs.append(int(specs.split(" ")[10]))

                if min(all_sf_runs) >= run:
                    print("Need to go to the previous session")
                    closest_run, closest_date = find_closest_run(float('inf'), date - 1)
                else:
                    closest_run = max([r for r in all_sf_runs if r <= seq])

            return closest_run, closest_date
        
        def find_specs(run, date):
            closest_run, closest_date = find_closest_run(run, date)

            if closest_date == None:
                return "N/A"

            else:
                sp_str = ''
                with open(f"/storage/ODT_setuplog/{closest_date}.txt") as file:
                    for specs in file.read().split("------------------------------------\n")[1:]:
                        if int(specs.split(" ")[10]) == int(closest_run):
                            sp_str = specs
                return sp_str, closest_date, closest_run


        sp_str, closest_date, closest_run = find_specs(seq, date)


        sp_form = [s.split(" ") for s in sp_str.split("\n")]
        header1 = " ".join(sp_form[0])
        header2 = " ".join(sp_form[1])
        header3 = " ".join(sp_form[2])

        output = f"{'-' * 16} ODT SCAN Parameters set on {closest_date}-{str(closest_run).zfill(4)} {'-' * 16}\n"
        output += f"{header1[4:-4]}\n"
        output += f"{header2[4:-4]}\n"
        # output += f"{header3}\n\n"
        for s in sp_form[3:-3]:
            first_string = " ".join(s[:-2])
            output += f"{first_string : <50}{s[-2] : >10} {s[-1]}\n"
        output += f"{'-' * 63}"

        return output



    # --------------------------------------------
    # ----------- Watcher & Updater --------------
    # --------------------------------------------

    def quickLiveUpdate(self, state):
        if state:
            self.spinBox_updateTime.setValue(1)
            self.liveUpdate()
        else:
            self.spinBox_updateTime.setValue(0)
            self.liveUpdate()

    def latestRunToday(self):
        # Save the largest runID, to be used for comparison for the live update
        # Get today's date, formatted
        date = datetime.datetime.now()
        datestring = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        path = "/storage/data/" + datestring + "/"
        if os.path.exists(path):
            dirList = [x for x in os.listdir(path) if os.path.isdir(path + x)]
            if len(dirList) > 0:
                output = sorted(dirList)[-1]
            else:
                output = None
        else:
            output = None
        return datestring, output

    def liveUpdate(self):
        """
        On begining the live update, check for the latest run. During the live update,
        every interval check call update function, which will check for the
        2.fits file in the subsequent folder.
        """
        self.datetimetoday, self.latestRun = self.latestRunToday()
        if self.latestRun == None:
            self.latestRun = -1
        self.nextRun = str(int(self.latestRun) + 1).zfill(4)
        if self.spinBox_updateTime.value() != 0:
            print("Updating now ...")
            self.i = 0
            self.interval = int(self.spinBox_updateTime.value())
            self.qTimer = QtCore.QTimer()
            self.qTimer.setInterval(self.interval * 1000)
            self.qTimer.timeout.connect(self.updateFunction)
            self.qTimer.start()
        else:
            try:
                print("Live update mode - STOPPED.")
                self.qTimer.stop()
            except:
                pass

    def updateFunction(self):
        """
        Check if the file 2.fits appears in the folder one larger than the highest numbered folder
        available when we started the update. If it had, update the table, and uploade the plot with
        that file. The moment it exists, call the liveUpdate function again, which will reset the
        latestRun and nextRun.
        If the Auto Fit button is not checked, set the fit mode combobox to viewer.
        """
        # if os.path.exists("/storage/data/" + self.datetimetoday + "/" + self.nextRun + "/2.fits"):
        # The parameters.param file appears later, so it seems smarter to use that one - we need it for analysis.
        if os.path.exists(self.path + self.datetimetoday + "/" + self.nextRun + "/parameters.param"):
            self.qTimer.stop()
            self.updateDateList()
            # self.initialParams()
            self.updateRunList(self.datetimetoday)
            # Select the corresponding row - this is important for refreshing the plot later on
            rowlist = []
            itemlist = []
            for row in range(self.tableWidget.rowCount()):
                _item = self.tableWidget.item(row, 1)
                if _item:
                    item = self.tableWidget.item(row, 1).text()
                    rowlist.append(row)
                    itemlist.append(int(item))
            indx = (rowlist[itemlist.index(max(itemlist))])
            self.tableWidget.selectRow(indx)
            if not self.checkBox_autoFit.isChecked():
                self.comboBox_mode.setCurrentText("Viewer")
            self.updatePlot()
            self.liveUpdate()
        else:
            pass



app = QtWidgets.QApplication(sys.argv)
mainWindow = BEC_LIVE_PLOT_APP()
mainWindow.show()
sys.exit(app.exec_())
