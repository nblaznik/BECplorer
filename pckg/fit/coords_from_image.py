from matplotlib.widgets import EllipseSelector, RectangleSelector
from OAH_refocus import *
import numpy as np
import matplotlib.pyplot as plt

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


parser = argparse.ArgumentParser()
parser.add_argument('date', help='Date at which the shot to analyze was made.', metavar='YYYYMMDD')
parser.add_argument('shot', help='Number of the shot in question.', metavar='XXXX')
parser.add_argument('-M', help='Fit mode selection. Choices are "nofit", "bimodal", "gauss" and "tf".', default='bimodal', metavar='mode')

args, unknown = parser.parse_known_args(sys.argv[1:])

input_folder = '/storage/data/' + args.date + '/' + args.shot + '/'

# Check for refocused one and display the 0th one
files = []
for file in os.listdir(input_folder):
    if file.startswith("pics"):
        files.append(file)


if args.M == "normal":
    atom = pyfits.open(input_folder + '0.fits')[0].data.astype(float)[0]  # .mean(axis=0)
    flat = pyfits.open(input_folder + '1.fits')[0].data.astype(float)[0]  # .mean(axis=0)
    dark = pyfits.open(input_folder + '2.fits')[0].data.astype(float).mean(axis=0)
    pic = (atom - dark) / (flat - dark)
    vmin = 0
    vmax = 1.3

else:

    if len(files) == 0:
        # atom = pyfits.open(input_folder + '0.fits')[0].data.astype(float)[0]  # .mean(axis=0)
        # flat = pyfits.open(input_folder + '1.fits')[0].data.astype(float)[0]  # .mean(axis=0)
        # dark = pyfits.open(input_folder + '2.fits')[0].data.astype(float).mean(axis=0)
        # pic = (atom - dark) / (flat - dark)
        pic = HI_refocus(
            date=args.date, shot=args.shot, num=0, dz_focus=0.0, quad="quad1", cut=(1, -1, 1, -1)
        )
        vmin = -1
        vmax = 1

    else:
        pics = pyfits.open(input_folder + files[0])[0].data.astype(float).mean(axis=0)
        pic = pics
        vmin = -3 #pic.min()
        vmax = 3 #pic.max()


    if args.M == "PCA":
        pic = pic[5:-5, 50:-50]





def select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    global x1, x2, y1, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    title.set_text("Coordinates Recorded: \n X = {:} - {:}, Y = {:} - {:} \nPress Enter to save.".format(round(x1), round(x2), round(y1), round(y2)))
    fig.canvas.draw_idle()

def toggle_selector(event):
    print('Key pressed.')
    print(event.key)
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
    elif event.key == ' ':
        # print("Enter pressed")
        plt.close()

fig, ax = plt.subplots()

ax.imshow(pic, vmin=vmin, vmax=vmax, cmap='afmhot_r', interpolation='none', origin="lower", aspect='auto')

N = 100000  # If N is large one can see improvement by using blitting.
x = np.linspace(0, 10, N)



if args.M == "notusePCA":
    print(f"{args.date}_{int(args.shot)-1}.npy")
    if os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{args.date}_{str(int(args.shot)-1).zfill(4)}.npy"):
        # Load the coordinates from the previous shot
        y1, y2, x1, x2 = np.load(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{args.date}_{str(int(args.shot)-1).zfill(4)}.npy", allow_pickle=True)
        print()
        selectors = []
        for selector_class in [RectangleSelector]:
            selector = selector_class(
                ax, select_callback,
                useblit=True,
                button=[1, 3],  # disable middle button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            selector.extents = (x1+10, x2-10, y1+10, y2-10)
            selectors.append(selector)

        fig.canvas.mpl_connect('key_press_event', toggle_selector)

    else: 
        print("No previous coordinates found. Please select a region.")
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



else: 
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


title = ax.set_title("Make a selection and \n press Enter to save the coordinates.")
fig.canvas.draw_idle()
plt.show()


# coords = [[x1, x2, y1, y2], [centerx, centery], [wx, wy]]
# coords = [[int(x1), int(x2), int(y1), int(y2)], [int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)], [int(0.2*(x2-x1)), int(0.2*(x2-x1))]]
coords = [[int(y1), int(y2), int(x1), int(x2)], [int(y1+(y2-y1)/2), int(x1+(x2-x1)/2), None, None], [int(0.2*(x2-x1)), int(0.2*(x2-x1)), None, None]]
print(coords)

if args.M == "OAHcooling":
    np.save("/storage/BECViewer/variables/fittingparamsOAH.npy", coords)

elif args.M == "just_function":
    np.save("/home/bec_lab/Desktop/arrays/temp/temp.npy", coords)

elif args.M == "PCA":
    print("PCA Dimensions:")
    print("wtop, wbottom, wleft, wright = {:}, {:}, {:}, {:}".format(round(y1), round(y2), round(x1), round(x2)))
    np.save(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{args.date}_{args.shot}.npy", [round(y1-10), round(y2+10), round(x1-10), round(x2+10)], allow_pickle=True)

else:
    np.save("/storage/BECViewer/variables/fittingparams.npy", coords)
