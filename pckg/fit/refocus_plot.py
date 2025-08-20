#!/usr/bin/env python3

from OAH_refocus import *
import matplotlib.pyplot as plt 
from OAHDEV_functions import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import savgol_filter
import os
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

# Pretty Matplotlib Text
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Done shots 63, 84, 90, 96, 98
all_shots = []
all_dates = []

os.system('clear')
print("Available dates and shots:")
i = 0
for x in sorted(os.listdir("/home/bec_lab/Desktop/imgs/focus_shift_slider/")): 
    if x[-5:] == '1.npy':
        print(f"[{i}]: ", x[:8], "-", x[9:-9])
        all_dates.append(x[:8])
        all_shots.append(x[9:-9])
        i += 1


sel = int(input("Select the index of the run you want. \n"))

date = all_dates[sel]
shot = all_shots[sel]

os.system('clear')
print("Available dates and shots:")
i = 0
for x in sorted(os.listdir("/home/bec_lab/Desktop/imgs/focus_shift_slider/")): 
    if x[-5:] == '1.npy':
        if i == sel: 
            print(f"[x]: ", x[:8], "-", x[9:-9])
            i += 1
        else: 
            print(f"[{i}]: ", x[:8], "-", x[9:-9])
            i += 1



if shot in [96]:
    num_frames = 101  # Change this to match the number of focus steps
    focus_values = np.linspace(-0.04, 0.04, num_frames)

elif shot in [98, 111, 112]: 
    num_frames = 51  # Change this to match the number of focus steps
    focus_values = np.linspace(-0.04, 0.04, num_frames)

else:
    num_frames = 51  # Change this to match the number of focus steps
    focus_values = np.linspace(-0.02, 0.02, num_frames)


if os.path.exists(f"/home/bec_lab/Desktop/imgs/focus_shift_slider/{date}_{shot}_ang1.npy"):
    print(f"Processing for {date} - {shot} done before.")
    ang1 = np.load(f"/home/bec_lab/Desktop/imgs/focus_shift_slider/{date}_{shot}_ang1.npy")
    ang2 = np.load(f"/home/bec_lab/Desktop/imgs/focus_shift_slider/{date}_{shot}_ang2.npy")
else:
    print(f"Processing for {date} - {shot} has not been done before. Run 'refocus_save.py' first.")
    print(f"Other available date/shots combinations")
    for x in os.listdir("/home/bec_lab/Desktop/imgs/focus_shift_slider/"): 
        if x[-5:] == '1.npy':
            print(x[:8], "-", x[9:11])
    quit()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

# Create figure
fig = plt.figure(figsize=(17, 7))
gs = GridSpec(3, 6, height_ratios=[17, 1, 1], figure=fig)

# Create image axes
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])

# Share axes
ax1.sharex(ax2)
ax1.sharey(ax2)

# Initial indices
idx1 = num_frames // 2
idx2 = num_frames // 2


xc1 = 500
xc2 = 500
# Display images
im1 = ax1.imshow(ang1[idx1][849:1384, 920:2222], vmin=-0.5, vmax=1.2, cmap='afmhot_r', aspect='auto')
im2 = ax2.imshow(ang2[idx2][849:1384, 920:2222], vmin=-0.5, vmax=1.2, cmap='afmhot_r', aspect='auto')
l1,  = ax3.plot(range(len(ang1[idx1][849:1384, 920+xc1:2222-xc2])), ang1[idx1][849:1384, 920+xc1:2222-xc2].mean(axis=1), alpha=0.4, c='C0')
l2,  = ax3.plot(range(len(ang2[idx2][849:1384, 920+xc1:2222-xc2])), ang2[idx2][849:1384, 920+xc1:2222-xc2].mean(axis=1), alpha=0.4, c='C1')

# Set titles
ax1.set_title(f"Quad1 - focus1 = {round(focus_values[idx1], 3)}")
ax2.set_title(f"Quad2 - focus2 = {round(focus_values[idx2], 3)}")
plt.suptitle(f"{date}-{str(shot).zfill(4)}")

# Create slider axes
ax_slider1 = fig.add_subplot(gs[1, 0:3])
ax_slider2 = fig.add_subplot(gs[1, 3:6])
ax_slider3 = fig.add_subplot(gs[2, :])

# Remove ticks from slider axes
for ax in [ax_slider1, ax_slider2, ax_slider3]:
    ax.set_xticks([])
    ax.set_yticks([])

# Create sliders
slider1 = Slider(ax_slider1, "Q1", 0, num_frames - 1, valinit=num_frames//2, valstep=1)
slider2 = Slider(ax_slider2, "Q2", 0, num_frames - 1, valinit=num_frames//2, valstep=1)
slider3 = Slider(ax_slider3, "Q1+Q2", 0, num_frames - 1, valinit=num_frames//2, valstep=1)
slider1.valtext.set_visible(False)
slider2.valtext.set_visible(False)
slider3.valtext.set_visible(False)

# Update functions
def update1(val):
    idx1 = int(slider1.val)
    im1.set_data(ang1[idx1][849:1384, 920:2222])
    l1.set_ydata(ang1[idx1][849:1384, 920+xc1:2222-xc2].mean(axis=1))
    ax1.set_title(f"Quad1 - focus1 = {round(focus_values[idx1], 4)}")
    fig.canvas.draw_idle()

def update2(val):
    idx2 = int(slider2.val)
    im2.set_data(ang2[idx2][849:1384, 920:2222])
    l2.set_ydata(ang2[idx2][849:1384, 920+xc1:2222-xc2].mean(axis=1))
    ax2.set_title(f"Quad2 - focus2 = {round(focus_values[idx2], 4)}")
    fig.canvas.draw_idle()

def update3(val):
    idx3 = int(slider3.val)
    im1.set_data(ang1[idx3][849:1384, 920:2222])
    im2.set_data(ang2[idx3][849:1384, 920:2222])
    l1.set_ydata(ang1[idx3][849:1384, 920+xc1:2222-xc2].mean(axis=1))
    l2.set_ydata(ang2[idx3][849:1384, 920+xc1:2222-xc2].mean(axis=1))
    ax1.set_title(f"Quad1 - focus1 = {round(focus_values[idx3], 4)}")
    ax2.set_title(f"Quad2 - focus2 = {round(focus_values[idx3], 4)}")
    fig.canvas.draw_idle()

# Connect sliders
slider1.on_changed(update1)
slider2.on_changed(update2)
slider3.on_changed(update3)

plt.show()




# # Create figure and axes
# fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
# plt.subplots_adjust(bottom=0.3)  # Leave space for sliders

# idx1 = num_frames//2
# idx2 = num_frames//2
# im1 = ax[0].imshow(ang1[idx1][849:1384, 920:2222], vmin=-0.5, vmax=1.2, cmap='afmhot_r', aspect='auto')
# im2 = ax[1].imshow(ang2[idx2][849:1384, 920:2222], vmin=-0.5, vmax=1.2, cmap='afmhot_r', aspect='auto')
# im3 = ax[2].plot(range(len(ang2[idx2][849:1384, 920:2222])), ang2[idx2][849:1384, 920:2222].mean(axis=1))
# ax[0].set_title(f"Quad1 - focus1 = {round(focus_values[idx1], 3)}")
# ax[1].set_title(f"Quad2 - focus2 = {round(focus_values[idx2], 3)}")


# plt.suptitle(f"{date}-{str(shot).zfill(4)}")
# # Create slider axes below each subplot
# ax_slider1 = plt.axes([0.15, 0.1, 0.3, 0.03])  # Slider for Quad1
# ax_slider2 = plt.axes([0.55, 0.1, 0.3, 0.03])  # Slider for Quad2
# ax_slider3 = plt.axes([0.15, 0.01, 0.7, 0.03])  # Slider for Both

# # left, bottom, width, height

# # Sliders
# slider1 = Slider(ax_slider1, "Quad1", 0, num_frames - 1, valinit=num_frames//2, valstep=1)
# slider2 = Slider(ax_slider2, "Quad2", 0, num_frames - 1, valinit=num_frames//2, valstep=1)
# slider3 = Slider(ax_slider3, "Q1+Q2", 0, num_frames - 1, valinit=num_frames//2, valstep=1)


# # Update functions
# def update1(val):
#     idx1 = int(slider1.val)
#     im1.set_data(ang1[idx1][849:1384, 920:2222])
#     fig.canvas.draw_idle()
#     ax[0].set_title(f"Quad1 - focus1 = {round(focus_values[idx1], 4)}")

# def update2(val):
#     idx2 = int(slider2.val)
#     im2.set_data(ang2[idx2][849:1384, 920:2222])
#     fig.canvas.draw_idle()
#     ax[1].set_title(f"Quad2 - focus2 = {round(focus_values[idx2], 4)}")

# def update3(val):
#     idx3 = int(slider3.val)
#     im1.set_data(ang1[idx3][849:1384, 920:2222])
#     im2.set_data(ang2[idx3][849:1384, 920:2222])
#     fig.canvas.draw_idle()
#     ax[0].set_title(f"Quad1 - focus1 = {round(focus_values[idx3], 4)}")
#     ax[1].set_title(f"Quad2 - focus2 = {round(focus_values[idx3], 4)}")


# slider1.on_changed(update1)
# slider2.on_changed(update2)
# slider3.on_changed(update3)

# plt.show()
