import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from coolingOAH import *

# Generate runs you want to investigate
runs = []
for i in range(50, 68, 1):
        runs.append([20220523, i])

alltempsx = []
alltempsz = []
allpnums = []
time_l = []

for date, run in runs:
        path = '/storage/data/' + str(date) + '/'
        image = str(run).zfill(4) + '/'
        full_path = path + image
        cooling_path = full_path + "OAHcooling/"

        alltempsx.append(np.load(cooling_path + "temps_x.npy"))
        alltempsz.append(np.load(cooling_path + "temps_z.npy"))
        allpnums.append(np.load(cooling_path + "pnums.npy"))
        time_l.append(np.load(cooling_path + "time_list.npy"))


fig2 = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
gs.update(wspace=0.25, hspace=0.)  # set the spacing between axes.

ax1 = fig2.add_subplot(gs[0, 0])
ax2 = fig2.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig2.add_subplot(gs[:, 1])
plt.setp(ax1.get_xticklabels(), visible=False)

i = 0
for date, run in runs:
        ax1.plot(time_l[i], allpnums[i], label="{:}-{:}".format(date, run))
        ax2.plot(time_l[i], alltempsx[i], label="{:}-{:}-$T_x$".format(date, run))
        # ax2.plot(time_l[i], alltempsz[i], label="{:}-{:}-$T_z$".format(date, run))
        ax3.scatter(np.log(alltempsx[i]), np.log(allpnums[i]), label="{:}-{:}-$T_x$".format(date, run))
        # ax3.scatter(np.log(alltempsz[i]), np.log(allpnums[i]), label="{:}-{:}-$T_z$".format(date, run))
        ax3.grid()
        ax3.legend()

        ax1.set_ylabel("Particle Number")
        ax2.set_ylabel("Temparature [K]")
        ax2.set_xlabel("Time [ms]")
        ax3.set_xlabel("Log (T)")
        ax3.set_ylabel("Log (N)")

        ax1.ticklabel_format(axis='both', style='', scilimits=(0, 0))
        ax2.ticklabel_format(axis='both', style='', scilimits=(0, 0))
        ax3.set_title("{:} -- {:}".format(date, run))
        i+=1

plt.show()