import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import numpy as np
import csv
import datetime
import os
import sys
import argparse
from matplotlib.widgets import TextBox
import time
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec

plt.style.use('bmh')

parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-D', help='date', metavar='YYYYMMDD')
parser.add_argument('-R', help='run', metavar='XXXX')
args, unknown = parser.parse_known_args(sys.argv[1:])



def latestRunToday(datestring):
    # Save the largest runID, to be used for comparison for the live update
    # Get today's date, formatted
    # Rather get the latest run, with gaussian number calculated.
    path = "/storage/data/" + datestring + "/"
    if os.path.exists(path):
        dirList = [x for x in os.listdir(path) if os.path.isdir(path + x)]
        if len(dirList) > 0:
            output = sorted(dirList,  reverse=True)
            for run in output:
                if os.path.exists(path + "/" + run + "/fit_gauss.param" ):
                    return datestring, run
                elif os.path.exists(path + "/" + run + "/fit_bimodal.param" ):
                    return datestring, run
        else:
            output = None
    else:
        output = None
    return datestring, output

def get_fit_parameter(date, seq, param):
    """
    Get the value of the fitted parameters, such as the number of atoms or temperature
    from the fit_mode.param file. Get all of them, and append the values in an array.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    if os.path.exists(path + date + '/' + run_id + '/fit_gauss.param'):
        with open(path + date + '/' + run_id + '/fit_gauss.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                if row[0] == param:
                    var = float(row[1])
    elif os.path.exists(path + date + '/' + run_id + '/fit_bimodal.param'):
        with open(path + date + '/' + run_id + '/fit_bimodal.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                if row[0] == param:
                    var = float(row[1])
    elif os.path.exists(path + date + '/' + run_id + '/fit_tf.param'):
        with open(path + date + '/' + run_id + '/fit_tf.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                if row[0] == param:
                    var = float(row[1])
    return var

def updateData(init, last, datestring, param):
    def getIgnore(datestring):
        """Get ignore files."""
        ignore = []
        if os.path.exists("/storage/BECViewer/ignore/{:}".format(datestring)):
            with open("/storage/BECViewer/ignore/{:}".format(datestring), "r") as f:
                for line in f:
                    ignore.append(int(line.strip()))
            ignore = sorted(list(dict.fromkeys(ignore)))  # eliminate duplicates
        return ignore

    ignore = getIgnore(datestring)

    xs = []
    ys = []

    for i in range(init, int(last) + 1):
        if i not in ignore:
            try:
                ys.append(get_fit_parameter(datestring, i, param))
                xs.append(i)
            except:
                pass
    return xs, ys

def generateTrend(xs, ys, last_count=2, n=2):
    """ Generates trend based on the last __ measurements. """
    ys_lower = []
    ys_higher = []
    for i in range(len(ys)):
        if i > last_count:
            # Uncertainty = mean distance between last five ruins
            dists = [abs(ys[i-j] - ys[i-j-1]) for j in range(last_count)]
            unc = 1.5*np.mean(dists)
            ys_lower.append(ys[i] - unc)
            ys_higher.append(ys[i] + unc)
        else: # if not enough data, just do 5 %
            ys_lower.append(ys[i] * 0.85)
            ys_higher.append(ys[i] * 1.15)

    if len(xs[::n]) < 4:
        f_low = interp1d(xs, ys_lower, kind='cubic')
        f_high = interp1d(xs, ys_higher, kind='cubic')
        xs_smooth = np.linspace(min(xs), max(xs), 5000)
        print("Not enough measurements yet. Integrating all.")

    else:
        f_low = interp1d(xs[::n], ys_lower[::n], kind='cubic')
        f_high = interp1d(xs[::n], ys_higher[::n], kind='cubic')
        xs_smooth = np.linspace(min(xs[::n]), max(xs[::n]), 5000)

    return xs_smooth, f_low(xs_smooth), f_high(xs_smooth)


if args.D:
    datestring = str(args.D)
    init = int(args.R)

else:
    # Input - set range, set date
    init_date = input("Enter the date in the form YYYYMMDD or press enter for today's date.\n")
    if init_date == "":
        date = datetime.datetime.now()
        datestring = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
    else:
        datestring = init_date

    init = input("Where do you want to start the range? Press 0 or enter for the entire range.\n")
    if init == "":
        init = 0
    init = int(init)

# Generate data
last = latestRunToday(datestring)[1]
xs, ys = updateData(init, last, datestring, "ncount")
xs_tx, ys_tx = updateData(init, last, datestring, "tx")
xs_tz, ys_tz = updateData(init, last, datestring, "tz")

# Initial plot
plt.ion()
fig = plt.figure(figsize=(17, 12))
gs = gridspec.GridSpec(2, 2)

# Particle Number Plot
ax = fig.add_subplot(gs[0, :])
ax.set_title("Particle number and Temperature through the runs.\n", fontsize=15, weight='bold')
dots1 = ax.scatter(xs[:-1], ys[:-1], color='royalblue')
line1, = ax.plot(xs[:-1], ys[:-1], c='royalblue', ls='--', linewidth=1, alpha=0.25)
dots2 = ax.scatter(xs[-1], ys[-1], c='indianred')
line2, = ax.plot(xs[-2:], ys[-2:], c='indianred', ls='-', linewidth=1, alpha=0.45)
trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
text = ax.text(1.01, ys[-1], str(round(int(ys[-1]) / 1e6, 1)) + "M", fontsize=15, weight='bold', c='indianred', transform=trans)
ax.grid(visible=True, alpha=0.4)
# ax.set_xlabel("RUN")
ax.set_ylabel("Particle Number")

# Temperatures
ax2x = fig.add_subplot(gs[1, :], sharex=ax)

dots1x = ax2x.scatter(xs_tx, ys_tx, color='royalblue', label="$T_x$")
#dots1z = ax2x.scatter(xs_tz, ys_tz, color='forestgreen', label="$T_z$")
line1x, = ax2x.plot(xs_tx[:-1], ys_tx[:-1], c='royalblue', ls='--', linewidth=1, alpha=0.25)
#line1z, = ax2x.plot(xs_tz[:-1], ys_tz[:-1], c='forestgreen', ls='--', linewidth=1, alpha=0.25)
# dots2x = ax2x.scatter(xs_tx[-1], ys_tx[-1], c='indianred')
# dots2z = ax2x.scatter(xs_tz[-1], ys_tz[-1], c='indianred')
line2x, = ax2x.plot(xs_tx[-2:], ys_tx[-2:], c='indianred', ls='-', linewidth=1, alpha=0.45)
#line2z, = ax2x.plot(xs_tz[-2:], ys_tz[-2:], c='indianred', ls='-', linewidth=1, alpha=0.45)
trans = transforms.blended_transform_factory(ax2x.transAxes, ax2x.transAxes)
textx = ax2x.text(1.01, 0.2, str(round((ys_tx[-1]) * 1e6, 2)) + "\u03BCK", fontsize=15, weight='bold', c='royalblue', transform=trans)
#textz = ax2x.text(1.01, 0.1, str(round((ys_tz[-1]) * 1e6, 2)) + "\u03BCK", fontsize=15, weight='bold', c='forestgreen', transform=trans)
ax2x.grid(visible=True, alpha=0.4)
ax2x.legend()
ax2x.set_xlabel("RUN")
ax2x.set_ylabel("Temperature")

plt.setp(ax.get_xticklabels(), visible=False)

# Generate uncertainty line
alpha_linefill = 0.1
xs_smooth, ys_low, ys_high = generateTrend(xs, ys)
uncert = ax.fill_between(xs_smooth, ys_low, y2=ys_high, color='royalblue', alpha=alpha_linefill)


# Calculate and set the lims
edgex = 0.05
edgey = 0.01
xlim = min(xs) - ((max(xs) - min(xs)) * edgex / (1 - 2 * edgex)), max(xs) + ((max(xs) - min(xs)) * edgex / (1 - 2 * edgex))
# ylim = min(ys) - ((max(ys)-min(ys)) * edgey/(1-2*edgey)), max(ys) + ((max(ys)-min(ys)) * edgey/(1-2*edgey))
ylim = max(-0.05*max(ys_high), min(ys_low) - ((max(ys_high) - min(ys_low)) * edgey / (1 - 2 * edgey))), max(ys_high) + ((max(ys_high) - min(ys_low)) * edgey / (1 - 2 * edgey))
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Calculate and set the lims for temps
edgex = 0.05
edgey = 0.01
xlim = min(xs_tx) - ((max(xs_tx) - min(xs_tx)) * edgex / (1 - 2 * edgex)), max(xs_tx) + ((max(xs_tx) - min(xs_tx)) * edgex / (1 - 2 * edgex))
ylim = max(-0.05 * max(ys_tx), min(ys_tx) - ((max(ys_tx) - min(ys_tx)) * edgey / (1 - 2 * edgey))), max(ys_tx) + ((max(ys_tx) - min(ys_tx)) * edgey / (1 - 2 * edgey))
ax2x.set_xlim(xlim)
ax2x.set_ylim(ylim)

fig.canvas.draw()
plt.savefig(f"/home/bec_lab/Desktop/imgs/pnums/latest_update_{datestring}.png")

fig.canvas.flush_events()

i = 0
print("\nRunning the live plot now... \nPress Ctrl+C to interrupt the code, or q to interupt the plot.\n")
print("| LAST DETECTED - {:} |".format(last))

try:
    while True:
        if latestRunToday(datestring)[1] > last:
            last = latestRunToday(datestring)[1]
            print("| NEW FILE DETECTED - {:} |".format(last))
            xs, ys = updateData(init, last, datestring, "ncount")
            xs_tx, ys_tx = updateData(init, last, datestring, "tx")
            xs_tz, ys_tz = updateData(init, last, datestring, "tz")

            dots1.set_offsets(np.c_[xs[:-1], ys[:-1]])
            line1.set_data(xs[:-1], ys[:-1])
            dots2.set_offsets(np.c_[xs[-1], ys[-1]])
            line2.set_data(xs[-2:], ys[-2:])
            text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
            text.set_position((1.02, ys[-1]))

            dots1x.set_offsets(np.c_[xs_tx, ys_tx])
            line1x.set_data(xs_tx[:-1], ys_tx[:-1])
            line2x.set_data(xs_tx[-2:], ys_tx[-2:])
            textx.set_text(str(round((ys_tx[-1]) * 1e6, 2)) + "\u03BCK")
            textx.set_position((1.02, 0.2))

            #dots1z.set_offsets(np.c_[xs_tz, ys_tz])
            #line1z.set_data(xs_tz[:-1], ys_tz[:-1])
            #line2z.set_data(xs_tz[-2:], ys_tz[-2:])
            #textz.set_text(str(round((ys_tz[-1]) * 1e6, 2)) + "\u03BCK")
            #textz.set_position((1.02, 0.1))

            uncert.remove()
            fig.canvas.draw()  # Redraw after removal

            # Generate uncertainty line
            alpha_linefill = 0.1
            xs_smooth, ys_low, ys_high = generateTrend(xs, ys)
            uncert = ax.fill_between(xs_smooth, ys_low, y2=ys_high, color='royalblue', alpha=alpha_linefill)

            # Calculate and set the lims
            edgex = 0.05
            edgey = 0.01
            xlim = min(xs) - ((max(xs)-min(xs)) * edgex/(1-2*edgex)), max(xs) + ((max(xs)-min(xs)) * edgex/(1-2*edgex))
            # ylim = min(ys) - ((max(ys)-min(ys)) * edgey/(1-2*edgey)), max(ys) + ((max(ys)-min(ys)) * edgey/(1-2*edgey))
            ylim = max(-0.05*max(ys_high), min(ys_low) - ((max(ys_high)-min(ys_low)) * edgey/(1-2*edgey))), max(ys_high) + ((max(ys_high)-min(ys_low)) * edgey/(1-2*edgey))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # Calculate and set the lims for temps
            edgex = 0.05
            edgey = 0.01
            xlim = min(xs_tx) - ((max(xs_tx) - min(xs_tx)) * edgex / (1 - 2 * edgex)), max(xs_tx) + ((max(xs_tx) - min(xs_tx)) * edgex / (1 - 2 * edgex))
            ylim = max(-0.05*max(ys_tx), min(ys_tx) - ((max(ys_tx) - min(ys_tx)) * edgey / (1 - 2 * edgey))), max(ys_tx) + ((max(ys_tx) - min(ys_tx)) * edgey / (1 - 2 * edgey))
            ax2x.set_xlim(xlim)
            ax2x.set_ylim(ylim)

            plt.setp(ax.get_xticklabels(), visible=False)
            fig.set_size_inches(17, 12, forward=True)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.savefig(f"/home/bec_lab/Desktop/imgs/pnums/latest_update_{datestring}.png")

        else:
            fig.canvas.flush_events()
            i+=1

            # time.sleep(0.1)  # to make it easier on resources, but seems to work fine even without it
        if not plt.fignum_exists(fig.number):
            print("Plot window closed. Stopping.")
            break

except KeyboardInterrupt:
    print("\nExiting the live plotting.")
    pass




