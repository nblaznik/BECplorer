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
plt.style.use('bmh')

parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('-M', help='mode.', metavar='YYYYMMDD')
args, unknown = parser.parse_known_args(sys.argv[1:])


if 0:
    plt.style.use('dark_background')



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
        else:
            output = None
    else:
        output = None
    return datestring, output

def get_fit_parameter(date, seq):
    """
    Get the value of the fitted parameters, such as the number of atoms or temperature
    from the fit_mode.param file. Get all of them, and append the values in an array.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    with open(path + date + '/' + run_id + '/fit_gauss.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        next(csvreader)
        next(csvreader)
        for row in csvreader:
            if row[0] == "ntherm":
                ntherm = float(row[1])
            elif row[0] == "gamp_p":
                gamp = float(row[1])
            elif row[0] == "gxw_p":
                gxw = float(row[1])
            elif row[0] == "gyw_p":
                gyw = float(row[1])
            elif row[0] == "tx":
                tx = float(row[1])
            elif row[0] == "tz":
                tz = float(row[1])
    N = 2*np.pi * gamp * gxw * gyw
    V = (2*np.pi)**(3/2) * gxw**2 * gyw
    density = N/V
    gamma = density * np.sqrt(((tx+tz)/2))
    return gamma

def get_variable(date, seq, mode):
    """
    Get the value of the fitted parameters, such as the number of atoms or temperature
    from the fit_mode.param file. Get all of them, and append the values in an array.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    try:
        with open(path + date + '/' + run_id + '/parameters_mod.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                if row[0] == mode:
                    val = float(row[1])
    except:
        with open(path + date + '/' + run_id + '/parameters.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                if row[0] == mode:
                    val = float(row[1])
    return val

def updateData(init, last, datestring, mode):
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

    if mode == "NUM":
        for i in range(init, int(last) + 1):
            if i not in ignore:
                try:
                    ys.append(get_fit_parameter(datestring, i))
                    xs.append(i)
                except:
                    pass
    else:
        for i in range(init, int(last) + 1):
            if i not in ignore:
                try:
                    ys.append(get_fit_parameter(datestring, i))
                    xs.append(get_variable(datestring, i, mode))
                except:
                    pass

    return xs, ys



if args.M:
    date = datetime.datetime.now()
    datestring = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
    init = int(args.M)
    mode = "NUM"

else:
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


    mode = input("Against what variable do you want it plotted? [rftime, rftime1, rftime2, rftime3]. Press enter for vs RUN plot. \n")
    if mode == "":
        mode = "NUM"





# Generate data
last = latestRunToday(datestring)[1]
xs, ys = updateData(init, last, datestring, mode)


# Initial plot
plt.ion()
fig = plt.figure(figsize=(17, 8))
ax = fig.add_subplot(111)
dots1 = ax.scatter(xs[:-1], ys[:-1], color='royalblue')
# line1, = ax.plot(xs[:-1], ys[:-1], c='royalblue', ls='--', linewidth=1, alpha=0.25)
dots2 = ax.scatter(xs[-1], ys[-1], c='indianred')
line2, = ax.plot(xs[-2:], ys[-2:], c='indianred', ls='-', linewidth=1, alpha=0.45)
trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
text = ax.text(1.01, ys[-1], str(round(int(ys[-1]) / 1e13, 1)), fontsize=15, weight='bold', c='indianred', transform=trans)
ax.grid(visible=True, alpha=0.4)
ax.set_title("Efficiency vs {:}.\n".format(mode), fontsize=15, weight='bold')
ax.set_xlabel(mode)
ax.set_ylabel("$\gamma$ [arb.]")


# # Generate uncertainty line
# alpha_linefill = 0.1
# xs_smooth, ys_low, ys_high = generateTrend(xs, ys)
#
#
# uncert = ax.fill_between(xs_smooth, ys_low, y2=ys_high, color='royalblue', alpha=alpha_linefill)


# Calculate and set the lims
edgex = 0.05
edgey = 0.01
xlim = min(xs) - ((max(xs) - min(xs)) * edgex / (1 - 2 * edgex)), max(xs) + ((max(xs) - min(xs)) * edgex / (1 - 2 * edgex))
ylim = min(ys) - ((max(ys)-min(ys)) * edgey/(1-2*edgey)), max(ys) + ((max(ys)-min(ys)) * edgey/(1-2*edgey))
ax.set_xlim(xlim)
ax.set_ylim(ylim)


fig.canvas.draw()
fig.canvas.flush_events()

i = 0
print("\nRunning the live plot now... \nPress Ctrl+C to interrupt the code, or q to interupt the plot.\n")
print("| LAST DETECTED - {:} |".format(last))

try:
    while True:
        if latestRunToday(datestring)[1] > last:
            last = latestRunToday(datestring)[1]
            print("| NEW FILE DETECTED - {:} |".format(last))
            xs, ys = updateData(init, last, datestring, mode)
            dots1.set_offsets(np.c_[xs[:-1], ys[:-1]])
            # line1.set_data(xs[:-1], ys[:-1])
            dots2.set_offsets(np.c_[xs[-1], ys[-1]])
            line2.set_data(xs[-2:], ys[-2:])
            text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
            text.set_position((1.02, ys[-1]))

            # uncert.remove()
            # Generate uncertainty line
            # alpha_linefill = 0.1
            # xs_smooth, ys_low, ys_high = generateTrend(xs, ys)
            # uncert = ax.fill_between(xs_smooth, ys_low, y2=ys_high, color='royalblue', alpha=alpha_linefill)

            # Calculate and set the lims
            edgex = 0.05
            edgey = 0.01
            xlim = min(xs) - ((max(xs)-min(xs)) * edgex/(1-2*edgex)), max(xs) + ((max(xs)-min(xs)) * edgex/(1-2*edgex))
            ylim = min(ys) - ((max(ys)-min(ys)) * edgey/(1-2*edgey)), max(ys) + ((max(ys)-min(ys)) * edgey/(1-2*edgey))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            fig.canvas.draw()
            fig.canvas.flush_events()

        else:
            fig.canvas.flush_events()
            i+=1
        # time.sleep(1)  # to make it easier on resources, but seems to work fine even without it

except KeyboardInterrupt:
    print("\nExiting the live plotting.")
    pass




