import os
import sys
import csv
import time 
import argparse
import datetime
import numpy as np
import textwrap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
matplotlib.use('TkAgg')  # Switch to non-interactive backend to avoid focus stealing

# CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-D', help='date', metavar='YYYYMMDD')
parser.add_argument('-R', help='run', metavar='XXXX')
args, unknown = parser.parse_known_args(sys.argv[1:])
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

### DATA FUNCTIONS
def latestRunToday(datestring):
    # Save the largest runID, to be used for comparison for the live update
    # Get today's date, formatted
    # Rather get the latest run, with gaussian number calculated.
    path = "/storage/data/" + datestring + "/"
    if os.path.exists(path):
        dirList = [x for x in os.listdir(path) if os.path.isdir(path + x)]
        if len(dirList) > 0:
            output = sorted(dirList, reverse=True)
            for run in output:
                if os.path.exists(path + "/" + run + "/fit_gauss.param"):
                    return datestring, run
                elif os.path.exists(path + "/" + run + "/fit_bimodal.param"):
                    return datestring, run
        else:
            output = None
    else:
        output = None
    return datestring, output

def isCondensed(date, seq):
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    with open(path + date + '/' + run_id + '/fit_bimodal.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        next(csvreader)
        next(csvreader)
        # Check if condensation
        for row in csvreader:
            if row[0] == "ntherm":
                nth = float(row[1])
            elif row[0] == "ntf":
                ntf = float(row[1])
        if (ntf + nth) == 0:
            return False
        if ntf / (ntf + nth) > 0.1:  # if more than 10%
            return True
        else:
            return False
        
def get_fit_parameter(date, seq, param):
    """
    Get the value of the fitted parameters, such as the number of atoms or temperature
    from the fit_mode.param file. Get all of them, and append the values in an array.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    condensed = False
    if os.path.exists(path + date + '/' + run_id + '/fit_bimodal.param'):
        with open(path + date + '/' + run_id + '/fit_bimodal.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            next(csvreader)
            next(csvreader)
            for row in csvreader:
                if row[0] == param:
                    var = float(row[1])
            condensed = isCondensed(date, seq)

    elif os.path.exists(path + date + '/' + run_id + '/fit_gauss.param'):
        with open(path + date + '/' + run_id + '/fit_gauss.param') as paramfile:
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

    else: 
        var = 0
    return var, condensed

def get_efficiency(date, seq):
    """
    Get the value of the fitted parameters, such as the number of atoms or temperature
    from the fit_mode.param file. Get all of them, and append the values in an array.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    if os.path.exists(path + date + '/' + run_id + '/fit_gauss.param'):
        mode = '/fit_gauss.param'
    elif os.path.exists(path + date + '/' + run_id + '/fit_bimodal.param'):
        mode = '/fit_bimodal.param'
    elif os.path.exists(path + date + '/' + run_id + '/fit_tf.param'):
        mode = '/fit_tf.param'

    with open(path + date + '/' + run_id + mode) as paramfile:
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

def getIgnore(datestring):
    """Get ignore files."""
    ignore = []
    if os.path.exists("/storage/BECViewer/ignore/{:}".format(datestring)):
        with open("/storage/BECViewer/ignore/{:}".format(datestring), "r") as f:
            for line in f:
                ignore.append(int(line.strip()))
        ignore = sorted(list(dict.fromkeys(ignore)))  # eliminate duplicates
    return ignore

def get_comment(date, run):
    """ Fetch the comment, if there is one """
    run = str(run).zfill(4)
    if os.path.exists("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run)):
        with open("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run), "r") as commentFile:
            comment = commentFile.read()
    else:
        comment = ""
    return comment

def which_comment(date):
    """ Get which shots have a comment. """
    shot_numbers = []
    for filename in os.listdir('/storage/BECViewer/comments/'):
        if filename.endswith('.txt'):
            file_date, shot = filename.replace('.txt', '').split('_')
            if file_date == date:
                shot_numbers.append(int(shot))

    return sorted(shot_numbers)
 
def updateData(init, last, datestring, param, eff=False, conds_check=False):
    ignore = getIgnore(datestring)
    xs = []
    ys = []
    xs_condensed = []
    ys_condensed = []

    for i in range(init, int(last) + 1):
        if i not in ignore:
            try:
                if eff:
                    ys.append(get_efficiency(datestring, i))
                    xs.append(i)
                else:
                    ys.append(get_fit_parameter(datestring, i, param)[0])
                    xs.append(i)
                    if conds_check:
                        if get_fit_parameter(datestring, i, param)[1]:
                            ys_condensed.append(get_fit_parameter(datestring, i, param)[0])
                            xs_condensed.append(i)
            except:
                pass
    if conds_check:
        return xs, ys, xs_condensed, ys_condensed
    else:
        return xs, ys
    

### BUTTON FUNCTIONS
def update_layout():
    """ Function to update layout dynamically and ensure x-ticks/labels on the lowest visible plot."""
    visible_axes = [axs[i] for i in range(3) if visibility[i]]
    n = len(visible_axes)

    # Rearrange subplots
    for idx, ax in enumerate(reversed(visible_axes)):
        ax.set_position([0.05, 0.2 + (idx * 0.75 / n), 0.85, 0.66 / n])

    # Ensure x-ticks and labels are shown only on the lowest visible plot
    for ax in axs:
        plt.setp(ax.get_xticklabels(), visible=False)  # Hide all x-tick labels
        ax.xaxis.set_tick_params(which='both', labelbottom=False)  # Hide x-ticks
        
    if visible_axes:
        # Show x-ticks and labels on the lowest visible plot
        lowest_visible_ax = visible_axes[-1]
        plt.setp(lowest_visible_ax.get_xticklabels(), visible=True)  # Show x-tick labels
        # lowest_visible_ax.set_xlabel("RUN")  # Only label the x-axis on the bottom plot
        lowest_visible_ax.xaxis.set_tick_params(which='both', labelbottom=True)  # Show x-ticks

    plt.draw()

def toggle_subplot1(event):
    visibility[0] = not visibility[0]
    axs[0].set_visible(visibility[0])
    update_layout()

def toggle_subplot2(event):
    visibility[1] = not visibility[1]
    axs[1].set_visible(visibility[1])
    update_layout()

def toggle_subplot3(event):
    visibility[2] = not visibility[2]
    axs[2].set_visible(visibility[2])
    update_layout()

def ntherm_pnum(event):
    global pnum_var
    global xs, ys
    pnum_var= 'ntherm'
    xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
    annotated_x_values = which_comment(datestring) # comments 
    ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values]
    line_annot.set_offsets(np.c_[annotated_x_values, ys_comms])
    line1.set_xdata(xs)
    line1.set_ydata(ys)
    scatter1.set_offsets(np.c_[xs, ys])
    scatter1_c.set_offsets(np.c_[xs_condensed, ys_condensed])
    axs[0].relim(visible_only=True)
    axs[0].autoscale_view()
    text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
    text.set_position((1.01, ys[-1]))
    plt.draw()
    update_layout()

def ntf_pnum(event):
    global pnum_var
    global xs, ys
    pnum_var= 'ntf'
    xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
    annotated_x_values = which_comment(datestring) # comments 
    ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values]
    line_annot.set_offsets(np.c_[annotated_x_values, ys_comms])
    line1.set_xdata(xs)
    line1.set_ydata(ys)
    scatter1.set_offsets(np.c_[xs, ys])
    scatter1_c.set_offsets(np.c_[xs_condensed, ys_condensed])
    axs[0].relim(visible_only=True)
    axs[0].autoscale_view()
    text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
    text.set_position((1.01, ys[-1]))
    plt.draw()
    update_layout()

def ncount_pnum(event):
    global pnum_var
    global xs, ys
    pnum_var= 'ncount'
    xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
    annotated_x_values = which_comment(datestring) # comments 
    ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values]
    line_annot.set_offsets(np.c_[annotated_x_values, ys_comms])
    line1.set_xdata(xs)
    line1.set_ydata(ys)
    scatter1.set_offsets(np.c_[xs, ys])
    scatter1_c.set_offsets(np.c_[xs_condensed, ys_condensed])
    axs[0].relim(visible_only=True)
    axs[0].autoscale_view()
    text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
    text.set_position((1.01, ys[-1]))
    plt.draw()
    update_layout()

def ntotal_pnum(event):
    global pnum_var
    global xs, ys
    pnum_var= 'ntotal'
    xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
    annotated_x_values = which_comment(datestring) # comments 
    ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values]
    line_annot.set_offsets(np.c_[annotated_x_values, ys_comms])
    line1.set_xdata(xs)
    line1.set_ydata(ys)
    scatter1.set_offsets(np.c_[xs, ys])
    scatter1_c.set_offsets(np.c_[xs_condensed, ys_condensed])
    axs[0].relim(visible_only=True)
    axs[0].autoscale_view()
    text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
    text.set_position((1.01, ys[-1]))
    plt.draw()
    update_layout()

def tx_only(event):
    global tx_visible
    tx_visible = not tx_visible
    line2x.set_visible(tx_visible) 
    scatter2x.set_visible(tx_visible)
    axs[1].relim(visible_only=True)
    axs[1].autoscale_view()
    plt.draw()
    # update_layout()

def tz_only(event):
    global tz_visible
    tz_visible = not tz_visible
    line2z.set_visible(tz_visible)  
    scatter2z.set_visible(tz_visible)
    axs[1].relim(visible_only=True)
    axs[1].autoscale_view()
    plt.draw()
    # update_layout()
    
def style_button(button, color='#2c3e50', hovercolor='#34495e', text_color='white', fontsize=8, weight='bold'):
    """ Button Style """
    button.label.set_fontsize(fontsize)
    button.label.set_fontweight(weight)
    button.label.set_color(text_color)
    button.color = color
    button.hovercolor = hovercolor

def update_init(val):
    global init
    global xs
    global ys # need for clicking to work.
    init = int(slider.val)  # Update the global init variable with the slider's value
    # Recompute the data with the updated 'init' value
    xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
    xs_tx, ys_tx = updateData(init, last, datestring, "tx")
    xs_tz, ys_tz = updateData(init, last, datestring, "tz")
    xs_eff, ys_eff = updateData(init, last, datestring, "--", eff=True)
    annotated_x_values = [k for k in which_comment(datestring) if k > init]
    ignore = getIgnore(datestring)
    ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values if i not in ignore]

    # Update for lineplots
    line1.set_xdata(xs)
    line1.set_ydata(ys)
    line2x.set_xdata(xs_tx)
    line2x.set_ydata(ys_tx)
    line2z.set_xdata(xs_tz)
    line2z.set_ydata(ys_tz)
    line3.set_xdata(xs_eff)
    line3.set_ydata(ys_eff)
    
    # Update scatter plots with new points
    scatter1.set_offsets(np.c_[xs, ys])
    scatter1_c.set_offsets(np.c_[xs_condensed, ys_condensed])
    line_annot.set_offsets(np.c_[annotated_x_values, ys_comms])
    scatter2x.set_offsets(np.c_[xs_tx, ys_tx])
    scatter2z.set_offsets(np.c_[xs_tz, ys_tz])
    scatter3.set_offsets(np.c_[xs_eff, ys_eff])

    # Fix the scales and text
    axs[0].relim()
    axs[0].autoscale_view()
    axs[1].relim()
    axs[1].autoscale_view()
    axs[2].relim()
    axs[2].autoscale_view()

    # Update the particle number text
    text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
    text.set_position((1.01, ys[-1]))
    textx.set_text(str(round((ys_tx[-1]) * 1e6, 2)) + "\u03BCK")
    textz.set_text(str(round((ys_tz[-1]) * 1e6, 2)) + "\u03BCK")
    plt.draw()

    # Update the comments list
    annotated_x_values = which_comment(datestring)


### CLICKING-COMMENT FUNCTIONS
def create_text_box(ax, text, x, y):
    """
    Creates a styled text box above the specified data point with an arrow pointing to it.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to place the text box.
    text (str): The text content of the box.
    x (float): x-coordinate for the text box (data point).
    y (float): y-coordinate for the text box (data point).
    """
    text = "\n".join(textwrap.wrap(text, width=20))

    # Box properties
    bbox_props = {
        "boxstyle": "round,pad=0.5",  # Rounded edges with padding
        "facecolor": "#f0f0f0",       # Light grey background
        "edgecolor": "#333333",       # Dark grey border
        "linewidth": 1.5,             # Border line width
        "alpha": 0.9,                 # Slightly transparent
    }
    
    # Text properties
    text_props = {
        "fontsize": 8,                # Font size
        "fontweight": "bold",         # Bold text
        "color": "#333333",           # Dark grey text color
        "ha": "center",               # Centered horizontally
        "va": "center"                # Centered vertically
    }
    
    # Arrow properties
    arrow_props = {
        "arrowstyle": "->",          # Simple arrow
        "color": "#333333",          # Dark grey color for the arrow
        "lw": 1.5                    # Arrow line width
    }
    
    # Create the annotation with an arrow
    annot = ax.annotate(
        text,                        # Text to display
        xy=(x, y),                   # Point to annotate (data point)
        xycoords='data',             # Coordinates of the data point
        xytext=(70, 70),             # Offset the text box (e.g., 40 points above the data point)
        textcoords="offset pixels",  # Interpret the offset as points
        bbox=bbox_props,             # Add the styled bounding box
        arrowprops=arrow_props,      # Add an arrow pointing to the data point
        **text_props                 # Additional text properties
    )
    
    return annot

def add_annotation(ax, x_val, y_val):
    """Create and add an annotation to the plot at the specified location."""
    text = f"{get_comment(datestring, x_val)}" 
    annot = create_text_box(ax, text, x_val, y_val)
    return annot

def add_vertical_line(x_val):
    """Draw a faint dotted vertical line across all subplots at the given x-value."""
    lines = []
    for ax in axs:
        line = ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)
        lines.append(line)
    return lines

def on_click(event):
    if event.inaxes == ax_slider:
        return
    for (scatter, ax, y_data) in [(scatter1, axs[0], ys)]:
        if event.inaxes == ax:
            contains, ind = scatter.contains(event)
            if contains:
                index = ind["ind"][0]
                x_val, y_val = xs[index], y_data[index]
                # Check if the x_val is in the allowed list
                if x_val in annotated_x_values:
                    if (ax, x_val) in annotations:
                        # Remove annotation and vertical line if they exist
                        annotations[(ax, x_val)].remove()
                        del annotations[(ax, x_val)]
                        for line in v_lines[(ax, x_val)]:
                            line.remove()
                        del v_lines[(ax, x_val)]
                    else:
                        # Remove any existing annotation and line on this plot
                        for key in list(annotations.keys()):
                            if key[0] == ax:
                                annotations[key].remove()
                                del annotations[key]
                                for line in v_lines[key]:
                                    line.remove()
                                del v_lines[key]

                        # Add new annotation and vertical line
                        annot = add_annotation(ax, x_val, y_val)
                        annotations[(ax, x_val)] = annot
                        v_lines[(ax, x_val)] = add_vertical_line(x_val)

                    fig.canvas.draw_idle()


############################### Initialize data ###############################
# Generate data
pnum_var = "ntherm" # start with ntherm, but can be changed to ntotal or ncount
last = latestRunToday(datestring)[1]
xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
xs_tx, ys_tx = updateData(init, last, datestring, "tx")
xs_tz, ys_tz = updateData(init, last, datestring, "tz")
xs_eff, ys_eff = updateData(init, last, datestring, "-", eff=True)
annotated_x_values = which_comment(datestring) # comments 
ignore = getIgnore(datestring)
ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values if i not in ignore]


############################### Plotting ###############################
# Set up figure and subplots with a shared x-axis
fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=True)
fig.subplots_adjust(bottom=0.2)

# lineplots 
line1, = axs[0].plot(xs, ys, label='# of particles')
line2x, = axs[1].plot(xs_tx, ys_tx, label='$T_x$', color='sandybrown')
line2z, = axs[1].plot(xs_tz, ys_tz, label='$T_z$', color='forestgreen')
line3, = axs[2].plot(xs_eff, ys_eff, label='eff', color='purple')

# scatterplots
line_annot = axs[0].scatter(annotated_x_values, ys_comms, s=80, facecolor='royalblue', edgecolor='midnightblue', lw=2, label='comment')
scatter1 = axs[0].scatter(xs, ys, color='royalblue', alpha=0.7)
scatter1_c = axs[0].scatter(xs_condensed, ys_condensed,  alpha=0.7, color='lightcoral', label='condensed')
scatter2x = axs[1].scatter(xs_tx, ys_tx, color='sandybrown', alpha=0.7)
scatter2z = axs[1].scatter(xs_tz, ys_tz, color='forestgreen', alpha=0.7)
scatter3 = axs[2].scatter(xs_eff, ys_eff, color='purple', alpha=0.7)

# text
transN = transforms.blended_transform_factory(axs[0].transAxes, axs[0].transData)
transT = transforms.blended_transform_factory(axs[1].transAxes, axs[1].transAxes)
text =  axs[0].text(1.01, ys[-1], str(round(int(ys[-1]) / 1e6, 1)) + "M", fontsize=15, weight='bold', c='indianred', transform=transN)
textx = axs[1].text(1.01, 0.2, str(round((ys_tx[-1]) * 1e6, 2)) + "\u03BCK", fontsize=15, weight='bold', c='sandybrown', transform=transT)
textz = axs[1].text(1.01, 0.1, str(round((ys_tz[-1]) * 1e6, 2)) + "\u03BCK", fontsize=15, weight='bold', c='forestgreen', transform=transT)

# labels
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.setp(axs[0].get_xticklabels(), visible=True)
plt.setp(axs[1].get_xticklabels(), visible=True)
axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Set initial visibility states
visibility = [True, False, False]
tx_visible = True
tz_visible = True
for i in range(3):
    axs[i].set_visible(visibility[i])

############################### Buttons and Sliders ###############################
# particle number toggle
ax_button_ntherm = plt.axes([0.05, 0.02, 0.065, 0.02])
ax_button_ncount = plt.axes([0.125, 0.02, 0.065, 0.02])
ax_button_ntotal = plt.axes([0.20, 0.02, 0.065, 0.02])
ax_button_ntf = plt.axes([0.275, 0.02, 0.065, 0.02])
button2_ntherm = Button(ax_button_ntherm, 'ntherm')
button_ncount = Button(ax_button_ncount, 'ncount')
button_ntotal = Button(ax_button_ntotal, 'ntotal')
button_ntf = Button(ax_button_ntf, 'ntf')
style_button(button2_ntherm)
style_button(button_ncount)
style_button(button_ntotal)
style_button(button_ntf)
button2_ntherm.on_clicked(ntherm_pnum)
button_ncount.on_clicked(ncount_pnum)
button_ntotal.on_clicked(ntotal_pnum)
button_ntf.on_clicked(ntf_pnum)

# temperature toggles
ax_button_tx = plt.axes([0.35, 0.02, 0.14, 0.02])
ax_button_tz = plt.axes([0.50, 0.02, 0.14, 0.02])
button_tx = Button(ax_button_tx, r'$T_x$')
button_tz = Button(ax_button_tz, r'$T_z$')
style_button(button_tx)
style_button(button_tz)
button_tx.on_clicked(tx_only)
button_tz.on_clicked(tz_only)

# subplots toggles
ax_button1 = plt.axes([0.05, 0.05, 0.29, 0.035])
ax_button2 = plt.axes([0.35, 0.05, 0.29, 0.035])
ax_button3 = plt.axes([0.65, 0.05, 0.29, 0.035])
button1 = Button(ax_button1, 'Toggle Particle Number')
button2 = Button(ax_button2, 'Toggle Temperature')
button3 = Button(ax_button3, 'Toggle Efficiency')
style_button(button1)
style_button(button2)
style_button(button3)
button1.on_clicked(toggle_subplot1)
button2.on_clicked(toggle_subplot2)
button3.on_clicked(toggle_subplot3)

# a slider 
ax_slider = plt.axes([0.65, 0.02, 0.29, 0.02], facecolor='#2c3e50')
slider = Slider(ax_slider, '', valmin=0, valmax=int(last), valinit=int(init), valstep=1, color='#2c3e50')
slider.on_changed(update_init)

# Connect the click event
fig.canvas.mpl_connect("button_press_event", on_click)

# Initial layout adjustment
update_layout()
annotations = {}    # Store annotations
v_lines = {}        # Store vertical lines

############################### Real-time data update loop ###############################
plt.ion()  # Turn on interactive mode
time.sleep(0.5)  # Slight delay to avoid immediate focus grab
print("\nRunning the live plot now... \nPress Ctrl+C to interrupt the code, or q to interupt the plot.\n")
print("| LAST DETECTED - {:} |".format(last))

try:
    while True:
        if latestRunToday(datestring)[1] > last:
            last = latestRunToday(datestring)[1]
            print("| NEW FILE DETECTED - {:} |".format(last))

            # Get new data
            xs, ys, xs_condensed, ys_condensed = updateData(init, last, datestring, pnum_var, conds_check=True)
            xs_tx, ys_tx = updateData(init, last, datestring, "tx")
            xs_tz, ys_tz = updateData(init, last, datestring, "tz")
            xs_eff, ys_eff = updateData(init, last, datestring, "--", eff=True)
            annotated_x_values = which_comment(datestring) 
            ys_comms = [get_fit_parameter(datestring, i, pnum_var)[0] for i in annotated_x_values]

            # Update for lineplots
            line1.set_xdata(xs)
            line1.set_ydata(ys)
            line2x.set_xdata(xs_tx)
            line2x.set_ydata(ys_tx)
            line2z.set_xdata(xs_tz)
            line2z.set_ydata(ys_tz)
            line3.set_xdata(xs_eff)
            line3.set_ydata(ys_eff)

            # Update scatter plots with new points
            scatter1.set_offsets(np.c_[xs, ys])
            scatter1_c.set_offsets(np.c_[xs_condensed, ys_condensed])
            line_annot.set_offsets(np.c_[annotated_x_values, ys_comms])
            scatter2x.set_offsets(np.c_[xs_tx, ys_tx])
            scatter2z.set_offsets(np.c_[xs_tz, ys_tz])
            scatter3.set_offsets(np.c_[xs_eff, ys_eff])

            # Fix the scales
            axs[0].relim()
            axs[0].autoscale_view()
            axs[1].relim()
            axs[1].autoscale_view()
            axs[2].relim()
            axs[2].autoscale_view()

            # Text
            text.set_text(str(round(int(ys[-1]) / 1e6, 1)) + "M")
            text.set_position((1.01, ys[-1]))
            textx.set_text(str(round((ys_tx[-1]) * 1e6, 2)) + "\u03BCK")
            textx.set_position((1.01, 0.2))
            textz.set_text(str(round((ys_tz[-1]) * 1e6, 2)) + "\u03BCK")
            textz.set_position((1.01, 0.1))

            # Update the range of the slider 
            slider.valmax = int(last)
            slider.ax.set_xlim(slider.valmin,slider.valmax)

            # Update the figure
            plt.draw()

            # Update the comments list
            annotated_x_values = which_comment(datestring)


        else:
            plt.pause(1)  # Pause for 5 seconds
        
        if not plt.fignum_exists(fig.number):
            print("Plot window closed. Stopping.")
            break

except KeyboardInterrupt:
    print("\nReal-time plotting stopped.")
