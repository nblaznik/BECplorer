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
import datetime 

# Pretty Matplotlib Text
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


num_frames = 51
focus_values = np.linspace(-0.04, 0.04, num_frames)
# Done shots 63, 84

date = 20250305
for shot in [98, 111, 112]:
# shot = 96
    current_time = datetime.datetime.now()   
    timedelta = datetime.timedelta(seconds=int(num_frames*15))

    endtime = (current_time + timedelta).time().strftime('%H:%M:%S')

    print()
    if os.path.exists(f"/home/bec_lab/Desktop/imgs/focus_shift_slider/{date}_{shot}_ang1.npy"):
        print(f"Processing for {date} - {shot} has already been done before.")

    else:
        print(f"Processing for {date} - {shot} NOT done before.")
        print(f"Doing it now. Expect it to last about {round(num_frames*15/60, 1)} min")
        print(f"Should be finished by {endtime}")
        ang1 = [HI_refocus(date, shot, 0, dz_focus=foc, quad="quad1", output="amp") for foc in focus_values]
        ang2 = [HI_refocus(date, shot, 0, dz_focus=foc, quad="quad2", output="amp") for foc in focus_values]
        np.save(f"/home/bec_lab/Desktop/imgs/focus_shift_slider/{date}_{shot}_ang1.npy", ang1)
        np.save(f"/home/bec_lab/Desktop/imgs/focus_shift_slider/{date}_{shot}_ang2.npy", ang2)

