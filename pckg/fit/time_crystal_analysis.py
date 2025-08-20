import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys


parser = argparse.ArgumentParser()
parser.add_argument('date', help='Date at which the shot to analyze was made.', metavar='YYYYMMDD')
parser.add_argument('shot', help='Number of the shot in question.', metavar='XXXX')
args, unknown = parser.parse_known_args(sys.argv[1:])


date = args.date
shot = args.shot
input_folder = '/storage/data/' + date + '/' + shot + '/'



print(" This will analyse the time crystal data for date {:}, shot {:}.".format(date, shot))


# Let's see:
# 1. OAH Processing, if it has not yet been done.

# 2. OAH cooling thing, where we fit each image with a



plt.figure()
plt.savefig(input_folder + "TimeCrystal.png")
