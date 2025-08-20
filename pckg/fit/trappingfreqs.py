# A file for plotting and extracting the trapping frequencies.
# To generate the file of all the fit parameters, first run the coolingOAH.py script.
import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import math

date = 20220825
shot = 149

# Get data:
path = '/storage/data/' + str(date) + '/'
image = str(shot).zfill(4) + '/'
full_path = path + image
cooling_path = full_path + "OAHcooling/"
par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'tfamp', 'tfxw', 'tfyw', 'gamp', 'gxw', 'gyw']

odroutput = np.load(cooling_path + 'odrout_betas.npy')
time = np.load(cooling_path + 'time_list.npy')

time = [i/1000 for i in time]
xmid = [i[3] for i in odroutput]
ymid = [i[4] for i in odroutput]

print(len(time), len(xmid), len(ymid))


# Fit data:
def f(B, x):
    return B[0] * np.sin((2*np.pi)*B[1]*(x + B[2])) + B[3]


# Run the ODR Fit procedure.
odrmodel = odr.Model(f)  # Store information for the gaussian fitting model
odrdata_x = odr.Data(time, xmid)
odrdata_z = odr.Data(time, ymid)
odrobj_x = odr.ODR(odrdata_x, odrmodel, beta0=[50, 115,  6, 14])
odrobj_z = odr.ODR(odrdata_z, odrmodel, beta0=[1, 15, 1, 53])
# odrobj_x.set_job(2)  # Ordinary least-sqaures fitting
odrobj_z.set_job(2)  # Ordinary least-sqaures fitting
odrout_x = odrobj_x.run()
odrout_z = odrobj_z.run()

fitresult_x = [f(odrout_x.beta, t) for t in time]
fitresult_z = [f(odrout_z.beta, t) for t in time]

print(len(time), len(fitresult_x), len(fitresult_z))

print("-"*40)
print("AMPLITUDE X = ", odrout_x.beta[0])
print("FREQUENCY X = ", odrout_x.beta[1])
print("HORIZONTAL SHIFT X = ", odrout_x.beta[2])
print("VERTICAL SHIFT X = ", odrout_x.beta[3])
print("-"*40)
print()
print("-"*40)
print("AMPLITUDE Z = ", odrout_z.beta[0])
print("FREQUENCY Z = ", odrout_z.beta[1])
print("HORIZONTAL SHIFT Z = ", odrout_z.beta[2])
print("VERTICAL SHIFT Z = ", odrout_z.beta[3])
print("-"*40)

time_det = np.linspace(min(time), max(time), num=1000)


fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].scatter(time, odroutput[:, 3])
ax[0].plot(time_det, odrout_x.beta[0]*np.sin((2*np.pi)*odrout_x.beta[1]*(time_det + odrout_x.beta[2])) + odrout_x.beta[3], c='r', ls='--')
ax[0].set_title("Frequency = {:} Hz".format(round(odrout_x.beta[1], 2)))
ax[1].scatter(time, odroutput[:, 4])
ax[1].plot(time_det, odrout_z.beta[0]*np.sin((2*np.pi)*odrout_z.beta[1]*(time_det + odrout_z.beta[2])) + odrout_z.beta[3], c='r', ls='--')
ax[1].set_title("Frequency = {:} Hz".format(round(odrout_z.beta[1], 2)))

ax[0].set_ylabel("x-position")
ax[1].set_ylabel("y-position")

plt.tight_layout()

plt.savefig("/home/bec_lab/Desktop/UUOneDrive/trappingfrequencies.png")
plt.show()