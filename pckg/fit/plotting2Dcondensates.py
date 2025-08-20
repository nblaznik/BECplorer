from pylab import figure, cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
from mpmath import polylog, dirichlet


kB = np.float64(1.38064852E-23)
m = np.float64(3.81923979E-26)
hb = np.float64(1.0545718E-34)
h = np.float64(6.62607015E-34)
asc = np.float64(2.802642E-9)
mu0 = np.float64(1E-50)
e0 = np.float64(8.854187E-12)
pix_size = np.float64(6.5E-6 / 2.63)


bin_f = 8
size = (2160 // bin_f, 2560 // bin_f)


def n_TH(T, mu, tof, w_r, w_z, size=size):
    """ Calculate the column densities. """
    # Create an empty output array
    output = np.zeros(size)
    l_db = ((2 * np.pi * hb ** 2) / (m * kB * T)) ** (1 / 2)


    factor = 1 / (l_db **2 ) * ((1 / (1 + w_r ** 2 * tof ** 2)) ** (1 / 2) * (1 / (1 + w_z ** 2 * tof ** 2)) ** (1 / 2)) * (kB * T / (hb * w_r))

    for i in range(size[0]):
        for j in range(size[1]):
            x_c = (i * bin_f - size[0]*bin_f/2) * pix_size
            z_c = (j * bin_f - size[1]*bin_f/2) * pix_size
            Z_p = np.exp(1 / (kB * T) * (mu - (m * w_r ** 2 * x_c**2) / (2 + 2 * w_r ** 2 * tof ** 2) - (m * w_z ** 2 * z_c**2) / (2 + 2 * w_z ** 2 * tof ** 2)))
            # pl = polylog(2, Z_p)
            # output[i][j] = pl * factor
            output[i][j] = Z_p * factor
    n_cl = output
    return n_cl


def n_CD(T, mu, tof, w_r, w_z, size=size):
    output = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            x_c = (i * bin_f - size[0]*bin_f/2) * pix_size
            z_c = (j * bin_f - size[1]*bin_f/2) * pix_size
            R_r = ( 2 * mu / (m * w_r**2 ) )**(1/2)
            R_z = ( 2 * mu / (m * w_z**2 ) )**(1/2)
            lam_r = ( 1 + w_r**2 * tof**2 )**(1/2)
            lam_z = 1 + (w_z / w_r)**2 * (w_r * tof * np.arctan(w_r * tof) - np.log((1 + w_r**2 * tof**2)**(1/2)))
            U0 = 4 * np.pi * hb**2 * asc / m
            n_cd = max(4 * mu * R_r / (3 * U0 * lam_r**2 * lam_z) * ( 1 - ( x_c / (lam_r * R_r) )**2 - ( z_c / ( lam_z * R_z) )**2 ), 0)
            output[i][j] = n_cd**(3/2)
    return output


def n_CD2(T, mu, tof, w_r, w_z, size=size):
    output = np.zeros(size)
    for x in range(size[0]):
        for z in range(size[1]):
            R_c = 1
            Z_c = 1
            n_c0 = 1
            n_cd = n_c0 * max(1 - x**2 / R_c**2 - z**2 / Z_c**2, 0)
            output[x][z] = n_cd
    return output

def absorption(n_c):
    sc_cross = (3 * 589e-9 ** 2) * (5. / 18.) / ( 2 * np.pi )
    I_rel = np.exp(-n_c * sc_cross)
    return I_rel

def numCD(mu, w_r, w_z):
    U0 = 4 * np.pi * hb**2 * asc / m
    N_CD = 8 * np.pi / 15 * ((2 * mu) / (m * (w_r**2 * w_z)**(2/3)))**(3/2)*(mu/U0)
    return N_CD

def numTH(mu_th, T, w_r, w_z):
    N_TH = ((kB * T) / (hb * (w_r**2 * w_z)**(1/3)))**3 * 1.202 # abs(polylog(3, np.exp(mu_th/(kB*T))))
    return N_TH


def critTemp(N, w_r, w_z):
    Tc = hb * (w_r**2 * w_z)**(1/3) / kB * ( N / polylog(3, 1)) ** 1/3
    return Tc



# PARAMS
w_r = 115 * 2*np.pi
w_z = 5 * 2*np.pi
T = 5.88e-6
mu = 1.1849602992239313e-30
if mu > 0:
    mu_th = 0
else:
    mu_th = mu
tof = 0.015
mu_hz = mu / h

# Generate data
nTH_absorption = absorption(0.0001*n_TH(T=T, mu=mu_th, tof=tof, w_r=w_r, w_z=w_z))
nCD_absorption = absorption(n_CD(T=T, mu=mu, tof=tof, w_r=w_r, w_z=w_z))

# nTH_absorption = n_TH(T=T, mu=mu_th, tof=tof, w_r=w_r, w_z=w_z)
# nCD_absorption = n_CD(T=T, mu=mu, tof=tof, w_r=w_r, w_z=w_z)

# Plotting params
vmin, vmax = 0, 1
cmap = cm.afmhot_r

fig, ax = plt.subplots(1, 2, figsize=(13, 8))
fig.suptitle("TOF = {:} ms     $\mu$ = {:} kHz     T = {:} $\mu$K \n$\\omega_\\rho = {:} \\times 2 \pi$     $\omega_z = {:} \\times 2 \pi$\n\n".format(tof*1000, round(mu_hz/1000, 2), T*1e6, round(w_r/2/np.pi), round(w_z/2/np.pi)))
im = ax[0].imshow(nTH_absorption, origin='upper', cmap=cmap)#, vmin=vmin, vmax=vmax)
ax[1].imshow(nCD_absorption, origin='upper', cmap=cmap)#, vmin=vmin, vmax=vmax)
ax[0].set_title("{:} M Th. particles".format(round(numTH(mu_th=mu_th, T=T, w_r=w_r, w_z=w_z)/1e6, 2)))
ax[1].set_title("{:} M Cd. particles".format(round(numCD(mu=mu, w_r=w_r, w_z=w_z)/1e6, 2)))
# ax2 = ax[0].twinx()
# ax2.set_ylim(-3, 5)
# ax2.plot(nTH_absorption[size[0]//2])
# ax3 = ax[1].twinx()
# ax3.set_ylim(-3, 5)
# ax3.plot(nCD_absorption[size[0]//2])

plt.colorbar(im, ax=(ax[0], ax[1]), orientation='horizontal', aspect=60)
plt.show()


quit()

# fig2, ax2 = plt.subplots(1, 1)
# ax2.imshow(absorption(n_TH(T=T, mu=mu_th, tof=tof, w_r=w_r, w_z=w_z) + n_CD(T=T, mu=mu, tof=tof, w_r=w_r, w_z=w_z)))
# plt.show()







# # Is this in meters? I assume so, right, all other units are fairly standard.
# xmin = -1280 * 2.47148288973384e-06
# xmax= 1280 * 2.47148288973384e-06
# xstep = (xmax - xmin)/256
#
# zmin = -1080 * 2.47148288973384e-06
# zmax = 1080 * 2.47148288973384e-06
# zstep = (zmax - zmin)/216

# x1, x2 = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(zmin, zmax, zstep))






def update(val):
    mu = mu_val.val*1e-32
    tof = tof_val.val*1e-3
    T = T_val.val*1e-6
    y = absorption(n_TH(T=T, mu=0, tof=tof, w_r=w_r, w_z=w_z)
                   + n_CD(T=T, mu=mu, tof=tof, w_r=w_r, w_z=w_z))
    im1.set_data(y)
    # l1.set_xdata(y[:, y.shape[1]//2])
    # l2.set_ydata(y[y.shape[0]//2])
    print("Particle number CONDENSED: {:} M".format(round(numCD(mu, T, w_r, w_z)), 3))
    print("Particle number THERMAL: {:} M".format(round(numTH(0, T, w_r, w_z)/1e6), 3))
    fig.canvas.draw()

y = absorption(n_TH(T=0.5e-6, mu=0, tof=0.015, w_r=w_r, w_z=w_z) + n_CD(T=0.5e-6, mu=5e-32, tof=0.010, w_r=w_r, w_z=w_z))

fig = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(11, 7, left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.25, hspace=0.25)
ax = fig.add_subplot(gs[0:6, 0:6])
ax2 = fig.add_subplot(gs[0:6, 6])
ax3 = fig.add_subplot(gs[6, 0:6])
Tax = fig.add_subplot(gs[8, 0:7])
tofax = fig.add_subplot(gs[9, 0:7])
muax = fig.add_subplot(gs[10, 0:7])



im1 = ax.imshow(y, origin='upper', cmap=cm.afmhot_r)
# l1, = ax2.plot(y[:, y.shape[1]//2], np.arange(zmin, zmax, zstep))
# l2, = ax3.plot(y[y.shape[0]//2])
ax2.invert_xaxis()
ax2.set_xlim(-0.1, 1.1)
ax3.set_ylim(-0.1, 1.1)
ax.tick_params(bottom=False, top=True, left=True, right=False)
ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False)
ax2.set_yticks([])
ax3.set_xticks([])
T_val = Slider(Tax, 'T [$\mu K$]', 0, 5,  valinit=.5)
tof_val = Slider(tofax, 'tof [ms]', 0, 50, valinit=15)
mu_val = Slider(muax, '$\mu$ [J or smth]', 0, 100, valinit=5)
T_val.on_changed(update)
tof_val.on_changed(update)
mu_val.on_changed(update)
plt.colorbar(im1, ax=[ax, ax2, ax3, Tax, tofax, muax], pad=0.1)
plt.show()
