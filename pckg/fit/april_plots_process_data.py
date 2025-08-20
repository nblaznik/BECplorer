import os
import warnings
warnings.filterwarnings('ignore')
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import own fit functions
import OAH_functions as f1
from OAHDEV_functions import *
from OAH_refocus import *

# Constants:
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
pix_size = 6.5E-6 / 2.63
# Light field properties
lamb0 = 589.1E-9  # Wavelength
k0 = 2 * np.pi / lamb0  # k-vector


# Randomize the background, or even get the mean or something? 
def HI_refocus_custom(date, shot, num, dz_focus, quad="quad1", num_flat=0, shift_pix=[0, 0], output="ang", plot=False, cut=[xmin, xmax, zmin, zmax]):
    """
    Main function for generating the output arrays for the holgoraphic imaging, which also includes the refocusing.
    What we do, is we process image - first take a FTT, apply Tukey windows, shift, refocus, and back iFFT.
    ------
    :param date: Input date of the image to be analysed
    :param shot: Input shot of the image to be analysed
    :param num: Input sequence number of the image to be analysed
    :param dz_focus: The focus parameter that determines the focus of the image
    :param quad: Parameter to determine which quad we are cutting.
    :param output: Output the phase (ang) or amplituide (amp)
    :param plot: Boolean to specify whether to plot the image of the angle or not. Better not when used in a loop.
    :return: Returns an array of the angle of the ratio of the iFFT's of the atom and flat image.
    """
    # ------------------------------------------------- IMPORTS -----------------------------------------------------
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    
    xmin, xmax, zmin, zmax = cut

    if num != 0: 
        num_flat = num_flat % num 
        
    # Opening files
    atoms = pyfits.open(path + image + '0.fits')[0].data.astype(float)[num][xmin:xmax, zmin:zmax]
    flat = pyfits.open(path + image + '1.fits')[0].data.astype(float)[num_flat][xmin:xmax, zmin:zmax]
    dark = pyfits.open(path + image + '2.fits')[0].data.astype(float).mean(axis=0)[xmin:xmax, zmin:zmax]

    # ----------------------------------------------- CORRECTIONS ---------------------------------------------------
    # Creates a squaroid dark edge
    atoms = f1.squaroid(atoms - dark, width=0.51)
    flat = f1.squaroid(flat - dark, width=0.51)
    # --------------------------------------------------- FFT --------------------------------------------------------
    # Take the FTT of the atoms
    fft_atoms = np.fft.fft2(atoms)
    fft_flat = np.fft.fft2(flat)

    # We create the Tukey windows cutouts for the data and the background. Additional cuts in x,z direction can be
    # passed, but this is optional; the default value is 0. Note that because you later cut everything in the
    # same size, this may affect the rest too.
    quad1, q1peak = f1.box_cutter_pad_ellips(fft_atoms, quad, 0, 0, edge_x=10, edge_z=80)
    flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat, quad, 0, 0, edge_x=10, edge_z=80) #indices=q1peak)
    
    print("SHOT = ", shot)
    print("NUM = ", num)
    # Cutting the quads in the same sizes.
    quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

    # ------------------------------------------------ FFT SHIFT ----------------------------------------------------
    # Now we FFT shift the zero frequency to the center
    fft1 = np.fft.fftshift(quad1cut)
    flatfft1 = np.fft.fftshift(flatq1cut)

    
    # ------------------------------------------------ PHASE SHIFT & FFT SHIFT ----------------------------------------------------
    # Shift in real space = linear phase in FFT space (optional)
    shift_x, shift_z = shift_pix

    Nz, Nx = quad1cut.shape  # Shape is (z, x)
    kx = np.fft.fftfreq(Nx)  # Now in cycles per pixel
    kz = np.fft.fftfreq(Nz)
    KX, KZ = np.meshgrid(kx, kz)         # Create full 2D grid

    # Linear phase ramp
    phase_ramp = np.exp(-1j * 2 * np.pi * (KX * shift_x + KZ * shift_z))

#     # Apply ramp before shifting
#     fft1 = np.fft.fftshift(quad1cut * phase_ramp)
#     flatfft1 = np.fft.fftshift(flatq1cut * phase_ramp)

    
    # ------------------------------------------------ REFOCUSING ---------------------------------------------------
    fft_kx = np.fft.fftfreq(fft1.shape[1], d=pix_size)  # Discrete FFT Sample Frequency in x
    fft_ky = np.fft.fftfreq(fft1.shape[0], d=pix_size)  # Discrete FFT Sample Frequency in z
    fft_k2 = fft_kx[None, :] ** 2 + fft_ky[:, None] ** 2  # Discrete FFT Sample Frequency in main axes multiplied

    # Determine the focus factor and refocus
    focus = np.exp(-1j * fft_k2 * dz_focus / (2 * k0))
    fft1 = fft1 * focus * phase_ramp
    flatfft1 = flatfft1 * focus * phase_ramp


    # ------------------------------------- INVERSE FFT -------------------------------------------------
    inv1 = np.fft.ifft2(fft1) / np.fft.ifft2(flatfft1)
    inv1 = inv1[border_x:-border_x, border_z:-border_z]

    # Get Phase
    ang1 = np.angle(inv1)
    ang1 = f1.unwrapper(ang1)
    amp1 = np.abs(inv1) ** 2

    # normalize amplitude?
    normfactor = amp1.mean()  # [300:900, 300:900].mean()
    amp1 = amp1 - normfactor 
    amp1 = normalize(amp1)[0] # Use the function above to normalize the image.

    # Normalize
    normfactor = ang1.mean()  # [300:900, 300:900].mean()
    ang1 = ang1 - normfactor 
    ang1 = normalize(ang1)[0] # Use the function above to normalize the image.

    if plot:
        plt.imshow(ang1, cmap='Greys', interpolation='none', origin="lower")
        plt.title(str(dz_focus))
        plt.colorbar()
        plt.show()

    if output == "amp":
        return amp1
    elif output == "ang": 
        return ang1
    elif output == "quad":
        return quad1
    
def binImage(pic, xbin, zbin):
    """ A function to bin the pic file based on the bin parameters. """
    # If pic file not a multiple of bin, cut from the edge so it is.
    if pic.shape[0] % xbin != 0:
        pic = pic[:-(pic.shape[0] % xbin), :]
    if pic.shape[1] % zbin != 0:
        pic = pic[:, :-(pic.shape[1] % zbin)]
    pic = pic.reshape(pic.shape[0] // xbin, xbin, pic.shape[1] // zbin, zbin).mean(axis=3).mean(axis=1)
    return pic

def getPhysical(all_fits):
    phys_vars = []
    for shot_info in all_fits:
        xbin = 4
        zbin = 4
        pixelsize = 2.47148288973384e-06 
        kB = 1.38064852E-23
        m = 3.81923979E-26
        hb = 1.0545718E-34
        asc = 2.802642E-9
        mu0 = 1e-50
        e0 = 8.854187E-12
        fx = 115
        fz = 15
        wavelength = 589e-9
        detuning = 0
        prefactor = float((1 + 4 * detuning ** 2) * 2 * np.pi / (3 * (wavelength ** 2)) * 18. / 5.)

#         par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'tfamp', 'tfxw', 'tfyw', 'gamp', 'gxw', 'gyw']
#         bin_scaling = np.array([1., 1., 1., xbin, zbin, 1., xbin, zbin, 1., xbin, zbin])
#         rng_offset = np.array([0., 0., 0., xmin, zmin, 0., 0., 0., 0., 0., 0.])
#         to_physical = np.array([1., 1., 1., pixelsize, pixelsize, prefactor, pixelsize, pixelsize, prefactor, pixelsize, pixelsize])
        
        par_names = ['offset', 'ampl', 'ang', 'xmid', 'ymid', 'gamp', 'gxw', 'gyw']
        bin_scaling = np.array([1., 1., 1., xbin, zbin,  1., xbin, zbin])
        rng_offset = np.array([0., 0., 0., xmin, zmin, 0., 0., 0.])
        to_physical = np.array([1., 1., 1., pixelsize, pixelsize, prefactor, pixelsize, pixelsize])

        # Converts the fit results to absolute pixel values in the unbinned image.

        fit_results = shot_info * bin_scaling + rng_offset
        phys_results = fit_results * to_physical

        tof = 0
        ntherm = 0
        ntf = 0
        tx = 0
        tz = 0
        mux = 0
        muz = 0
        mun = 0

        ntherm = 2 * np.pi * phys_results[5] * phys_results[6] * phys_results[7] 
        tx = 1 / kB * m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
        tz = 1 / kB * m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
#         mux = m / 1 * (fx * np.pi * 2 * phys_results[6]) ** 2 / (1 + (tof * fx * np.pi * 2) ** 2)
#         muz = m / 1 * (fz * np.pi * 2 * phys_results[7]) ** 2 / (1 + (tof * fz * np.pi * 2) ** 2)
#         mun = 1.47708846953 * np.power(
#             ntf * asc / (np.sqrt(hb / (m * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)))),
#             2. / 5.) * hb * np.power(8 * np.pi ** 3 * fx ** 2 * fz, 1. / 3.)

        phys_vars.append([ntf, ntherm, tx, tz, mux, muz, mun])
    return phys_vars    

def cut_arr(arr, cut_v):
    return np.array(arr[cut_v[0]:cut_v[1], cut_v[2]:cut_v[3]])

def get_nr_of_atoms(date, shot):
    fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/0.fits'
    nr = len(pyfits.open(fits_path)[0].data.astype(float))
    return nr

def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def fit_1d_data(ydata):
    xdata = range(len(ydata))
    # Initial guess: amplitude, center, width, offset
    p0 = [max(ydata), xdata[np.argmax(ydata)], np.std(xdata), min(ydata)]
    # Fit the data
    try:
        popt, pcov = curve_fit(gaussian, xdata, ydata, p0=p0)
    
    except: 
        popt = np.array([0, 0, 0, 0])

    return popt  # fitted parameters: a, x0, sigma, offset

save_folder = "/home/bec_lab/Desktop/imgs/SOAH/SpinAnalysisApril2025/"

# Load the run
all_params = [
    [20250418,  73],  # 0
    [20250418,  80],  # 1
    [20250418,  83],  # 2
    [20250418,  88],  # 3
    [20250418,  94],  # 4
    [20250418,  99],  # 5
    [20250418, 102],  # 6
    [20250418, 103],  # 7
    [20250418, 104],  # 8
    [20250418, 105],  # 9
    [20250418, 106],  # 10
    [20250418, 107],  # 11
    [20250418, 108],  # 12
    [20250418, 109],  # 13
    [20250418, 110],  # 14
    [20250418, 111],  # 15
    [20250418, 112],  # 16
    [20250418, 113],  # 17
    [20250418, 114],  # 18
    [20250418, 115],  # 19
    [20250422,  16],  # 20
    [20250422,  18],  # 21
    [20250422,  20],  # 22
    [20250422,  21],  # 23
    [20250422,  21],  # 24
    [20250422,  22],  # 25
    [20250422,  24],  # 26
    [20250422,  25],  # 27
    [20250422,  26],  # 28
    [20250422,  27],  # 29
    [20250422,  28],  # 30
    [20250422,  29],  # 31
    [20250422,  30],  # 32
    [20250422,  31],  # 33
    [20250422,  32],  # 34
    [20250422,  33],  # 35
    [20250422,  65],  # 36
    [20250422,  67],  # 37
    [20250423,  29],  # 38
    [20250423,  43],  # 39
    [20250423,  46],  # 40
    [20250423,  49],  # 41
    [20250423,  50],  # 42
    [20250423,  51],  # 43
    [20250423,  56],  # 44
    [20250423,  64],  # 45
    [20250423,  67],  # 46
    [20250423,  69],  # 47
    [20250423,  70],  # 48
    [20250423,  71],  # 49
    [20250423,  72],  # 50
    [20250423,  73],  # 51
    [20250423,  74],  # 52
    [20250423,  75],  # 53
    [20250423,  76],  # 54
    [20250423,  77],  # 55
    [20250423,  78],  # 56
    [20250423,  79],  # 57
    [20250423,  81],  # 58
    [20250423,  82],  # 59
    [20250424,  21],  # 60
    [20250424,  24],  # 61
    [20250424,  37],  # 62
    [20250424,  38],  # 63
    [20250424,  40],  # 64
    [20250424,  41],  # 65
    [20250424,  42],  # 66
    [20250424,  43],  # 67
    [20250424,  44],  # 68
    [20250424,  45],  # 69
    [20250424,  46],  # 70
    [20250424,  47],  # 71
    [20250424,  48],  # 72
    [20250424,  51],  # 73
    [20250424,  54],  # 74
    [20250424,  56],  # 75
    [20250424,  84],  # 76
    [20250424,  85],  # 77
    [20250424,  88],  # 78
    [20250424,  90],  # 79
    [20250424,  95],  # 80
    [20250424,  96],  # 81
    [20250424,  99],  # 82
    [20250424, 102],  # 83
    [20250424, 104],  # 84
    [20250424, 110],  # 85
    [20250425,  19],  # 86
    [20250425,  21],  # 87
    [20250425,  22],  # 88
    [20250425,  23],  # 89
    [20250425,  24],  # 90
    [20250425,  25],  # 91
    [20250425,  26],  # 92
    [20250425,  31],  # 93
    [20250425,  33],  # 94
    [20250425,  34],  # 95
    [20250425,  35],  # 96
    [20250425,  41],  # 97
    [20250425,  42],  # 98
    [20250425,  43],  # 99
    [20250425,  44],  # 100
    [20250428,  10],  # 101
    [20250428,  11],  # 102
    [20250428,  12],  # 103
    [20250428,  13],  # 104
    [20250428,  14],  # 105
    [20250428,  15],  # 106
    [20250428,  16],  # 107
    [20250428,  17],  # 108
    [20250428,  18],  # 109
    [20250428,  19],  # 110
    [20250428,  20],  # 111
    [20250428,  21],  # 112
    [20250428,  22],  # 113
    [20250428,  23],  # 114
    [20250428,  24],  # 115
    [20250428,  25],  # 116
    [20250428,  26],  # 117
    [20250428,  46],  # 118
    [20250428,  47],  # 119
    [20250428,  48],  # 120
    [20250428,  49],  # 121
    [20250428,  50],  # 122
    [20250428,  51],  # 123
    [20250428,  52],  # 124
    [20250428,  53],  # 125
    [20250428,  54],  # 126
    [20250428,  55],  # 127
    [20250428,  56],  # 128
    [20250428,  57],  # 129
    [20250428,  58],  # 130
    [20250428,  59],  # 131
    [20250428,  60],  # 132
    [20250428,  61],  # 133
    [20250428,  62],  # 134
    [20250428,  63],  # 135
    [20250428,  64],  # 136
    [20250428,  65],  # 137
    [20250428,  66],  # 138
    [20250428,  67],  # 139
    [20250428,  68],  # 140
    [20250428,  69],  # 141
    [20250428,  70],  # 142
    [20250428,  71],  # 143
    [20250428,  72],  # 144
    [20250428,  73],  # 145
    [20250428,  74],  # 146
    [20250428,  75],  # 147
    [20250428,  76],  # 148
    [20250428,  77],  # 149
    [20250428,  78],  # 150
    [20250428,  79],  # 151
    [20250428,  80],  # 152
    [20250428,  81],  # 153
    [20250428,  82],  # 154
    [20250428,  83],  # 155
    [20250428,  84],  # 156
    [20250430,   6],  # 157
    [20250430,   7],  # 158
    [20250430,   8],  # 159
    [20250430,   9],  # 160
    [20250430,  10],  # 161
    [20250430,  11],  # 162
    [20250430,  12],  # 163
    [20250430,  13],  # 164
    [20250430,  14],  # 165
    [20250430,  15],  # 166
    [20250430,  16],  # 167
    [20250430,  17],  # 168
    [20250430,  18],  # 169
    [20250430,  19],  # 170
    [20250430,  20],  # 171
    [20250430,  23],  # 172
    [20250430,  25],  # 173
    [20250430,  26],  # 174
    [20250430,  28],  # 175
    [20250430,  32],  # 176
    [20250430,  33],  # 177
    [20250430,  34],  # 178
    [20250430,  37],  # 179
    [20250430,  49],  # 180
    [20250430,  50],  # 181
    [20250430,  51],  # 182
    [20250430,  52],  # 183
    [20250430,  53],  # 184
    [20250430,  56],  # 185
    [20250430,  58],  # 186
    [20250430,  60],  # 187
    [20250430,  61],  # 188
    [20250430,  62],  # 189
    [20250430,  64],  # 190
    [20250430,  68],  # 191
    [20250430,  70],  # 192
    [20250430,  71],  # 193
    [20250430,  72],  # 194
    [20250430,  74],  # 195
    [20250430,  76],  # 196
    [20250430,  77],  # 197
    [20250430,  83],  # 198
    [20250430,  84],  # 199
    [20250430,  85],  # 200
    [20250430,  91],  # 201
    [20250430,  92],  # 202
    [20250430,  93],  # 203
    [20250430,  96],  # 204
    [20250430,  97],  # 205
    [20250430,  98],  # 206
    [20250430,  99],  # 207
    [20250430, 103],  # 208
    [20250430, 104],  # 209
    [20250430, 106],  # 210
    [20250430, 108],  # 211
    [20250430, 109],  # 212
    [20250430, 110],  # 213
    [20250430, 111],  # 214
    [20250430, 114],  # 215
    [20250430, 115],  # 216
    [20250430, 116],  # 217
    [20250501,  34],  # 217
    [20250501,  35],  # 217
    [20250501,  36],  # 217
    [20250501,  39],  # 217
    [20250501,  40],  # 217
    [20250501,  41],  # 217
    [20250501,  42],  # 217
    [20250501,  43],  # 217
    [20250501,  44],  # 217
    [20250501,  45],  # 217
    [20250501,  46],  # 217
    [20250501,  47],  # 217
    [20250501,  48],  # 217
    [20250501,  50],  # 217
    [20250501,  51],  # 217
    [20250501,  52],  # 217
    [20250501,  56],  # 217
    [20250501,  57],  # 217
    [20250501,  58],  # 217
    [20250501,  59],  # 217
    [20250501,  60],  # 217
    [20250501,  61],  # 217
    [20250501,  62],  # 217
    [20250501,  63],  # 217
    [20250501,  64],  # 217
    [20250501,  65],  # 217
    [20250501,  66],  # 217
    [20250501,  67],  # 217
    [20250501,  68],  # 217
    [20250501,  69],  # 217
    [20250501,  72],  # 217
    [20250501,  73],  # 217
    [20250501,  74],  # 217
    [20250501,  75],  # 217
    [20250501,  76],  # 217
    [20250501,  77],  # 217
    [20250501,  78],  # 217
    [20250501,  79],  # 217
    [20250501,  80],  # 217
    [20250501,  81],  # 217
    [20250501,  82],  # 217
    [20250501,  83],  # 217
    [20250501,  84],  # 217
    [20250502,  11],
    [20250502,  12],
    [20250502,  13],
    [20250502,  14],
    [20250502,  16],
    [20250502,  17],
    [20250502,  18],
    [20250502,  19],
    [20250502,  21], 
    [20250502,  24], 
    [20250502,  27], 
    [20250502,  28], 
    [20250502,  31], 
    [20250502,  32], 
    [20250502,  34], 
    [20250502,  37], 
    [20250502,  38], 
    [20250502,  39], 
    [20250502,  40], 
    [20250502,  41], 
    [20250502,  43], 
    [20250502,  44], 
    [20250502,  45], 
    [20250502,  46], 
    [20250502,  47], 
    [20250502,  48], 
    [20250502,  60], 
    [20250502,  65], 
    [20250502,  66], 
    [20250502,  69], 
    [20250502,  71], 
    [20250502,  74], 
    [20250502,  75], 
    [20250502,  76], 
    [20250502,  77], 
    [20250502,  78], 
    [20250502,  79], 
    [20250502,  80], 
    [20250502,  81], 
    [20250502,  82], 
    [20250502,  83], 
    [20250502,  84], 
    [20250502,  86], 
    [20250502,  88], 
    [20250502,  89], 
    [20250502,  92], 
    [20250502,  96], 
    [20250502,  95], 
    [20250502,  98], 
    [20250502,  99], 
    [20250502,  101], 
    [20250502,  102], 
    [20250502,  103], 
    [20250502,  104], 
    [20250502,  105], 
    [20250502,  107], 
    [20250502,  108], 
    [20250502,  109], 
    [20250502,  110], 
    [20250502,  111], 
    [20250502,  112], 
    [20250502,  113], 
    [20250502,  114], 
    [20250502,  115], 
    [20250502,  116], 
    [20250502,  117], 
    [20250502,  118], 
    [20250502,  119], 
    [20250502,  120], 
    [20250502,  121], 
    [20250502,  122], 
    [20250502,  124], 
    [20250507,  2],
    [20250507,  3],
    [20250507,  4],
    [20250507,  5],
    [20250507,  6],
    [20250507,  7],
    [20250507,  8],
    [20250507,  9],
    [20250507,  10],
    [20250507,  11],
    [20250507,  12],
    [20250507,  16],
    [20250507,  17],
    [20250507,  18],
    [20250507,  19],
    [20250507,  22],
    [20250507,  23],
    [20250507,  24],
    [20250507,  25],
    [20250507,  26],
    [20250507,  27],
    [20250507,  28],
    [20250507,  29],
    [20250507,  30],
    [20250507,  31],
    [20250507,  32],
    [20250507,  33],
    [20250507,  34],
    [20250507,  35],
    [20250507,  37],
    [20250507,  38],
    [20250507,  39],
    [20250507,  40],
    [20250507,  41],
    [20250507,  42],
    [20250507,  43],
    [20250507,  44],
    [20250507,  45],
    [20250507,  46],
    [20250507,  47],
    [20250507,  48],
    [20250507,  49],
    [20250507,  50],
    [20250507,  51],
    [20250507,  52],
    [20250507,  53],
    [20250507,  55],
    [20250507,  56],
    [20250507,  57],
    [20250507,  58],
    [20250507,  59],
    [20250507,  60],
    [20250507,  61],
    [20250507,  62],
    [20250507,  63],
    [20250507,  64],
    [20250507,  65],
    [20250507,  66],
    [20250507,  67],
    [20250507,  68],
    [20250507,  69],
    [20250507,  70],
    [20250507,  71],
    [20250507,  73],
    [20250507,  74],
    [20250507,  76],
    [20250507,  77],
    [20250507,  78],
    [20250507,  79],
    [20250507,  83],
    [20250507,  84],
    [20250507,  85],
    [20250507,  86],
    [20250507,  87],
    [20250507,  88],
    [20250507,  89],
    [20250507,  92],
    [20250507,  95],
]




save_folder = "/home/bec_lab/Desktop/imgs/SOAH/SpinAnalysisApril2025/"


for params in all_params:
    # params = all_params[-1]  # <--- 
    date = params[0]
    shot = params[1]
    print(shot)

    # Load or process
    if os.path.exists(f"{save_folder}data/{date}_{shot}_ang_is.npy"):
        print("Already processed.")

    else: 
        dz_focus = 0.0055
        num_of_nums = get_nr_of_atoms(date, shot)
        ang_is = [[HI_refocus_custom(date, shot, num, dz_focus, num_flat=num, shift_pix=[0, shift_z], quad=q, output="ang")for q, shift_z in zip(["quad1", "quad2"], [0, -0.48])] for num in range(num_of_nums)] # try different dark images, to avoid features
        np.save(f"{save_folder}data/{date}_{shot}_ang_is", ang_is)

