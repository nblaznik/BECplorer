# Script to study the phase of the atoms, normalised to the background. Taken and modified based on Jasper's old script
# for such analysis.

# The main function in this script takes .fits files taken through OAH, and prepares the files in a single .fits file,
# where the data has been analysed. Later on, these files can be analyzed.

# Last updated in November 2021 by Nejc Blaznik

# ---------------------------------------------------- IMPORTS -------------------------------------------------------
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os
# from matplotlib.widgets import Slider
from skimage.restoration import unwrap_phase
from numpy.polynomial import chebyshev as cheb

# Import own fit functions
try:
    from ..fit import OAH_functions as f1
except:
    import OAH_functions as f1
# from ..data_analysis import fits_analysis as fa
from scipy.optimize import curve_fit

# --------------------------------------------------- CONSTANTS ------------------------------------------------------
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
pix_size = 6.5E-6 / 2.63

# -------------------------------------------------- PARAMETERS ------------------------------------------------------
# Frequencies
# MT
# fx = 91.9
# fz = 15.2
# Decompressed trap 2.5
fx = 59.97
fz = 15.028
# Decompressed trap 1.5
# fx   = 44.6
# fz   = 15.4
# --------------------------------------------------
# Light field properties
lamb0 = 589.1E-9  # Frequency
k0 = 2 * np.pi / lamb0  # k-vector
# --------------------------------------------------
# Detuning
det_1 = -2. * 87.5 - 2 * 57.5
det_0 = det_1 + 15.8
det_2 = det_1 - 34.4
linew = 9.7
# --------------------------------------------------
# Physical Parameters
ntherm = 0.
ntf = 0.
tx = 0.
tz = 0.
mux = 0.
muz = 0.
mun = 0.
ncount = 0.
ntotal = 0.
TF_mu = 0.
# --------------------------------------------------
# Polarizability
polar2 = f1.polarizability(det_1, 0, lamb0) * k0 / 2.
polariz = polar2.real
polariz_i = polar2.imag
# --------------------------------------------------
border_x = 5
border_z = 50
# --------------------------------------------------
# Define the cutout if need to be
# ((xmin, xmax), (zmin, zmax)) = ((900, 1200), (500, 1300))
# ((xmin, xmax), (zmin, zmax)) = ((1, -1), (50, 1750))
((xmin, xmax), (zmin, zmax)) = ((1, -1), (1, -1))


# --------------------------------------------------
def normalize(ang):
    """
    Returns the normalized image of the cloud - it fits each horizontal line of the image with a simple linear fit.
    Then it performs the vertical linear fit. This way, we get the background image, which we subtract from the actual image.
    It seems however that we might even have some quadratic, or maybe a gaussian would be easier?
    """

    def linear_f(x, a, b):
        """ Faster, but does poor job. """
        return a * x + b

    def quadratic_f(x, a, b, c):
        """ Slower, but better equipped for normalizing"""
        return a * x ** 2 + b * x + c

    xdata = np.arange(0, len(ang[0]))
    ydata = np.arange(0, len(ang))
    arr_gradient = []
    arr_gradient_2 = []
    # Linear fit - horizontal gradient (for each line individualy)
    for i in range(ang.shape[0]):
        pars, cov = curve_fit(f=quadratic_f, xdata=xdata, ydata=ang[i], p0=[0, 0, 0])
        arr_line = quadratic_f(xdata, pars[0], pars[1], pars[2])
        arr_gradient.append(arr_line)
    lingrad = arr_gradient.copy()

    # Linear fit - vertical gradient (for each line individualy)
    arr_gradient = np.transpose(arr_gradient)  # Transpose
    for i in range(ang.shape[1]):
        pars, cov = curve_fit(f=quadratic_f, xdata=ydata, ydata=arr_gradient[i], p0=[0, 0, 0])
        arr_line = quadratic_f(ydata, pars[0], pars[1], pars[2])
        arr_gradient_2.append(arr_line)
    arr_gradient = np.transpose(arr_gradient_2)  # Transpose back

    ang = ang - arr_gradient
    return ang, arr_gradient, lingrad


# def normalize(image, order_x=7, order_y=3):
#     """
#     Normalize an image by removing smooth background gradients using Chebyshev polynomial fitting.

#     First fits each horizontal row with a Chebyshev polynomial (along X),
#     then fits each vertical column of the fitted background (along Y).

#     Parameters:
#         image (2D np.ndarray): Input phase or intensity image.
#         order_x (int): Polynomial order for horizontal fitting.
#         order_y (int): Polynomial order for vertical fitting.

#     Returns:
#         normalized_image: Image with smooth background subtracted.
#         background: 2D fitted background that was subtracted.
#         intermediate_fit: Fit after horizontal direction (before vertical fitting).
#     """
#     xdim, ydim = image.shape
#     x = np.linspace(-1, 1, ydim)  # Horizontal axis (columns)
#     y = np.linspace(-1, 1, xdim)  # Vertical axis (rows)

#     # --- Step 1: Horizontal fitting (row-wise)
#     intermediate_fit = np.zeros_like(image)
#     for i in range(xdim):
#         coeffs = cheb.chebfit(x, image[i, :], order_x)
#         intermediate_fit[i, :] = cheb.chebval(x, coeffs)

#     # --- Step 2: Vertical fitting (column-wise)
#     background = np.zeros_like(image)
#     for j in range(ydim):
#         coeffs = cheb.chebfit(y, intermediate_fit[:, j], order_y)
#         background[:, j] = cheb.chebval(y, coeffs)

#     # --- Normalize
#     normalized_image = image - background
#     return normalized_image, background, intermediate_fit


def HI_refocus(date, shot, num, dz_focus, quad="quad1", output="ang", plot=False, cut=[xmin, xmax, zmin, zmax]):
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

    # Opening files
    atoms = pyfits.open(path + image + '0.fits')[0].data.astype(float)[num][xmin:xmax, zmin:zmax]
    flat = pyfits.open(path + image + '1.fits')[0].data.astype(float)[0][xmin:xmax, zmin:zmax]
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
    
    
    # Cutting the quads in the same sizes.
    quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

    # ------------------------------------------------ FFT SHIFT ----------------------------------------------------
    # Now we FFT shift the zero frequency to the center
    fft1 = np.fft.fftshift(quad1cut)
    flatfft1 = np.fft.fftshift(flatq1cut)

    # ------------------------------------------------ REFOCUSING ---------------------------------------------------
    fft_kx = np.fft.fftfreq(fft1.shape[1], d=pix_size)  # Discrete FFT Sample Frequency in x
    fft_ky = np.fft.fftfreq(fft1.shape[0], d=pix_size)  # Discrete FFT Sample Frequency in z
    fft_k2 = fft_kx[None, :] ** 2 + fft_ky[:, None] ** 2  # Discrete FFT Sample Frequency in main axes multiplied
    ly = 0  # .5E6#-3E6#.                                           # Adjusting the fft_ky array
    coma_y_arg = ly * fft_ky[:, None] * (3 * fft_k2 / k0 ** 2) / k0
    lin_y = np.exp(-1j * coma_y_arg)

    # Determine the focus factor and refocus
    focus = np.exp(-1j * fft_k2 * dz_focus / (2 * k0))
    fft1 = fft1 * focus * lin_y
    flatfft1 = flatfft1 * focus * lin_y

    # ------------------------------------- INVERSE FFT -------------------------------------------------
    inv1 = np.fft.ifft2(fft1) / np.fft.ifft2(flatfft1)
    inv1 = inv1[border_x:-border_x, border_z:-border_z]

    # Get Phase
    ang1 = np.angle(inv1)
    ang1 = f1.unwrapper(ang1)
    amp1 = np.abs(inv1) ** 2

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
    

def preprocessHI_refocus(date, shot, dz_focus, update, num="single", quad="quad1", output="ang"):
    """
    The function used to pre-process all the images, when multiple loops are used when imaging with OAH.
    It bascially calls the HI_refocus for each image within a run ID, and appends all the information in
    a single array file, and saves it as pics_{dz_focus}.fits.
    """
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    full_path = path + image

    # If the preprocessing was done before, take that file
    if os.path.exists(full_path + f"pics_foc_ss_{quad}_{output}_{dz_focus}.fits"):
        print("Analysis previously completed")
        output_OAH = pyfits.open(full_path + f"pics_foc_ss_{quad}_{output}_{dz_focus}.fits")[0].data.astype(float)[:]
        return output_OAH
    
    elif os.path.exists(full_path + f"pics_foc_ss_{quad}_{output}_{num}_{dz_focus}.fits"):
        print("Analysis previously completed")
        output_OAH = pyfits.open(full_path + f"pics_foc_ss_{quad}_{output}_{num}_{dz_focus}.fits")[0].data.astype(float)[:]
        return output_OAH
    
    elif os.path.exists(full_path + "pics_foc_{:}_{:}.fits".format(quad, dz_focus)):
        print("Analysis previously completed")
        output_OAH = pyfits.open(full_path + "pics_foc_{:}_{:}.fits".format(quad, dz_focus))[0].data.astype(float)[:]
        return output_OAH

    # Import .fits file to get the number of shots
    atom_all = pyfits.open(path + image + '0.fits')[0].data.astype(float)[:]
    out_arr = []

    # If a single shot only
    if atom_all.shape[0] == 1:
        update.setValue(50)
        ang = HI_refocus(date, shot, 0, dz_focus, quad=quad, output=output, plot=False)
        output_OAH = [ang]  # We do this, so that we can call the 0th element of the group, as we do with multiple images.
        prihdr = pyfits.open(full_path + "0.fits")[0].header  # Get the header
        hdu = pyfits.PrimaryHDU(output_OAH, header=prihdr)
        hdu.writeto(full_path + f"pics_foc_ss_{quad}_{output}_{dz_focus}.fits")
        update.setValue(100)
        return output_OAH

    elif num != "single":
        update.setValue(50)
        ang = HI_refocus(date, shot, num, dz_focus, quad=quad, output=output, plot=False)
        output_OAH = [ang]  # We do this, so that we can call the 0th element of the group, as we do with multiple images.
        prihdr = pyfits.open(full_path + "0.fits")[0].header  # Get the header
        hdu = pyfits.PrimaryHDU(output_OAH, header=prihdr)
        hdu.writeto(full_path + f"pics_foc_ss_{quad}_{output}_{num}_{dz_focus}.fits")
        update.setValue(100)
        return output_OAH
    
    else:
        for it in range(atom_all.shape[0]):
            os.system('clear')
            update.setValue(it * 100 // atom_all.shape[0])
            ang = HI_refocus(date, shot, it, dz_focus, quad=quad, output=output, plot=False)
            out_arr.append(ang)

        # Save as .fits
        prihdr = pyfits.open(full_path + "0.fits")[0].header  # Get the header
        hdu = pyfits.PrimaryHDU(out_arr, header=prihdr)
        hdu.writeto(full_path + "pics_foc_{:}_{:}.fits".format(quad, dz_focus))
        return out_arr


def preprocessHI_refocus_multi(date, shot, dz_focus, update, quad="quad1", ):
    """
    The function used to pre-process all the images, when multiple imaging loops are used when imaging with OAH.
    Because the saving is weird, we have to first combine all arrays, then get the seed from the weeds.
    Basically performing the HI_refocus on all shots, and saving them together in a single output array (and as a
    .fits file).
    """
    # ------------------------------------------------- IMPORTS -----------------------------------------------------
    path = '/storage/data/' + str(date) + '/'
    image = str(shot).zfill(4) + '/'
    full_path = path + image

    dz_focus = round(dz_focus, 4)

    # If the preprocessing was done before, take that file
    if os.path.exists(full_path + "pics_foc_{:}_{:}.fits".format(quad, dz_focus)):
        print("Analysis previously completed")
        output = pyfits.open(full_path + "pics_foc_{:}_{:}.fits".format(quad, dz_focus))[0].data.astype(float)[:]
        return output

    # Here we collect all the fits files that have only a single digit in the name.
    # We do this to avoid collecting other pre-processed files. Then we sort 0.fits -> X.fits.
    fits_files = [f for f in os.listdir(path + image) if (f.endswith('.fits') and len(f[:-5]) == 1)]
    fits_files = np.sort(fits_files)


    # Toss them all together in a single multi array.
    multi_array = []
    for file in fits_files:
        multi_array.append(pyfits.open(path + image + file)[0].data.astype(float)[:])

    nr_shots = len(np.concatenate(multi_array)) - 4

    multi_array = np.concatenate(multi_array)
    out_arr = []

    for i in range(nr_shots):
        os.system('clear')
        flat = multi_array[-4][xmin:xmax, zmin:zmax]
        dark = multi_array[-3][xmin:xmax, zmin:zmax]
        atoms_all = multi_array[:-4]
        atoms = atoms_all[i][xmin:xmax, zmin:zmax]
        print(i)
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
        quad1, q1peak = f1.box_cutter_pad_ellips(fft_atoms, quad, 10, 100, edge_x=10, edge_z=80)
        flatq1, f1peak = f1.box_cutter_pad_ellips(fft_flat, quad, indices=q1peak)

        # Cutting the quads in the same sizes.
        quad1cut, flatq1cut = f1.sizecomp(quad1, flatq1)

        # ------------------------------------------------ FFT SHIFT ----------------------------------------------------
        # Now we FFT shift the zero frequency to the center
        fft1 = np.fft.fftshift(quad1cut)
        flatfft1 = np.fft.fftshift(flatq1cut)

        # ------------------------------------------------ REFOCUSING ---------------------------------------------------
        fft_kx = np.fft.fftfreq(fft1.shape[1], d=pix_size)  # Discrete FFT Sample Frequency in x
        fft_ky = np.fft.fftfreq(fft1.shape[0], d=pix_size)  # Discrete FFT Sample Frequency in z
        fft_k2 = fft_kx[None, :] ** 2 + fft_ky[:, None] ** 2  # Discrete FFT Sample Frequency in main axes multiplied
        ly = 0  # .5E6#-3E6#.                                           # Adjusting the fft_ky array
        coma_y_arg = ly * fft_ky[:, None] * (3 * fft_k2 / k0 ** 2) / k0
        lin_y = np.exp(-1j * coma_y_arg)

        # Determine the focus factor and refocus
        focus = np.exp(-1j * fft_k2 * dz_focus / (2 * k0))
        # focus = np.exp(1j * np.sqrt(k0**2 - fft_k2) * dz_focus)  # From the paper by Zhao


        fft1 = fft1 * focus * lin_y
        flatfft1 = flatfft1 * focus * lin_y

        # ------------------------------------- INVERSE FFT -------------------------------------------------
        inv1 = np.fft.ifft2(fft1) / np.fft.ifft2(flatfft1)
        inv1 = inv1[border_x:-border_x, border_z:-border_z]

        # Get Phase
        ang1 = np.angle(inv1)
        ang1 = f1.unwrapper(ang1)

        # Normalize
        normfactor = ang1.mean()  # [300:900, 300:900].mean()
        ang1 = ang1 - normfactor

        ang = normalize(ang1)[0] # Use the function above to normalize the image.
        update.setValue(i * 100 // nr_shots)
        out_arr.append(ang)

    # Save as .fits
    prihdr = pyfits.open(full_path + "0.fits")[0].header  # Get the header
    hdu = pyfits.PrimaryHDU(out_arr, header=prihdr)
    hdu.writeto(full_path + "pics_foc_{:}_{:}.fits".format(quad, dz_focus))
    return out_arr
