# Core libraries
import numpy as np
from scipy.signal.windows import tukey

from scipy.ndimage import gaussian_filter

# Custom OAH utilities
from OAH_refocus import *
from OAH_functions import *
from OAHDEV_functions import *

# Constants:
kB = 1.38064852E-23
m = 3.81923979E-26
hb = 1.0545718E-34
asc = 2.802642E-9
mu0 = 1E-50
e0 = 8.854187E-12
pix_size = 6.5E-6 / 2.63
lamb0 = 589.1E-9  # Wavelength
k0 = 2 * np.pi / lamb0  # k-vector

rng=np.random.default_rng() 
size=(2046, 2046) 
border=0


def condensate_noise(noise_strength=0.3, noise_scale=10, size=size):
    """
    Generate a 2D Gaussian condensate with Perlin-like noise.

    Parameters:
    - Nx, Ny: Dimensions of the output array.
    - wx, wy: Widths of the Gaussian envelope in x and y.
    - noise_strength: Amplitude of noise modulation.
    - noise_scale: Smoothing scale for Perlin-like noise.
    - seed: Optional random seed for reproducibility.

    Returns:
    - Xgrid, Ygrid: Meshgrids of the coordinates.
    - condensate: Noisy condensate matrix.
    """
    np.random.seed(4)
    raw_noise = np.random.rand(size[1], size[0])
    perlin_like = gaussian_filter(raw_noise, sigma=noise_scale)
    perlin_like = (perlin_like - perlin_like.min()) / (perlin_like.max() - perlin_like.min())
    return (1 + noise_strength * (perlin_like - 0.5))


# Example usage

def makeInterference(size=size, angle1=200, angle2=240, 
                     condensate=0, wx=0.002, wy=0.1, noise_params=[0, 5],
                     curvature_probe=0.001, curvature1=0.001, curvature2=0.001, 
                     windowalpha=0.01, windowtype="tukey", 
                     ref_crosstalk=False):
    """ 
    Create an interference pattern of a probe and two reference beams with curved wavefronts.
    angle1: Angle of the first reference beam, in degrees?
    angle2: Angle of the second reference beam.
    condensate: 0 or 1 - Whether to include a condensate-like phase object.
    noise_params: parameters for the noise in domains; [noise strenght, noise scale].
    wx: Gaussian width in x.
    wy: Gaussian width in y.
    curvature_probe: Curvature parameter for the probe beam.
    curvature1: Curvature parameter for the first reference beam.
    curvature2: Curvature parameter for the second reference beam.
    """
    
    # Create grid for image
    x = np.linspace(-1, 1, size[0] + 2 * border)
    y = np.linspace(-1, 1, size[1] + 2 * border)
    Xgrid, Ygrid = np.meshgrid(x, y)
    
    # Create a condensate-like phase object
    gaussian_matrix = condensate * np.exp(-((Xgrid / wx)**2 + (Ygrid / wy)**2)) * condensate_noise(noise_strength=noise_params[0], noise_scale=noise_params[1], size=size)
    True_phase = np.exp(2.0j * gaussian_matrix)
    
    # Define wave vectors based on the input angles
    k0 = angle1 * 2 * np.pi            # First reference beam wave number
    k1 = k0 * np.array([1.0, 1.0])     # First beam direction vector
    k02 = angle2 * 2 * np.pi           # Second reference beam wave number
    k2 = k02 * np.array([-1.0, 1.0])   # Second beam direction vector

    # Create the probe beam with curvature
    probe_beam = np.exp(1.0j * curvature_probe * (Xgrid**2 + Ygrid**2))
    
    # Create reference fields with curvature terms
    # Curvature terms: exp(j * curvature * (X^2 + Y^2)) 
    ref1 = np.exp(1.0j * np.tensordot(k1, [Xgrid, Ygrid], axes=[[0], [0]])) * np.exp(1.0j * curvature1 * (Xgrid**2 + Ygrid**2))
    ref2 = np.exp(1.0j * np.tensordot(k2, [Xgrid, Ygrid], axes=[[0], [0]])) * np.exp(1.0j * curvature2 * (Xgrid**2 + Ygrid**2))

#     # Generate interference patterns with the probe beam   
    if ref_crosstalk:
        InterferencePatternTotal = np.real((True_phase * probe_beam  + ref1 + ref2) * (True_phase * probe_beam  + ref1 + ref2).conj())
    
    else:
        InterferencePattern1 = np.real(True_phase * probe_beam * ref1) + 0.5
        InterferencePattern2 = np.real(True_phase * probe_beam * ref2) + 0.5
        InterferencePatternTotal = InterferencePattern1 + InterferencePattern2

    # Apply a window function for smoother edges in Fourier space
    window = 1
    if windowtype == "tukey":
        window = tukey(size[1], alpha=windowalpha, sym=True)[:, np.newaxis] * tukey(size[0], alpha=windowalpha, sym=True)[np.newaxis, :]
    
    elif windowtype == "oval":
        window = oval_window(size, alpha=windowalpha, aspect_ratio=1.5)

    elif windowtype == "triangle":
        window = triangular_window(size)

    elif windowtype == "circle":
        window = circular_window(size, radius=0.5)

    elif windowtype == "house":
        window = house_window(size, width_ratio=windowalpha, roof_height_ratio=0.3)

    Fourier_space = np.fft.fftshift(np.fft.fft2(InterferencePatternTotal * window))

    return InterferencePatternTotal * window, Fourier_space



def oval_window(size, alpha=0.5, aspect_ratio=1.5):
    """
    Creates an oval-shaped Tukey window.
    size: int, Size of the window (NxN).
    alpha: float, Shape parameter for the Tukey window (between 0 and 1).
    aspect_ratio: float, Ratio of width to height for the oval shape.
    """
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X**2) + (Y**2 / aspect_ratio**2))
    window = np.clip(1 - R, 0, 1) * tukey(size[0], alpha)[:, np.newaxis] * tukey(size[1], alpha)[np.newaxis, :]
    return window

def triangular_window(size):
    """
    Creates a triangular window.
    size: int, Size of the window (NxN).
    """
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    X, Y = np.meshgrid(x, y)
    R = np.abs(X) + np.abs(Y)
    window = np.clip(1 - R, 0, 1)
    return window

def circular_window(size, radius=0.5):
    """
    Creates a circular window.
    size: int, Size of the window (NxN).
    radius: float, Radius of the circle (between 0 and 1).
    """
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    window = np.clip(1 - (R / radius), 0, 1)
    return window


def house_window(size, width_ratio=0.9, roof_height_ratio=0.3, roof_offset=0.6):
    """
    Creates a house-shaped window (a trapezoid with a triangular roof shifted upwards).
    size: int, Size of the window (NxN).
    width_ratio: float, Width of the base relative to the image width.
    roof_height_ratio: float, Height of the roof relative to the image height.
    roof_offset: float, Vertical shift of the roof relative to the center of the base.
    """
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    X, Y = np.meshgrid(x, y)

    # Define the base (trapezoid) of the house
    base_window = np.clip(1 - np.abs(Y) / (1 - roof_height_ratio), 0, 1) * (np.abs(X) <= width_ratio)

    # Shift the triangular roof upward by `roof_offset`
    roof_window = np.clip(1 - (np.abs(X) / (2*width_ratio) + (-Y - roof_offset) / roof_height_ratio), 0, 1) * (Y <= -roof_offset + roof_height_ratio)

    # Combine base and roof to form the "house" shape
    window = np.maximum(base_window, roof_window)
    return window



border_x = 50
border_z = 50

def HI_refocus_synthetic(atoms, flat, dz_focus, quad="quad1", cut=[xmin, xmax, zmin, zmax]):
    """
    Main function for generating the output arrays for the holgoraphic imaging, which also includes the refocusing.
    What we do, is we process image - first take a FTT, apply Tukey windows, shift, refocus, and back iFFT.
    ------
    :param date: Input date of the image to be analysed
    :param shot: Input shot of the image to be analysed
    :param num: Input sequence number of the image to be analysed
    :param dz_focus: The focus parameter that determines the focus of the image
    :param quad: Parameter to determine which quad we are cutting.
    :param plot: Boolean to specify whether to plot the image of the angle or not. Better not when used in a loop.
    :return: Returns an array of the angle of the ratio of the iFFT's of the atom and flat image.
    """
    # ----------------------------------------------- CORRECTIONS ---------------------------------------------------
    # Creates a squaroid dark edge
    atoms = f1.squaroid(atoms, width=0.51)
    flat = f1.squaroid(flat, width=0.51)

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
#     ang1 = normalize(ang1)[0] # Use the function above to normalize the image.
    
    return ang1




