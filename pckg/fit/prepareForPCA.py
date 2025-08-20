import numpy as np
import astropy.io.fits as pyfits
from OAH_refocus import *
import OAH_functions as f1
from OAHDEV_functions import *

print(xmin, xmax, zmin, zmax)
def HI_save_complex(
    date, shot, num, dz_focus, 
    quad="quad1", num_flat=0, 
    shift_pix=(0, 0), 
    filetype="atoms", 
    nns=0,
):
    """
    Reconstructs complex field from off-axis holography image.

    Parameters:
        date (str or int): Date of the experiment (used in file path).
        shot (int): Shot number (used in folder name).
        num (int): Frame number within the shot.
        dz_focus (float): Propagation distance for angular spectrum method.
        quad (str): Which off-axis quadrant to extract (default "quad1").
        num_flat (int): Frame index for flat image (optional).
        shift_pix (tuple): Pixel shift (x, z) to apply in Fourier space.
        output (str): Output mode (currently unused).
        filetype (str): Either "atoms" or "flat" to specify which file to load.
        plot (bool): Whether to show intermediate plots (currently unused).
        cut (list or tuple): [xmin, xmax, zmin, zmax] region to crop from image.

    Returns:
        inv1 (ndarray): Reconstructed complex image.
    """
    xmin, xmax, zmin, zmax = (1, -1, 1, -1)  # Default cropping values
    num_flat = num_flat % num if num != 0 else num_flat
    shift_x, shift_z = shift_pix

    # Build paths
    base_path = f"/storage/data/{date}/{str(shot).zfill(4)}/"
    print(f"Processing shot: {date} - {str(shot).zfill(4)}, frame: {num}/{nns}, quad: {quad}, flat frame: {num_flat}, shift: {shift_pix}")

    # Load image
    if filetype == "atoms":
        atoms = pyfits.open(base_path + '0.fits')[0].data.astype(float)[num][xmin:xmax, zmin:zmax]
    elif filetype == "flat":
        atoms = pyfits.open(base_path + '1.fits')[0].data.astype(float)[num][xmin:xmax, zmin:zmax]
    else:
        raise ValueError("Invalid filetype: choose 'atoms' or 'flat'")

    # Load dark and subtract
    dark = pyfits.open(base_path + '2.fits')[0].data.astype(float).mean(axis=0)[xmin:xmax, zmin:zmax]
    atoms_corrected = f1.squaroid(atoms - dark, width=0.51)

    # Fourier transform
    fft_atoms = np.fft.fft2(atoms_corrected)
    quad_cut, q1peak = f1.box_cutter_pad_ellips(fft_atoms, quad, 0, 0, edge_x=10, edge_z=80)
    fft_shifted = np.fft.fftshift(quad_cut)

    # Phase ramp for pixel shift
    Nz, Nx = quad_cut.shape
    kx = np.fft.fftfreq(Nx)
    kz = np.fft.fftfreq(Nz)
    KX, KZ = np.meshgrid(kx, kz)
    phase_ramp = np.exp(-1j * 2 * np.pi * (KX * shift_x + KZ * shift_z))

    # Angular spectrum propagation
    fft_kx = np.fft.fftfreq(Nx, d=pix_size)
    fft_ky = np.fft.fftfreq(Nz, d=pix_size)
    fft_k2 = fft_kx[None, :]**2 + fft_ky[:, None]**2
    focus_kernel = np.exp(-1j * fft_k2 * dz_focus / (2 * k0))

    # Apply shift and propagation
    fft_propagated = fft_shifted * focus_kernel * phase_ramp
    inv1 = np.fft.ifft2(fft_propagated)

    # Return a non-normalized complex field
    return inv1

def get_nr_of_atoms(date, shot):
    fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/0.fits'
    nr = len(pyfits.open(fits_path)[0].data.astype(float))
    return nr

def get_nr_of_flats(date, shot):
    fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/1.fits'
    nr = len(pyfits.open(fits_path)[0].data.astype(float))
    return nr


dz_focus = 0.0055

from PCA_LIST import all_params

for params in all_params:
    date = params[0]
    shot = params[1]
    num_of_nums = get_nr_of_atoms(date, shot)

    # Check if exists:
    if not os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/OAH_processing/full_complex_field_{date}_{shot}_atoms.npy"):
        print(f"Processing shot {date} - {shot}...")
        for filetype in ["atoms", "flat"]: 
            ang_ful_complex = [[HI_save_complex(date, shot, num, dz_focus, num_flat=num, filetype=filetype, shift_pix=[0, shift_z], quad=q, nns=num_of_nums) for q, shift_z in zip(["quad1", "quad2"], [0, -0.48])] for num in range(num_of_nums)]
            np.save(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/OAH_processing/full_complex_field_{date}_{shot}_{filetype}", ang_ful_complex)
    else:
        print(f"Shot {date} - {shot} already processed, skipping...")
