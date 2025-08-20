import numpy as np
from skimage.restoration import unwrap_phase
import os

# ─────────────────────────────────────────────────────────────
# Constants and Default Settings
# ─────────────────────────────────────────────────────────────

# Default Chebyshev polynomial orders for gradient correction
gradientremovalorder_X_default = 2
gradientremovalorder_Y_default = 2

# Global containers for inpainting
empty_images = []
masked_empty_images = []
CovarianceMatrix = []
inversecovariancematrix = []
svd = []
left_svs = []
right_svs = []


# Default mask region (top, bottom, left, right)
if os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/CUT_COORDS.npy"):
    # Load mask from file if it exists
    wtop, wbottom, wleft, wright = np.load(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/CUT_COORDS.npy", allow_pickle=True).tolist()
    print(f"Mask loaded: {wtop}, {wbottom}, {wleft}, {wright}")
else:
    # Default mask if file does not exist
    print("No mask file found, using default values.")
    wtop, wbottom, wleft, wright = 35, 65, 867, 2234
    



mask = (wtop, wbottom, wleft, wright)
# Debugging level
debuglevel = 9

# ─────────────────────────────────────────────────────────────
# Phase Gradient Removal
# ─────────────────────────────────────────────────────────────

def RemovePhaseGradient(image, gradientremovalorder_X=gradientremovalorder_X_default, 
                        gradientremovalorder_Y=gradientremovalorder_Y_default, 
                        exclude_atoms_from_gradient=False):
    """
    Remove phase gradients from an image using Chebyshev polynomial fitting
    to the unwrapped phase.
    """
    xdim, ydim = image.shape
    image_unwrapped = unwrap_phase(np.angle(image))

    # Coordinate axes
    X_axis = np.linspace(-1, 1, xdim)
    Y_axis = np.linspace(-1, 1, ydim)

    # Bitmasking
    bitmask = np.abs(image)**2
    if exclude_atoms_from_gradient:
        bitmask[mask[0]:mask[1], mask[2]:mask[3]] = 0

    flat1 = np.sum(image_unwrapped * bitmask, axis=0) / np.sum(bitmask, axis=0)
    flat2 = np.sum(image_unwrapped * bitmask, axis=1) / np.sum(bitmask, axis=1)

    # Chebyshev fitting
    fit1 = np.polynomial.chebyshev.chebfit(Y_axis, flat1, gradientremovalorder_Y)
    fit2 = np.polynomial.chebyshev.chebfit(X_axis, flat2, gradientremovalorder_X)
    fit2[0] = 0  # Remove overall offset

    # Evaluate fitted background
    fiteval = (np.polynomial.chebyshev.chebval(Y_axis, fit1) +
               np.outer(np.ones_like(Y_axis), np.polynomial.chebyshev.chebval(X_axis, fit2)).T)

    return np.exp(-1.0j * fiteval) * image

# ─────────────────────────────────────────────────────────────
# Setting Empty Images for Inpainting
# ─────────────────────────────────────────────────────────────

def set_empty_images(imagelist, remove_gradients=False, exclude_atoms_from_gradient=False):
    """
    Preprocess and store empty reference images for inpainting.
    """
    global empty_images, masked_empty_images
    global CovarianceMatrix, inversecovariancematrix, svd 
    global left_svs, right_svs

    empty_images = imagelist.copy()

    if remove_gradients:
        for i in range(empty_images.shape[0]):
            empty_images[i] = RemovePhaseGradient(empty_images[i], exclude_atoms_from_gradient=exclude_atoms_from_gradient)

    masked_empty_images = empty_images.copy()
    # print(f"Setting empty images with shape: {empty_images.shape}")

    # Mask central region
    masked_empty_images[:, mask[0]:mask[1], mask[2]:mask[3]] = 0.0

    # Flatten
    masked_empty_images = masked_empty_images.reshape(masked_empty_images.shape[0], -1)
    empty_images = empty_images.reshape(empty_images.shape[0], -1)

    # Covariance and SVD
    CovarianceMatrix = np.conjugate(masked_empty_images) @ empty_images.T
    inversecovariancematrix = np.linalg.inv(CovarianceMatrix)
    eigenvalues, eigenvectors = np.linalg.eigh(CovarianceMatrix)
    svd = (eigenvalues, eigenvectors)

    right_svs = eigenvectors.T @ np.conjugate(masked_empty_images)
    left_svs = eigenvectors.T @ empty_images

# ─────────────────────────────────────────────────────────────
# Inpainting Function
# ─────────────────────────────────────────────────────────────

def inpaint1(image1, use_svd=0, remove_gradients=False, 
             exclude_atoms_from_gradient=False, use_Tikhonov=0):
    """
    Inpaint a single image using stored empty image basis.
    """
    global svd, left_svs, right_svs, CovarianceMatrix

    if remove_gradients:
        image1 = RemovePhaseGradient(image1, exclude_atoms_from_gradient=exclude_atoms_from_gradient)

    img_res = image1.reshape(-1)
    eigenvalues, _ = svd

    if use_svd > 0:
        s = eigenvalues**-1
        s[:use_svd] = 0
        a1 = right_svs @ img_res
        inpainted = left_svs.T @ np.diag(s) @ a1.T
    elif use_Tikhonov > 0:
        s = eigenvalues / (eigenvalues**2 + use_Tikhonov * np.mean(eigenvalues**2))
        a1 = right_svs @ img_res
        inpainted = left_svs.T @ np.diag(s) @ a1.T
    else:
        inpainted = empty_images.T @ inversecovariancematrix @ (np.conjugate(masked_empty_images) @ img_res)

    return inpainted.reshape(image1.shape)[mask[0]:mask[1], mask[2]:mask[3]]

def inpaint(images, use_svd=0, remove_gradients=False, 
            exclude_atoms_from_gradient=False, use_Tikhonov=0):
    """
    Inpaint a batch of images using stored empty image basis.
    """
    inpainted_images = np.zeros((images.shape[0], mask[1]-mask[0], mask[3]-mask[2]), dtype=images.dtype)
    for i in range(images.shape[0]):
        inpainted_images[i] = inpaint1(images[i], use_svd, remove_gradients, exclude_atoms_from_gradient, use_Tikhonov)
    return inpainted_images

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

def set_mask(mask_):
    """Set the mask region."""
    global mask
    mask = mask_
    if debuglevel > 0:
        print(f"Mask set to: {mask}")

def set_debug_level(level):
    """Set the debug level (verbosity)."""
    global debuglevel
    debuglevel = level
    if debuglevel > 0:
        print(f"Debug level set to: {debuglevel}")

def set_gradient_removal_orders(order_X, order_Y):
    """Set Chebyshev polynomial orders for background phase correction."""
    global gradientremovalorder_X_default, gradientremovalorder_Y_default
    gradientremovalorder_X_default = order_X
    gradientremovalorder_Y_default = order_Y
    if debuglevel > 0:
        print(f"Gradient removal orders set to: X={order_X}, Y={order_Y}")

