# ------------------------------------------------------------------------------------------------------------------
# BEC LAB FITTING FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------

# Fitting functions used throughout the quick and slow analysis used in the BEC lab.
# At some point we should implement Sam's fitting functions to fit using physical parameters rather than
# simply reshaping the

# Updated on 18-04-2021 by Nejc Blaznik


# ---------------------------------------------------- IMPORTS -------------------------------------------------------
import numpy as np
from scipy.special import eval_legendre


# --------------------------------------------------- FUNCTIONS ------------------------------------------------------

def gauss1D(x, amp, sig, mu):
    """ Simple 1D Gaussian function, taking usual arguments. """
    return np.abs(amp) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def gauss2D(x, amp, sig, mu):
    """ Simple 2D Gaussian function, taking usual arguments, now double per variable. """
    return gauss1D(x[0], amp, sig[0], mu[0]) * gauss1D(x[1], 1., sig[1], mu[1])


def thomasfermi(x, amp, r, mid):
    """ A 2D Thomas Fermi function, a function of amplitude, radia and means (both x and y). """
    value = 1 - ((x[0] - mid[0]) / r[0]) ** 2 - ((x[1] - mid[1]) / r[1]) ** 2
    value[value < 0] = 0
    return np.abs(amp) * np.power(value, 3. / 2.)


def rotvars(rot, mid, x):
    """ The function rotating the variables by angle 'rot' around 'mid'. """
    x0 = np.cos(rot) * (x[0] - mid[0]) - np.sin(rot) * (x[1] - mid[1])
    x1 = np.cos(rot) * (x[1] - mid[1]) + np.sin(rot) * (x[0] - mid[0])
    return [x0, x1]


def gaussmod(B, x):
    """
    Modify the 2D Gaussian function - take the exponential of minus the Gaussian, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the thermal cloud.
    B array:
        B[0] = initial 'level' of the Gaussian, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = Gaussian parameters (amp, sig_x, sig_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return 0 * B[0] + B[1] * np.exp(-gauss2D(xrot, B[5], [B[6], B[7]], [0., 0.]))  # B[1]*


def tfmod(B, x):
    """
    Modify the 2D Thomas-Fermi function - take the exponential of minus the TF, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the condensate.
        B[0] = initial 'level' of the TF, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return 0. * B[0] + B[1] * np.exp(-thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]))


def bimodalmod(B, x):
    """
    Combine the Gaussian and TF model into a single bimodal function used to fit and model the distribution
    of the atoms in a superfluid - thermal cloud + the condensate. The variables for both of the fits are contained
    within the B array.
        B[0] = initial 'level' of the bimodal model, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
        B[8, 9, 10] = Gaussian parameters (amp, sig_x, sig_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + B[1] * np.exp(-(thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]) + gauss2D(xrot, B[8], [B[9], B[10]], [0., 0.])))



def gaussmod_OAH(B, x):
    """
    Modify the 2D Gaussian function - take the exponential of minus the Gaussian, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the thermal cloud.
    B array:
        B[0] = initial 'level' of the Gaussian, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = Gaussian parameters (amp, sig_x, sig_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + B[1] * gauss2D(xrot, B[5], [B[6], B[7]], [0., 0.])  # B[1]*


def tfmod_OAH(B, x):
    """
    Modify the 2D Thomas-Fermi function - take the exponential of minus the TF, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the condensate.
        B[0] = initial 'level' of the TF, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + B[1] * thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.])


def bimodalmod_OAH(B, x):
    """
    Combine the Gaussian and TF model into a single bimodal function used to fit and model the distribution
    of the atoms in a superfluid - thermal cloud + the condensate. The variables for both of the fits are contained
    within the B array.
        B[0] = initial 'level' of the bimodal model, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
        B[8, 9, 10] = Gaussian parameters (amp, sig_x, sig_y, mu_x = mu_y = 0.)
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    return B[0] + B[1] * (thomasfermi(xrot, B[5], [B[6], B[7]], [0., 0.]) + gauss2D(xrot, B[8], [B[9], B[10]], [0., 0.]))






def L(z, j):
    """    Difference between the j+2'th and the j'th Legendre polynomials    """
    legendre_sum = eval_legendre(j+2, z) - eval_legendre(j, z)
    legendre_sum *= np.piecewise(z, [abs(z) <=1, abs(z) > 1], [1, 0])

    return legendre_sum


def thomasFermiAxialMode(x, amp, r, mid, j, axamp, x_shift, sq):
    """
        A 2D Thomas Fermi function, a function of amplitude, radia and means (both x and y).
        Uf what, this function for some reason seems pretty good.
    """
    value = 1 - ((x[0] - mid[0]) / r[0]) ** 2 - ((x[1] - mid[1]) / r[1]) ** 2
    value[value < 0] = 0
    return (np.abs(amp) * np.power(value, 3. / 2.) + axamp*L((sq*x[1] + x_shift)/r[1], int(j))) * np.abs(amp) * np.power(value, 3. / 2.)


def tfaxialmod(B, x):
    """
    Modify the 2D Thomas-Fermi function - take the exponential of minus the TF, rotate if necessary and
    translate the function using the variables defined in the B array. This function is used to fit and model
    the distribution of the atoms in the condensate.
        B[0] = initial 'level' of the TF, here multiplied by 0 (no need for it thus far)
        B[1] = overall amplitude of the exponential
        B[2] = angle of rotation
        B[3, 4] = point around which we rotate
        B[5, 6, 7] = TF parameters (amp, r_x, r_y, mu_x = mu_y = 0.)
        B[8] = j parameter
    """
    xrot = rotvars(B[2], [B[3], B[4]], x)
    # print("Fitting for {:}".format(B))
    return 0. * B[0] + B[1] * np.exp(-thomasFermiAxialMode(xrot, B[5], [B[6], B[7]], [0., 0.], B[8], B[9], B[10], B[11]))


