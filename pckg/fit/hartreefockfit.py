# -*- coding: utf-8 -*-

# Created on Sat Apr 28 17:07:31 2018
# Last modified in Apr 2021
# @author: Jasper
# @edits and corrections: Nejc


# ----------------------------------------------------- IMPORTS ----------------------------------------------------
import numpy as np
import abel
from scipy.interpolate import RectBivariateSpline as rbs
import OAH_functions as f1
from mpmath import fp,mp
import matplotlib.pyplot as plt

# ---------------------------------------------------- CONSTANTS ---------------------------------------------------
hbar  = 1.05E-34
U0    = 1.0E-50
h     = 2.*np.pi*hbar
m     = 3.817E-26
kB    = 1.38E-23
lamb0 = 589.1E-9
kvec  = 2*np.pi/lamb0
num = 1E5
qmax0 = 5

try:
    polylog_arr = np.load("../../../old/python/polylog_arr.npy")
    if polylog_arr.shape[0] != num:
        print("Wrong Shape")
except:
    print("Error while loading g_3/2 array.")
    polylog_arr = np.arange(num,dtype=np.float64)
    for it in np.arange(polylog_arr.shape[0]):
        polylog_arr[it] = fp.polylog(3./2.,polylog_arr[it]/num)
    
    np.save("polylog_arr",polylog_arr)
    polylog_arr = np.load("../../../old/python/polylog_arr.npy")


# ---------------------------------------------------- FUNCTIONS ---------------------------------------------------
def fast_polylog(x):
    ind = int((x-1E-15)*num)
    rem = x*num-ind
    try:
#        return polylog_arr[ind]+(polylog_arr[ind+1]-polylog_arr[ind])*rem
         return polylog_arr[ind]
    except:
        return polylog_arr[-1]

f_polylog = np.vectorize(fast_polylog)

def thomas_fermi(Veff,mu):  ## (r,z)
    tmp = (mu - Veff) / U0
    tmp[tmp<0] = 0.
    return tmp

def thermal(Veff,mu,T): ## (r,z)
    ldB = np.sqrt((2*np.pi*hbar**2)/(m*kB*T))
    tmp = np.exp(-(Veff-mu)/(kB*T))
    return f_polylog(tmp)/ldB**3

# POPOV
def popov_integrand(q,a):
    """
    This is the Popov integrand WITHOUT divergence.
    """
    ratio = 1/np.sqrt(1-(1+q/a)**(-2))
    epsP  = a * np.sqrt((1+q/a)**2-1)
    return np.sqrt(q) * ratio * (np.exp(epsP)-1)**(-1) - 1/(2*np.sqrt(q))

def hf_integrand(q,a):
    return np.sqrt(q)  * (np.exp(q+a)-1)**(-1)

def popov_terms(a,qmax=qmax0):
    """
    These come out unnormalised.
    """
    a = a[:,None]
    q_arr,dq = np.linspace(0,qmax,num=50*qmax,retstep=True)
    q_arr = q_arr[None,:-1]+dq/2.
    
    pop_int = popov_integrand(q_arr,a)
    hf_int  = hf_integrand(q_arr,a)
    
    tmp = (pop_int-hf_int).sum(axis=1)*dq + np.sqrt(qmax)
    
    return tmp

def thermal_p(Veff,mu,T,n0U0): ## (r,z)
    ldB = np.sqrt((2*np.pi*hbar**2)/(m*kB*T))
    
    tmp = thermal(Veff,mu,T)
    
    mask_bec = n0U0 != 0
    
    if mask_bec.sum() != 0:
        tmp[mask_bec] += popov_terms(n0U0[mask_bec]/(kB*T)) * 2/( np.sqrt(np.pi)* ldB**3)
    
    return tmp

def bimodal_profile(omg_r,omg_z,mu,T,iterations=5,output_rz=False,x_size=0,z_size=0,x_points=201,z_points=200):

#    print(mu/h,T*1E9)

    T = np.abs(T)

    if x_size == 0:
        x_size = 5*np.sqrt((2*np.abs(mu))/(m*omg_r**2))
    if z_size == 0:
        z_size = 5*np.sqrt((2*np.abs(mu))/(m*omg_z**2))
       
    r = np.linspace(-x_size,x_size,num=x_points)
    z = np.linspace(-z_size,z_size,num=z_points)
    
    rv,zv = np.meshgrid(r,z,indexing="xy")
    
    Vext = 1./2. * m * (omg_r**2 * rv**2 + omg_z**2 * zv**2)
    
    n_0 = thomas_fermi(Vext,mu)
    n_t = thermal(Vext+2*U0*n_0,mu,T)
    
    for it in range(iterations):
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_0 = thomas_fermi(Veff-2*U0*n_0,mu)
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_t = thermal(Veff,mu,T)

    #plt.imshow(n_0+n_t)
    #plt.show()

    if output_rz == True:
        return rv,zv,n_0+n_t
    else:
        n_c = abel.Transform((n_0+n_t), direction='forward', method='hansenlaw').transform*(r[1]-r[0])
        return rv,zv,n_c

def bimodal_profile_p(omg_r,omg_z,mu,T,iterations=5,hf_it=2,output_rz=False,output_th = False,x_size=0,z_size=0,x_points=101,z_points=100):
    """
    Calculates the bimodal profile in the Popov approximation. Note that setting
    iterations to 0 reduces to the Hartree-Fock case (with hf_it as iteration number).
    """
    T = np.abs(T)

    if x_size == 0:
        x_size = 5*np.sqrt((2*np.abs(mu))/(m*omg_r**2))
    if z_size == 0:
        z_size = 5*np.sqrt((2*np.abs(mu))/(m*omg_z**2))
       
    r = np.linspace(0.,x_size,num=x_points)
    z = np.linspace(0.,z_size,num=z_points)
    
    rv,zv = np.meshgrid(r,z,indexing="xy")
    
    Vext = 1./2. * m * (omg_r**2 * rv**2 + omg_z**2 * zv**2)
    
    n_0 = thomas_fermi(Vext,mu)
    n_t = thermal(Vext+2*U0*n_0,mu,T)
        
    for it in range(hf_it):
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_0 = thomas_fermi(Veff-2*U0*n_0,mu)
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_t = thermal(Veff,mu,T)
    
    for it in range(iterations):
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_0 = thomas_fermi(Veff-2*U0*n_0,mu)
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_t = thermal_p(Veff,mu,T,U0*n_0)


    ## We only calculate one quadrant, this function pads over to the other quadrants.

    tmp_zv = np.zeros((z_points*2-1,x_points*2-1),dtype=float)
    tmp_rv = tmp_zv.copy()

    #print(zv.shape)

    ## First quadrant
    tmp_zv[z_points-1:,x_points-1:] = zv
    tmp_rv[z_points-1:,x_points-1:] = rv

    ## Second quadrant
    tmp_zv[:z_points-1,x_points-1:] = -zv[:0:-1,:]
    tmp_rv[:z_points-1,x_points-1:] = rv[:0:-1,:]

    ## Third quadrant
    tmp_zv[z_points-1:,:x_points-1] = zv[:,:0:-1]
    tmp_rv[z_points-1:,:x_points-1] = -rv[:,:0:-1]

    ## Fourth quadrant
    tmp_zv[:z_points-1,:x_points-1] = -zv[:0:-1,:0:-1]
    tmp_rv[:z_points-1,:x_points-1] = -rv[:0:-1,:0:-1]

    rv = tmp_rv
    zv = tmp_zv


    if output_th:
        tmp_nt = tmp_rv.copy()*0. 
        tmp_n0 = tmp_rv.copy()*0. 
        
        ## First quadrant
        tmp_n0[z_points-1:,x_points-1:] = n_0
        tmp_nt[z_points-1:,x_points-1:] = n_t
        
        ## Second quadrant
        tmp_n0[:z_points-1,x_points-1:] = n_0[:0:-1,:]
        tmp_nt[:z_points-1,x_points-1:] = n_t[:0:-1,:]

        ## Third quadrant
        tmp_n0[z_points-1:,:x_points-1] = n_0[:,:0:-1]
        tmp_nt[z_points-1:,:x_points-1] = n_t[:,:0:-1]

        ## Fourth quadrant
        tmp_n0[:z_points-1,:x_points-1] = n_0[:0:-1,:0:-1]
        tmp_nt[:z_points-1,:x_points-1] = n_t[:0:-1,:0:-1]

        n_0 = tmp_n0
        n_t = tmp_nt
    else:   
        ntot = n_0 + n_t
        tmp_ntot = tmp_rv.copy()*0. 
        ## First quadrant
        tmp_ntot[z_points-1:,x_points-1:] = ntot
        
        ## Second quadrant
        tmp_ntot[:z_points-1,x_points-1:] = ntot[:0:-1,:]

        ## Third quadrant
        tmp_ntot[z_points-1:,:x_points-1] = ntot[:,:0:-1]

        ## Fourth quadrant
        tmp_ntot[:z_points-1,:x_points-1] = ntot[:0:-1,:0:-1]

        ntot = tmp_ntot
   
    #plt.imshow(n_0+n_t)
    #plt.show()

    if output_th == False:
        if output_rz == True:
            return rv,zv,ntot
        else:
            n_c = abel.Transform(ntot, direction='forward', method='hansenlaw').transform*(r[1]-r[0])
            return rv,zv,n_c
    else:
        if output_rz == True:
            return rv,zv,n_0+n_t,n_t
        else:
            n_c = abel.Transform((n_0+n_t), direction='forward', method='hansenlaw').transform*(r[1]-r[0])
            n_tc = abel.Transform((n_t), direction='forward', method='hansenlaw').transform*(r[1]-r[0])
            return rv,zv,n_c,n_tc
    
def bimodal_profile_with_th(omg_r,omg_z,mu,T,iterations=5,output_rz=False,x_size=0,z_size=0,x_points=201,z_points=200):
    if x_size == 0:
        x_size = 5*np.sqrt((2*np.abs(mu))/(m*omg_r**2))
    if z_size == 0:
        z_size = 5*np.sqrt((2*np.abs(mu))/(m*omg_z**2))
       
    r = np.linspace(-x_size,x_size,num=x_points)
    z = np.linspace(-z_size,z_size,num=z_points)
    
    rv,zv = np.meshgrid(r,z,indexing="xy")
    
    Vext = 1./2. * m * (omg_r**2 * rv**2 + omg_z**2 * zv**2)
    
    n_0 = thomas_fermi(Vext,mu)
    n_t = thermal(Vext+2*U0*n_0,mu,T)
    
    for it in range(iterations):
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_0 = thomas_fermi(Veff-2*U0*n_0,mu)
        Veff = Vext + 2*U0 * (n_0+n_t)
        n_t = thermal(Veff,mu,T)

    #plt.pcolormesh(rv,zv,(n_0+n_t))

    if output_rz == True:
        return rv,zv,n_0+n_t,n_t
    else:
        n_c = abel.Transform((n_0+n_t), direction='forward', method='hansenlaw').transform*(r[1]-r[0])
        n_tc = abel.Transform((n_t), direction='forward', method='hansenlaw').transform*(r[1]-r[0])
        return rv,zv,n_c,n_tc

def rotvars(rot,mid,x):
    x0 = np.cos(rot)*(x[0]-mid[0])-np.sin(rot)*(x[1]-mid[1])
    x1 = np.cos(rot)*(x[1]-mid[1])+np.sin(rot)*(x[0]-mid[0])
    return [x0,x1]

def inv_rotvars(rot,mid,x):
    x0 = mid[0] + x[0] * np.cos(rot) + x[1] * np.sin(rot)
    x1 = mid[1] - x[0] * np.sin(rot) + x[1] * np.cos(rot)

    return [x0,x1]

def model_popov(B,x,fx,fz,for_plot=False):
    xrot = rotvars(B[1],[B[2],B[3]],x)
    

    omg_z = 2*np.pi*fz#B[6]
    omg_r = 2*np.pi*fx#B[7]
    mu    = B[4]
    T     = B[5]
    

    # Calculated non-rotated, non-scaled profile
    if for_plot == True:
        rv,zv,col,th_col = bimodal_profile_p(omg_r,omg_z,mu,T,output_th=True)#,x_size=0.2E-3,z_size=0.5E-3)
    else:
        rv,zv,col = bimodal_profile_p(omg_r,omg_z,mu,T)#,x_size=1E-3,z_size=1E-3)

    # Spline interpolate
    mod = rbs(zv[:,0],rv[0],col)

    if for_plot == True:
        mod2 = rbs(zv[:,0],rv[0],th_col)
        return B[0]+mod(xrot[0],xrot[1],grid=False), B[0]+mod2(xrot[0],xrot[1],grid=False)
    else:
        return B[0]+mod(xrot[0],xrot[1],grid=False)

def new_model_omgfit(B,x,fx,fz,for_plot=False):
    xrot = rotvars(B[1],[B[2],B[3]],x)
    

    omg_z = 2*np.pi*fz#B[6]
    omg_r = 2*np.pi*fx#B[7]
    mu    = B[4]
    T     = B[5]
    

    # Calculated non-rotated, non-scaled profile
    if for_plot == True:
        rv,zv,col,th_col = bimodal_profile_with_th(omg_r,omg_z,mu,T)#,x_size=0.2E-3,z_size=0.5E-3)
    else:
        rv,zv,col = bimodal_profile(omg_r,omg_z,mu,T)#,x_size=1E-3,z_size=1E-3)

    # Spline interpolate
    mod = rbs(zv[:,0],rv[0],col)

    if for_plot == True:
        mod2 = rbs(zv[:,0],rv[0],th_col)
        return B[0]+mod(xrot[0],xrot[1],grid=False), B[0]+mod2(xrot[0],xrot[1],grid=False)
    else:
        return B[0]+mod(xrot[0],xrot[1],grid=False)

def model_ll_correct(B,x,fx,fz,det_1,kvec,for_plot=False,output_rz=False):
    xrot = np.array(rotvars(B[1],[B[2],B[3]],x))*B[8]
    

    omg_z = 2*np.pi*fz#B[6]
    omg_r = 2*np.pi*fx#B[7]
    mu    = B[4]
    T     = B[5]
    
    ## Get the density from the previous fit function
    rv,zv,col,th_col = bimodal_profile_with_th(omg_r,omg_z,mu,T,output_rz=True)

    ## Calculate the Lorenz-Lorentz correction, only use real part
    polar2 = f1.polarizability(det_1,0,589.1E-9)
    nHF  = kvec*( np.sqrt(1+polar2*col/(1-polar2*col/3.)).real-1 )
    nHFt = kvec*( polar2*col/2 )

    if output_rz == False:
        nHF = abel.Transform(nHF, direction='forward', method='hansenlaw').transform*(rv[0,1]-rv[0,0])
        nHFt = abel.Transform(nHFt, direction='forward', method='hansenlaw').transform*(rv[0,1]-rv[0,0])

    # Spline interpolate
    mod = rbs(zv[:,0],rv[0],nHF)

    if for_plot == True:
        mod2 = rbs(zv[:,0],rv[0],nHFt)
        return B[0]+mod(xrot[0],xrot[1],grid=False), B[0]+mod2(xrot[0],xrot[1],grid=False)
    else:
        return B[0]+mod(xrot[0],xrot[1],grid=False)

def new_model_scale(B,x,fx,fz,for_plot=False):
    xrot = rotvars(B[1],[B[2],B[3]],x)#*B[8]
    
    omg_z = 2*np.pi*fz#B[6]
    omg_r = 2*np.pi*fx#B[7]
    mu    = B[4]
    T     = B[5]

    # Calculated non-rotated, non-scaled profile
    if for_plot == True:
        rv,zv,col,th_col = bimodal_profile_with_th(omg_r,omg_z,mu,T,output_rz=True)#,x_size=0.2E-3,z_size=0.5E-3)
    else:
        rv,zv,col = bimodal_profile(omg_r,omg_z,mu,T)#,x_size=1E-3,z_size=1E-3)

    # Spline interpolate
    mod = rbs(zv[:,0],rv[0],col)

    if for_plot == True:
        mod2 = rbs(zv[:,0],rv[0],th_col)
        return B[0]+mod(xrot[0]*B[8],xrot[1]*B[8],grid=False), B[0]+mod2(xrot[0]*B[8],xrot[1]*B[8],grid=False)
    else:
        return B[0]+mod(xrot[0]*B[8],xrot[1]*B[8],grid=False)
