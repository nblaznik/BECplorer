#Imports
import numpy as np
from skimage.restoration import unwrap_phase
# Constants

#pChebyshev polynomial orders for the background gradient correction.
gradientremovalorder_X_default=2
gradientremovalorder_Y_default=2

empty_images=[]
masked_empty_images=[]
CovarianceMatrix=[]
inversecovariancematrix=[]
svd=[]
wtop,wbottom,wleft,wright=25,65,1000,2000
mask=(wtop,wbottom,wleft,wright)
debuglevel=9
left_svs=[]
right_svs=[]

#####

def RemovePhaseGradient(image, gradientremovalorder_X=gradientremovalorder_X_default, gradientremovalorder_Y=gradientremovalorder_Y_default,exclude_atoms_from_gradient=False ):
  # Remove phase gradients from an image by Chebyshev fit to the unwrapped phase.
  (xdim_,ydim_)=image.shape
  image_unwrapped = unwrap_phase(np.angle(image))


  # Create a grid of points
  X_axis =np.linspace(-1,1,xdim_)
  Y_axis =np.linspace(-1,1,ydim_)


  bitmask=np.abs(image)**2
  if exclude_atoms_from_gradient:
      bitmask[mask[0]:mask[1], mask[2]:mask[3]]=0
  flat1=np.sum(image_unwrapped*bitmask,axis=0)/np.sum(bitmask,axis=0)
  flat2=np.sum(image_unwrapped*bitmask,axis=1)/np.sum(bitmask,axis=1)

  #else:
  #  flat1=np.mean(image_unwrapped,axis=0)
  #  flat2=np.mean(image_unwrapped,axis=1)
  #end if   
 
  fit1=np.polynomial.chebyshev.chebfit(Y_axis, flat1, gradientremovalorder_Y)
  fit2=np.polynomial.chebyshev.chebfit(X_axis, flat2, gradientremovalorder_X)
  fit2[0]=0
  fiteval=np.polynomial.chebyshev.chebval(Y_axis, fit1)+np.outer(np.ones_like(Y_axis),np.polynomial.chebyshev.chebval(X_axis, fit2)).T

  return np.exp(-1.0j*fiteval)*image
#end def RemovePhaseGradient

def set_empty_images(imagelist,remove_gradients=False,exclude_atoms_from_gradient=False):
    # Set the empty images to the provided image.
    global empty_images
    global masked_empty_images
    empty_images = imagelist.copy()
    if remove_gradients:
        for i in range (empty_images.shape[0]):
           empty_images[i] =RemovePhaseGradient(empty_images[i],exclude_atoms_from_gradient=exclude_atoms_from_gradient)
    masked_empty_images = empty_images.copy()
    print(f"Setting empty images with shape: {empty_images.shape}")
    masked_empty_images[:,mask[0]:mask[1], mask[2]:mask[3]] = 0.0
    masked_empty_images=masked_empty_images.reshape(masked_empty_images.shape[0],-1)
    empty_images=empty_images.reshape(empty_images.shape[0],-1)
    global CovarianceMatrix
    CovarianceMatrix=np.matrix(np.conjugate(masked_empty_images) @ empty_images.T)
    global inversecovariancematrix
    inversecovariancematrix = np.linalg.inv(CovarianceMatrix)
    global svd
    global left_svs
    global right_svs
    svd = np.linalg.eigh(CovarianceMatrix)
    right_svs= svd.eigenvectors.H @ np.conjugate(masked_empty_images)
    left_svs= (svd.eigenvectors).T @ empty_images

def inpaint1(image1, use_svd=0,remove_gradients=False,exclude_atoms_from_gradient=False,use_Tikhonov=0):
  # Inpaint the provided images using the empty images and the mask.
  global svd
  global left_svs
  global right_svs
  global CovarianceMatrix
  
  if remove_gradients:
     image1 = RemovePhaseGradient(image1,exclude_atoms_from_gradient=exclude_atoms_from_gradient)
  img_res=image1.reshape(-1)
  if use_svd>0:
    # Use SVD for inpainting
    ##CovarianceMatrix=np.matrix(np.conjugate(masked_empty_images) @ empty_images.T)
    s=svd.eigenvalues**-1
    s[:use_svd]=0
    a1=(right_svs @ img_res)
    inpainted = left_svs.T @ np.diag((s)) @ a1.T
  elif use_Tikhonov>0 :
    s=svd.eigenvalues/(svd.eigenvalues**2+use_Tikhonov*np.mean(svd.eigenvalues**2))
    a1=(right_svs @ img_res)
    inpainted = left_svs.T @ np.diag((s)) @ a1.T
  else:
    inpainted= empty_images.T  @ (inversecovariancematrix)@ (np.conjugate(masked_empty_images) @ img_res )
  #end if
  inpainted=inpainted.reshape(image1.shape)
  return inpainted[mask[0]:mask[1], mask[2]:mask[3]]
#end def inpaint

def inpaint(images, use_svd=0,remove_gradients=False,exclude_atoms_from_gradient=False,use_Tikhonov=0):
  inpainted_images = np.zeros((images.shape[0], mask[1]-mask[0],mask[3]-mask[2]), dtype=images.dtype)
  for i in range(images.shape[0]):
    inpainted_images[i] = inpaint1(images[i], use_svd, remove_gradients, exclude_atoms_from_gradient, use_Tikhonov) 
  return inpainted_images  

def set_mask(mask_):
  # Set the mask to the provided mask.
  global mask
  mask = mask_  
  if debuglevel > 0:
    print(f"Mask set to: {mask}")
#end def set_mask

def set_debug_level(level):
  # Set the debug level.
  global debuglevel
  debuglevel = level
  if debuglevel > 0:
    print(f"Debug level set to: {debuglevel}")      
#end def set_debug_level

def set_gradient_removal_orders(order_X, order_Y):
  # Set the Chebyshev polynomial orders for background gradient correction.
  global gradientremovalorder_X_default
  global gradientremovalorder_Y_default
  gradientremovalorder_X_default = order_X
  gradientremovalorder_Y_default = order_Y
  if debuglevel > 0:
    print(f"Gradient removal orders set to: X={gradientremovalorder_X_default}, Y={gradientremovalorder_Y_default}")        
#end def set_gradient_removal_orders
