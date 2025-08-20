# ─────────────────────────────────────────────────────────────
dz_focus = 0.0055
from PCA_LIST import all_params

for params in all_params:
    import os
    import numpy as np
    import astropy.io.fits as pyfits
    import importlib

    if os.path.exists(f'/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/PCA_processing/{params[0]}_{params[1]}/0000.npy'):
        print(f"Shot {params[0]} - {params[1]} already processed, skipping...")
        continue    
    else:   
        date = params[0]
        shot = params[1]
        print(date, shot)

        # ─────────────────────────────────────────────────────────────
        # Import Core Libraries
        # ─────────────────────────────────────────────────────────────

        import inpaintingfunction as inpaint

        # Find the masking coordinates, then import the rest. I'm sure there's a better way to do this. Don't change this part.

        if not os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{date}_{str(shot).zfill(4)}.npy"):
            print(f"Masking coordinates not found for {date} - {shot}, running coords_from_image.py to get them.")
            os.system(f'python3 /home/bec_lab/python/BECViewer/pckg/fit/coords_from_image.py {date} {str(shot).zfill(4)} -M PCA')

        else:
            print(f"Masking coordinates found for {date} - {shot}, loading them.")
            
        mask = np.load(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{date}_{str(shot).zfill(4)}.npy", allow_pickle=True)
        np.save(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/CUT_COORDS.npy", mask, allow_pickle=True)

        print("mask", mask)

        importlib.reload(inpaint)


        # from OAHDEV_functions import *
        # from OAH_refocus import *

        print("inpaint values", inpaint.wtop, inpaint.wbottom, inpaint.wleft, inpaint.wright)

        # ─────────────────────────────────────────────────────────────
        # Constants
        # ─────────────────────────────────────────────────────────────
        kB     = 1.38064852e-23  # Boltzmann constant
        k_B    = kB              # Alias for compatibility
        m      = 3.81923979e-26  # Sodium-23 atom mass
        m_na   = m               # Alias
        hb     = 1.0545718e-34   # Reduced Planck constant
        asc    = 2.802642e-9     # Scattering length
        mu0    = 1e-50           # (placeholder magnetic permeability?)
        e0     = 8.854187e-12    # Vacuum permittivity
        pix_size = 6.5e-6 / 2.63 # Pixel size in meters

        # Light properties
        lamb0 = 589.1e-9               # Wavelength of sodium D-line
        k0    = 2 * np.pi / lamb0      # Wave number



        def get_nr_of_atoms(date, shot):
            fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/0.fits'
            nr = len(pyfits.open(fits_path)[0].data.astype(float))
            return nr
        def get_nr_of_flats(date, shot):
            fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/1.fits'
            nr = len(pyfits.open(fits_path)[0].data.astype(float))
            return nr


        load_folder = '/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/OAH_processing/'
        save_folder = '/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/PCA_processing/'

        flat_nr = get_nr_of_flats(date, shot)
        if flat_nr < 2:
            print(f"Shot {date} - {shot} has only {flat_nr} flats, skipping...")
            exit()

        print("Number of flat images:", flat_nr)

        if os.path.exists(f"{save_folder}/{date}_{shot}/0000.npy"):
            print(f"Shot {date} - {shot} already processed, skipping...")

        else:
            if not os.path.exists(load_folder+f"full_complex_field_{date}_{shot}_atoms.npy"):
                print(f"Shot {date} - {shot} not processed, please run prepareForPCA.py first.")
                exit()

            pre_pro = True
            if pre_pro: 

                # ─────────────────────────────────────────────────────────────
                # Cropping Parameters for Loaded Images
                # ─────────────────────────────────────────────────────────────
                cropTop    = 5
                cropBottom = -5
                cropLeft   = 50
                cropRight  = -50

                # ─────────────────────────────────────────────────────────────
                # Chebyshev Polynomial Orders for Background Gradient Correction
                # ─────────────────────────────────────────────────────────────
                gradientremovalorder_X = 4
                gradientremovalorder_Y = 2

                # ─────────────────────────────────────────────────────────────
                # Image Selection Parameters
                # ─────────────────────────────────────────────────────────────
                number_of_images          = int(flat_nr * 0.75)             # Use 75% to speed up processing
                number_validation_images  = flat_nr - number_of_images      # Remaining images used for validation

                # ─────────────────────────────────────────────────────────────
                # File Paths
                # ─────────────────────────────────────────────────────────────
                file_blank = f'full_complex_field_{date}_{shot}_flat.npy'
                file_atoms = f'full_complex_field_{date}_{shot}_atoms.npy'

                # ─────────────────────────────────────────────────────────────
                # Load Data
                # ─────────────────────────────────────────────────────────────
                full_blank_images = np.load(load_folder + file_blank, allow_pickle=True)
                full_complex_images_atoms = np.load(load_folder + file_atoms)

                # ─────────────────────────────────────────────────────────────
                # Crop to Region of Interest
                # ─────────────────────────────────────────────────────────────
                blank_validation_images = full_blank_images[-number_validation_images:, :, cropTop:cropBottom, cropLeft:cropRight]
                blank_images = full_blank_images[:number_of_images, :, cropTop:cropBottom, cropLeft:cropRight]
                complex_images_atoms = full_complex_images_atoms[:, :, cropTop:cropBottom, cropLeft:cropRight]

                print(complex_images_atoms.shape)

                # ─────────────────────────────────────────────────────────────
                # Image Dimensions
                # ─────────────────────────────────────────────────────────────
                nimages, _, xdim, ydim = blank_images.shape


                inpaint.debuglevel=0
                inpaint.set_empty_images(blank_images[:,0,:,:],remove_gradients=True, exclude_atoms_from_gradient=False)

                # ─────────────────────────────────────────────────────────────
                # Inpaint a validation image
                # ─────────────────────────────────────────────────────────────
                inpainted = inpaint.inpaint1(
                    blank_validation_images[0, 0, :, :],
                    use_svd=0,
                    use_Tikhonov=0,
                    remove_gradients=True,
                    exclude_atoms_from_gradient=False
                )

                # ─────────────────────────────────────────────────────────────
                # Reference image with only gradient removed
                # ─────────────────────────────────────────────────────────────
                true_image = inpaint.RemovePhaseGradient(
                    blank_validation_images[0, 0],
                    exclude_atoms_from_gradient=False
                )[inpaint.wtop:inpaint.wbottom, inpaint.wleft:inpaint.wright]

                # ─────────────────────────────────────────────────────────────
                # Compute difference metric (scaled phase RMS error)
                # ─────────────────────────────────────────────────────────────
                diff_meas = 1000 * np.linalg.norm(
                    np.angle(inpainted.flatten()) - np.angle(true_image.flatten())
                ) / (true_image.size**0.5)

                print(f"Quality: {diff_meas:.3f}")

            
            os.makedirs(f"{save_folder}/{date}_{shot}/", exist_ok=True)
            nr_of_shots = get_nr_of_atoms(date, shot)
            all_images = [] 

            # Main processing loop
            for t in range(nr_of_shots):
                if os.path.exists(f"{save_folder}/{date}_{shot}/{str(t).zfill(4)}.npy"):
                    print(f"Shot {date} - {shot}, time step {t} already processed, skipping...")
                    continue
                print(f"Processing time step {t+1}/{nr_of_shots}...")
                print("Mask shape:", inpaint.wtop, inpaint.wbottom, inpaint.wleft, inpaint.wright)
                diffs_both = []
                # Instead of calling set_empty_images again:

                for i in range(2):
                    inpaint.set_empty_images(blank_images[:,i,:,:],remove_gradients=True, exclude_atoms_from_gradient=False)
                    image = complex_images_atoms[t, i, :, :]
                    # Run inpainting
                    test = inpaint.inpaint(complex_images_atoms[t:t+2, i, :, :], use_svd=0, use_Tikhonov=0, remove_gradients=True, exclude_atoms_from_gradient=False)

                    # Prepare images for display
                    inpainted_phase = np.angle(test[0])
                    original_cropped = inpaint.RemovePhaseGradient(image)[inpaint.wtop:inpaint.wbottom, inpaint.wleft:inpaint.wright]
                    original_phase = np.angle(original_cropped)
                    difference = original_phase - inpainted_phase    
                    diffs_both.append(difference)

                np.save(f"{save_folder}/{date}_{shot}/{str(t).zfill(4)}", diffs_both)
                all_images.append(np.array(diffs_both))
