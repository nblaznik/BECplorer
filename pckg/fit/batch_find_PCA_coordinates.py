import os

from PCA_LIST import all_params

for params in all_params:
    date = params[0]
    shot = params[1]
    # Find the masking coordinates, then import the rest. I'm sure there's a better way to do this. Don't change this part.
    if not os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{date}_{str(shot).zfill(4)}.npy"):
        print(f"Masking coordinates not found for {date} - {shot}, running coords_from_image.py to get them.")
        os.system(f'python3 /home/bec_lab/python/BECViewer/pckg/fit/coords_from_image.py {date} {str(shot).zfill(4)} -M PCA')

    else:
        print(f"Masking coordinates found for {date} - {shot}, loading them.")
        

