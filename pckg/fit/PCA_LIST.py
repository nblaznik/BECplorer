## JUST RUN PCA from terminal - it will run the list from here.


# Define the list that will be used in: 
# 1.) batch_find_PCA_coordinates.py
# 2.) prepareForPCA.py
# 3.) process_PCA.py
# Use as: "from PCA_LIST import all_params" 


# get last number

import os 
date = 20250819

last_run = int(sorted([x for x in os.listdir(f"/storage/data/{date}/")])[-1])
print(f"Last run: {last_run}")


all_params = [
    [date, i] for i in range(61,  last_run+1) # when running all 
]


import os 
import csv 

def get_parameter(date, seq, paramname):
    """
    Get the value of the parameter form the parameters.param file. Very often used, might be better to import
    it from the pcgk/fits file. But it might be better to include those functions here.
    """
    date = str(date)
    run_id = str(seq).zfill(4)
    path = "/storage/data/"
    param = "N/A"
#     try:
    with open(path + date + '/' + run_id + '/parameters.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        for row in csvreader:
            if row[0] == paramname:
                param = float(row[1])
#     except:
#         param = "N/A"
    return param

if __name__ == "__main__":
    print("Running PCA on the following parameters:")
    print("|   Date   |   Shot   | Coordinates Found | Pre-Processed | PCA Run | Nr Frames |")
    print("|:--------:|:--------:|:-----------------:|:-------------:|:-------:|:---------:|")
    for params in all_params:
        ## Make a little table summarizing if the cooridnates are found or not, whether it has been pre-processed or not, and whether the PCA has been run or not.
        date, shot = params
        if os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/cut_arrs/{date}_{str(shot).zfill(4)}.npy"):
            coords_found = u'\u2713'
        else:
            coords_found = " "

        if os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/OAH_processing/full_complex_field_{date}_{shot}_atoms.npy"):
            pre_processed = u'\u2713'
        else:
            pre_processed = " "

        if os.path.exists(f"/home/bec_lab/Desktop/imgs/SOAH/PCA_Analysis/data/PCA_processing/{date}_{shot}/0000.npy"):
            pca_run = u'\u2713'
        else:
            pca_run = " "

        nr_atoms = int(get_parameter(date, shot, "ATOM_IMAGES"))
        print(f"|{date:^10}|{shot:^10}|{coords_found:^19}|{pre_processed:^15}|{pca_run:^9}|{nr_atoms:^11}|")
    input("Press Enter to run missing, or Ctrl+C to abort.")  # Wait for user input before exiting
