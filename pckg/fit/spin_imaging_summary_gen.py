## This code creates a summary of the spin imaging data
import os 
import csv
import astropy.io.fits as pyfits
import argparse
import numpy as np
import sys 

# -------------------------------------------------  ARGUMENT PARSER -------------------------------------------------
# This allows for the command line usage.
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('date', help='Date at which the shot to analyze was made.', metavar='YYYYMMDD')
parser.add_argument('shot', help='Number of the shot in question.', metavar='XXXX')

# Parse arguments
args, unknown = parser.parse_known_args(sys.argv[1:])


# Parameter runs

all_params = [
    [20250418, 73, 0.0055, 50, 'same_flat'],    # 0
    [20250418, 80, 0.0055, 10, 'same_flat'],    # 1
    [20250418, 83, 0.0055, 10, 'same_flat'],    # 2
    [20250418, 88, 0.0055, 10, 'same_flat'],    # 3
    [20250418, 94, 0.0055, 25, 'diff_flat'],    # 4
    [20250418, 99, 0.0055, 25, 'diff_flat'],    # 5
    [20250418, 102, 0.0055, 25, 'diff_flat'],   # 6
    [20250418, 103, 0.0055, 25, 'diff_flat'],   # 7
    [20250418, 104, 0.0055, 25, 'diff_flat'],   # 8
    [20250418, 105, 0.0055, 25, 'diff_flat'],   # 9
    [20250418, 106, 0.0055, 25, 'diff_flat'],   # 10
    [20250418, 107, 0.0055, 25, 'diff_flat'],   # 11
    [20250418, 108, 0.0055, 25, 'diff_flat'],   # 12
    [20250418, 109, 0.0055, 25, 'diff_flat'],   # 13
    [20250418, 110, 0.0055, 25, 'diff_flat'],   # 14
    [20250418, 111, 0.0055, 25, 'diff_flat'],   # 15
    [20250418, 112, 0.0055, 25, 'diff_flat'],   # 16
    [20250418, 113, 0.0055, 25, 'diff_flat'],   # 17
    [20250418, 114, 0.0055, 25, 'diff_flat'],   # 18
    [20250418, 115, 0.0055, 25, 'diff_flat'],   # 19
    [20250422, 16,  0.0055, 25, 'diff_flat'],   # 20
    [20250422, 18,  0.0055, 25, 'diff_flat'],   # 21
    [20250422, 20,  0.0055, 50, 'diff_flat'],   # 22
    [20250422, 21,  0.0055, 50, 'diff_flat'],   # 23
    [20250422, 21,  0.0055, 50, 'diff_flat'],   # 24 (duplicate)
    [20250422, 22,  0.0055, 50, 'diff_flat'],   # 25
    [20250422, 24,  0.0055, 50, 'diff_flat'],   # 26
    [20250422, 25,  0.0055, 50, 'diff_flat'],   # 27
    [20250422, 26,  0.0055, 50, 'diff_flat'],   # 28
    [20250422, 27,  0.0055, 50, 'diff_flat'],   # 29
    [20250422, 28,  0.0055, 50, 'diff_flat'],   # 30
    [20250422, 29,  0.0055, 50, 'diff_flat'],   # 31
    [20250422, 30,  0.0055, 50, 'diff_flat'],   # 32
    [20250422, 31,  0.0055, 50, 'diff_flat'],   # 33
    [20250422, 32,  0.0055, 50, 'diff_flat'],   # 34
    [20250422, 33,  0.0055, 50, 'diff_flat'],   # 35
    [20250422, 65,  0.0055, 25, 'diff_flat'],   # 36
    [20250422, 67,  0.0055, 25, 'diff_flat'],   # 37
    [20250423, 29,  0.0055, 15, 'diff_flat'],   # 38
    [20250423, 43,  0.0055, 25, 'diff_flat'],   # 39
    [20250423, 46,  0.0055, 50, 'diff_flat'],   # 40
    [20250423, 49,  0.0055, 50, 'diff_flat'],   # 41
    [20250423, 50,  0.0055, 50, 'diff_flat'],   # 42
    [20250423, 51,  0.0055, 50, 'diff_flat'],   # 43
    [20250423, 56,  0.0055, 50, 'diff_flat'],   # 44
    [20250423, 64,  0.0055, 50, 'diff_flat'],   # 45
    [20250423, 67,  0.0055, 50, 'diff_flat'],   # 46
    [20250423, 69,  0.0055, 50, 'diff_flat'],   # 47
    [20250423, 70,  0.0055, 50, 'diff_flat'],   # 48
    [20250423, 71,  0.0055, 100, 'diff_flat'],  # 49
    [20250423, 72,  0.0055, 100, 'diff_flat'],  # 50
    [20250423, 73,  0.0055, 100, 'diff_flat'],  # 51
    [20250423, 74,  0.0055, 100, 'diff_flat'],  # 52
    [20250423, 75,  0.0055, 100, 'diff_flat'],  # 53
    [20250423, 76,  0.0055, 100, 'diff_flat'],  # 54
    [20250423, 77,  0.0055, 100, 'diff_flat'],  # 55
    [20250423, 78,  0.0055, 100, 'diff_flat'],  # 56
    [20250423, 79,  0.0055, 100, 'diff_flat'],  # 57
    [20250423, 81,  0.0055, 100, 'diff_flat'],  # 58
    [20250423, 82,  0.0055, 100, 'diff_flat'],  # 59
    [20250424, 21,  0.0055, 100, 'diff_flat'],  # 60  # 0.05 V SG
    [20250424, 24,  0.0055, 100, 'diff_flat'],  # 61  # 0.01 V SG 
    [20250424, 37,  0.0055, 100, 'diff_flat'],  # 62  # 0.01 V SG 
    [20250424, 38,  0.0055, 100, 'diff_flat'],  # 63  # 0.01 V SG 
    [20250424, 40,  0.0055, 25, 'diff_flat'],   # 64  
    [20250424, 41,  0.0055, 25, 'diff_flat'],   # 65   
    [20250424, 42,  0.0055, 25, 'diff_flat'],   # 66   
    [20250424, 43,  0.0055, 25, 'diff_flat'],   # 67   
    [20250424, 44,  0.0055, 25, 'diff_flat'],   # 68    
    [20250424, 45,  0.0055, 100, 'diff_flat'],  # 69   
    [20250424, 46,  0.0055, 100, 'diff_flat'],  # 70   
    [20250424, 47,  0.0055, 100, 'diff_flat'],  # 71  
    [20250424, 48,  0.0055, 100, 'diff_flat'],  # 72   
    [20250424, 51,  0.0055, 100, 'diff_flat'],  # 73   
    [20250424, 54,  0.0055, 100, 'diff_flat'],  # 74   
    [20250424, 56,  0.0055, 100, 'diff_flat'],  # 75   
    [20250424, 84,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 85,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 88,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 90,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 95,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 96,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 99,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 102,  0.0055, 10, 'diff_flat'],   # 75   
    [20250424, 104,  0.0055, 50, 'diff_flat'],   # 75   
    [20250424, 110,  0.0055, 50, 'diff_flat'],   # 75 
    [20250425, 19,  0.0055, 50, 'diff_flat'],   # 85   
    [20250425, 21,  0.0055, 50, 'diff_flat'],   # 85   
    [20250425, 22,  0.0055, 50, 'diff_flat'],   # 85   
    [20250425, 23,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 24,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 25,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 26,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 31,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 33,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 34,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 35,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 41,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 42,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 43,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 44,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 45,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 51,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 52,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 53,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 54,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 55,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 57,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 58,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 59,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 60,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 61,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 62,  0.0055, 50, 'diff_flat'],   # 85    
    [20250425, 63,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 10,  0.0055,  5, 'diff_flat'],   # 85    
    [20250428, 13,  0.0055, 25, 'diff_flat'],   # 85    
    [20250428, 15,  0.0055, 25, 'diff_flat'],   # 85    
    [20250428, 16,  0.0055, 25, 'diff_flat'],   # 85    
    [20250428, 17,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 18,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 19,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 20,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 21,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 22,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 23,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 24,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 25,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 26,  0.0055, 50, 'diff_flat'],   # 85    
    [20250428, 31,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 32,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 33,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 34,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 35,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 36,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 37,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 38,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 39,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 40,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 41,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 42,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428, 43,  0.0055, 200, 'diff_flat'],   # 85    
    [20250428,  44, 0.0055, 200, 'diff_flat'], 
    [20250428,  45, 0.0055, 200, 'diff_flat'],   # images to here 
    [20250428,  46, 0.0055, 200, 'diff_flat'], 
    [20250428,  47, 0.0055, 200, 'diff_flat'], 
    [20250428,  48, 0.0055, 200, 'diff_flat'], 
    [20250428,  49, 0.0055, 200, 'diff_flat'], 
    [20250428,  50, 0.0055, 200, 'diff_flat'], 
    [20250428,  51, 0.0055, 200, 'diff_flat'], 
    [20250428,  52, 0.0055, 200, 'diff_flat'], 
    [20250428,  53, 0.0055, 200, 'diff_flat'], 
    [20250428,  54, 0.0055, 200, 'diff_flat'], 
    [20250428,  55, 0.0055, 200, 'diff_flat'], 
    [20250428,  56, 0.0055, 200, 'diff_flat'], 
    [20250428,  57, 0.0055, 200, 'diff_flat'], 
    [20250428,  58, 0.0055, 200, 'diff_flat'], 
    [20250428,  59, 0.0055,  20, 'diff_flat'], 
    [20250428,  60, 0.0055,  20, 'diff_flat'], 
    [20250428,  61, 0.0055,  20, 'diff_flat'], 
    [20250428,  62, 0.0055,  20, 'diff_flat'], 
    [20250428,  63, 0.0055,  20, 'diff_flat'], 
    [20250428,  64, 0.0055,  20, 'diff_flat'], 
    [20250428,  65, 0.0055,  20, 'diff_flat'], 
    [20250428,  66, 0.0055, 200, 'diff_flat'], 
    [20250428,  67, 0.0055, 200, 'diff_flat'], 
    [20250428,  68, 0.0055, 200, 'diff_flat'], 
    [20250428,  69, 0.0055, 200, 'diff_flat'], 
    [20250428,  70, 0.0055, 200, 'diff_flat'], 
    [20250428,  71, 0.0055, 200, 'diff_flat'], 
    [20250428,  72, 0.0055, 200, 'diff_flat'], 
    [20250428,  73, 0.0055, 200, 'diff_flat'], 
    [20250428,  74, 0.0055, 200, 'diff_flat'], 
    [20250428,  75, 0.0055, 200, 'diff_flat'], 
    [20250428,  76, 0.0055, 200, 'diff_flat'], 
    [20250428,  77, 0.0055, 200, 'diff_flat'], 
    [20250428,  78, 0.0055, 200, 'diff_flat'], 
    [20250428,  79, 0.0055, 200, 'diff_flat'], 
    [20250428,  80, 0.0055, 200, 'diff_flat'], 
    [20250428,  81, 0.0055, 200, 'diff_flat'], 
    [20250428,  82, 0.0055, 200, 'diff_flat'], 
    [20250428,  83, 0.0055, 200, 'diff_flat'], 
    [20250428,  84, 0.0055, 200, 'diff_flat'],         
]


# Get comments 



# Get Spinflip Parameters
def getSpinflipParams(date, seq):
    """ Get additional information about previous analysis.
        I thought it might be useful, but it's not really.
    """
    date = int(date)
    seq = int(seq)

    def find_closest_date(date):
        all_sf_dates = [int(a) for a in sorted(os.listdir("/storage/spinflip_log"), reverse=True)]
        all_dates = [d for d in all_sf_dates if d <= date]
        if len(all_dates) != 0:
            closest_date = max(all_dates)
            return closest_date
        else:
            return None

    def find_closest_run(run, date):
        closest_date = find_closest_date(date)

        if closest_date == None:
            return [None, None]

        elif date != closest_date:
            # If not the same date, then find the last one from the closest date
            all_sf_runs = []
            with open(f"/storage/spinflip_log/{closest_date}") as file:
                for specs in file.read().split(f"{'-' * 71}\n\n\n")[:-1]:
                    all_sf_runs.append(int(specs[41:45]))
                closest_run = max(all_sf_runs)

        elif date == closest_date:
            # If the same date, there are two options -> then find the last one before the run
            all_sf_runs = []
            with open(f"/storage/spinflip_log/{closest_date}") as file:
                for specs in file.read().split(f"{'-' * 71}\n\n\n")[:-1]:
                    all_sf_runs.append(int(specs[41:45]))

            if min(all_sf_runs) >= run:
                print("Need to go to the previous session")
                closest_run, closest_date = find_closest_run(float('inf'), date - 1)
            else:
                closest_run = max([r for r in all_sf_runs if r <= seq])

        return closest_run, closest_date

    def find_specs(run, date):
        closest_run, closest_date = find_closest_run(run, date)

        if closest_date == None:
            return "N/A"

        else:
            sp_str = ''
            with open(f"/storage/spinflip_log/{closest_date}") as file:
                for specs in file.read().split(f"{'-' * 71}\n\n\n")[:-1]:
                    if int(specs[41:45]) == int(closest_run):
                        sp_str = specs
            return sp_str, closest_date, closest_run

    sp_str, closest_date, closest_run = find_specs(seq, date)
    sp_form = [s.split(" ") for s in sp_str.split("\n")]
    header = " ".join(sp_form[0])

    output = f"{'-' * 16} SPINFLIP Parameters set on {closest_date}-{str(closest_run).zfill(4)} {'-' * 16}\n"
    for s in sp_form[1:-1]:
        first_string = " ".join(s[:-2])
        output += f"{first_string : <50}{s[-2] : >10} {s[-1]}\n"

    return output

def getODTParams(date, seq):
    """ Get additional information about the ODTtrap
    """
    date = int(date)
    seq = int(seq)

    def find_closest_date(date):
        all_sf_dates = [int(a[:-4]) for a in sorted(os.listdir("/storage/ODT_setuplog"), reverse=True)]
        all_dates = [d for d in all_sf_dates if d <= date]
        if len(all_dates) != 0:
            closest_date = max(all_dates)
            return closest_date
        else:
            return None

    def find_closest_run(run, date):
        closest_date = find_closest_date(date)

        if closest_date == None:
            return [None, None]

        elif date != closest_date:
            # If not the same date, then find the last one from the closest date
            all_sf_runs = []
            with open(f"/storage/ODT_setuplog/{closest_date}.txt") as file:
                for specs in file.read().split("------------------------------------\n")[1:]:
                    all_sf_runs.append(int(specs.split(" ")[10]))
                closest_run = max(all_sf_runs)

        elif date == closest_date:
            # If the same date, there are two options -> then find the last one before the run
            all_sf_runs = []
            with open(f"/storage/ODT_setuplog/{closest_date}.txt") as file:
                for specs in file.read().split("------------------------------------\n")[1:]:
                    # print(specs.split(" ")[10])
                    all_sf_runs.append(int(specs.split(" ")[10]))

            if min(all_sf_runs) >= run:
                print("Need to go to the previous session")
                closest_run, closest_date = find_closest_run(float('inf'), date - 1)
            else:
                closest_run = max([r for r in all_sf_runs if r <= seq])

        return closest_run, closest_date
    
    def find_specs(run, date):
        closest_run, closest_date = find_closest_run(run, date)

        if closest_date == None:
            return "N/A"

        else:
            sp_str = ''
            with open(f"/storage/ODT_setuplog/{closest_date}.txt") as file:
                for specs in file.read().split("------------------------------------\n")[1:]:
                    if int(specs.split(" ")[10]) == int(closest_run):
                        sp_str = specs
            return sp_str, closest_date, closest_run


    sp_str, closest_date, closest_run = find_specs(seq, date)


    sp_form = [s.split(" ") for s in sp_str.split("\n")]
    header1 = " ".join(sp_form[0])
    header2 = " ".join(sp_form[1])
    header3 = " ".join(sp_form[2])

    output = f"{'-' * 16} ODT SCAN Parameters set on {closest_date}-{str(closest_run).zfill(4)} {'-' * 16}\n"
    output += f"{header1[4:-4]}\n"
    output += f"{header2[4:-4]}\n"
    # output += f"{header3}\n\n"
    for s in sp_form[3:-3]:
        first_string = " ".join(s[:-2])
        output += f"{first_string : <50}{s[-2] : >10} {s[-1]}\n"
    output += f"{'-' * 63}"

    return output

def getAllExperimantalParameters(date, seq):
    """ Get information about each run from the param files. """
    date = str(date)
    run_id = str(seq).zfill(4)
    output = "Information for date {:}, run ID {:}:\n{:}\n".format(date, run_id, "-" * 84)
    with open("/storage/data/" + date + '/' + run_id + '/parameters.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        countr = 0
        for row in csvreader:
            output += '{:<15} | {:>10}   ||   '.format(row[0], round(float(row[1]), 8))
            countr += 1
            if countr % 3 == 0:
                output += '\n'
        output += '\n'
        output += "-" * 84
    return output

def getComment(date, run):
    """ Fetch the comment, if there is one """
    run = str(run).zfill(4)
    if os.path.exists("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run)):
        with open("/storage/BECViewer/comments/{:}_{:}.txt".format(date, run), "r") as commentFile:
            comment = commentFile.read()
    else:
        comment = ""
    return comment
    
def get_nr_of_atoms(date, shot):
    fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/0.fits'
    nr = len(pyfits.open(fits_path)[0].data.astype(float))
    return nr

def get_nr_of_flats(date, shot):
    fits_path = f'/storage/data/{date}/{str(shot).zfill(4)}/1.fits'
    nr = len(pyfits.open(fits_path)[0].data.astype(float))
    return nr

def get_max_shot(datestring):
    # Save the largest runID, to be used for comparison for the live update
    # Get today's date, formatted
    path = f"/storage/data/{datestring}/"
    if os.path.exists(path):
        dirList = [x for x in os.listdir(path) if os.path.isdir(path + x)]
        if len(dirList) > 0:
            output = sorted(dirList)[-1]
        else:
            output = None
    else:
        output = None
    return int(output)

def which_comment(date):
    """ Get which shots have a comment. """
    shot_numbers = []
    for filename in os.listdir('/storage/BECViewer/comments/'):
        if filename.endswith('.txt'):
            file_date, shot = filename.replace('.txt', '').split('_')
            if str(file_date) == str(date):
                shot_numbers.append(int(shot))
    return sorted(shot_numbers)

def main(date, shot):
    """
    This function takes the date and shot number as input and returns all of the parameters for that shot, inlcuding the comments. 
    """
    print(f"Date: {date}")
    print(f"Shot number: {shot}")
    print("Nr of atom pictures: ", get_nr_of_atoms(date, shot))
    print("Nr of flat pictures: ", get_nr_of_flats(date, shot))
    print(getAllExperimantalParameters(date, shot))
    print(getSpinflipParams(date, shot))
    print(getODTParams(date, shot))
    print("Comments:")
    for sh in range(max(shot-30, 0), min(shot+30, get_max_shot(date))): 
        # print(sh)
        if sh in which_comment(date):
            if sh == shot:
                print("--------------------------------------------------")
                print(f"|{str(sh).zfill(4)}|:", getComment(date, sh))
                print("--------------------------------------------------")

            print(f"|{str(sh).zfill(4)}|", getComment(date, sh))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")

    

date = args.date
shot = args.shot
try:
    date = int(date)
    shot = int(shot)
except ValueError:
    print("Date and shot number must be integers.")
    sys.exit(1)

main(date, shot)




