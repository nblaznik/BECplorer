# A collection of supplementary functions used for import/export/analysis of the .fits files.
# ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------- IMPORTS ----------------------------------------------------
import csv


# ---------------------------------------------------- FUNCTIONS ---------------------------------------------------
def get_parameter(date, seq, paramname):
    """
    Gets the parameter from a particular run id. Path is defined relatively to the script. If you want
    time of flight for run id 0033, you execute: getParameter(33, "tof")
    -----
    :param date: The date of the run
    :param seq: Run number
    :param paramname: The name of the parameter (as a string). Choose between IterationNum,
    IterationCount, tof, probeshut, rftime, rftime1, rftime2, tofcomp, t_light, xcompdac_mot,
    ycompdac_mot, zcompdac_mot, xcurr_mot, ycurr_mot, zcurr_mot, xcompdac_mt, ycompdac_mt,
    zcompdac_mt, xcurr_mt, ycurr_mt, zcurr_mt, trap_depth, loopdummy, thold, img_wait,
    rftime3, reprate, probewait, probeprewait,
    :return: the value of the specified parameter
    """
    date = str(date)
    run_id = str(seq).zfill(4)
    with open("/storage/data/" + date + '/' + run_id + '/parameters.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        for row in csvreader:
            if row[0] == paramname:
                param = float(row[1])
    return param


