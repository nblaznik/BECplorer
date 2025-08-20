import csv
import matplotlib.pyplot as plt

def get_parameter(date, seq, paramname):
    """
    Get the value of the parameter form the parameters.param file. Very often used, might be better to import
    it from the pcgk/fits file. But it might be better to include those functions here.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    param = "N/A"
    try:
        with open(path + date + '/' + run_id + '/parameters.param') as paramfile:
            csvreader = csv.reader(paramfile, delimiter=',')
            for row in csvreader:
                if row[0] == paramname:
                    param = float(row[1])
    except:
        param = "N/A"
    return param


def get_fit_parameter(date, seq, mode):
    """
    Get the value of the fitted parameters, such as the number of atoms or temperature
    from the fit_mode.param file. Get all of them, and append the values in an array.
    """
    path = "/storage/data/"
    date = str(date)
    run_id = str(seq).zfill(4)
    mode = mode
    params = []
    with open(path + date + '/' + run_id + '/fit_' + mode + '.param') as paramfile:
        csvreader = csv.reader(paramfile, delimiter=',')
        next(csvreader)
        next(csvreader)
        for row in csvreader:
            params.append(float(row[1]))
    return params


lts = []
pns = []
lts2 = []
pns2 = []
lts3 = []
pns3 = []

for i in range(12, 25):
    lts.append(get_parameter(20220502, i, 'loading_time'))
    pns.append(get_fit_parameter(20220502, i, 'gauss')[17])

for i in range(25, 38):
    lts2.append(get_parameter(20220502, i, 'loading_time'))
    pns2.append(get_fit_parameter(20220502, i, 'gauss')[17])

for i in range(38, 56):
    lts3.append(get_parameter(20220502, i, 'loading_time'))
    pns3.append(get_fit_parameter(20220502, i, 'gauss')[17])


plt.scatter(lts, pns)
plt.plot(lts, pns, label="With LN2")
plt.scatter(lts2, pns2)
plt.plot(lts2, pns2, label="Without LN2")
plt.scatter(lts3, pns3)
plt.plot(lts3, pns3, label="Without LN2, later")
plt.xlabel("MOT loading time [s]")
plt.ylabel("Particle Number @ 60 sec")
plt.legend()
plt.show()
