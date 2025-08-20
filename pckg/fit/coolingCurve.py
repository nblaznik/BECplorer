"""
Created on Wed May 11 11:44:19 2022

@author: jbesc
"""

import numpy as np
import csv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


# Note: we should start saving fit uncertainty

def ParamRead(sys_path, str_lst):  # given a list of shots, reads in the gaussian fit and return particle number and temperature
    nthermal = []
    tx = []
    tz = []
    shots = []
    for shot in str_lst:
        filename = sys_path + '/' + shot
        files=listdir(filename)
        if 'fit_bimodal.param' in files:
            filename=filename+'/fit_bimodal.param'
        elif 'fit_gauss.param' in files:
            filename=filename+'/fit_gauss.param'
        try:
            with open(filename, "r") as varsfile:
                csvreader = csv.reader(varsfile, delimiter=',')
                for row in csvreader:
                    if row[0] == 'ntherm':
                        nthermal.append(float(row[-1]))
                    if row[0] == 'tx':
                        tx.append(float(row[-1]))
                    if row[0] == 'tz':
                        tz.append(float(row[-1]))
            shots.append(shot)
        except:
            print('file loading error')
    return shots, nthermal, tx, tz  # which temperature is better?


def NTplot(path, rev_list,seq_list):  # makes a loglog plot (N,T) for a sequance of shots and a single data point for the avarage and standard deviation of the reference list
    rev_shots, rev_nthermal, rev_tx, rev_tz = ParamRead(path, rev_list)  # reference point
    rev_Nth = np.mean(rev_nthermal)
    rev_T = np.mean(rev_tx)
    seq_shots, seq_nthermal, seq_tx, seq_tz = ParamRead(path, seq_list)  # sequence in which we want to find optimum
    print('SD:%0.2f' % (np.std(rev_nthermal)))
    plt.loglog(seq_tx, seq_nthermal, 'o')
    plt.xlabel('ln(T)')
    plt.ylabel('ln(N)')
    plt.grid(True, which="both", ls="--")
    #plt.xlim([10 ** -6, 1.5*10 ** -5])
    #plt.ylim([10 ** 7, 2.5 * 10 ** 8])
    plt.errorbar(rev_T, rev_Nth, xerr=np.std(rev_tx), yerr=np.std(rev_nthermal), fmt='.k')
    plt.show()


def Effplot(path, rev_list, seq_list):  # calculate and plot the efficiency parameter per shot
    rev_shots, rev_nthermal, rev_tx, rev_tz = ParamRead(path, rev_list)  # reference point
    rev_Nth = np.mean(rev_nthermal)
    rev_T = np.mean(rev_tx)
    seq_shots, seq_nthermal, seq_tx, seq_tz = ParamRead(path, seq_list)  # sequence in which we want to find optimum
    eff_param = []
    for i in range(0, len(seq_shots)):
        eff_param.append(-1*(np.log(rev_Nth) - np.log(seq_nthermal[i]) / (np.log(rev_T) - np.log(seq_tx[i]))))
    shotlistnumbers = list(map(int, seq_shot_list))
    plt.plot(shotlistnumbers, eff_param, 'o')
    plt.xticks(shotlistnumbers)
    plt.xlabel('Shot')
    plt.ylabel(r'$\alpha$')
    plt.show()


sys_path = "/storage/data/20220513"
#rev_shot_list = ['0001', '0002','0003', '0004','0005','0006','0007','0008','0009','0010']
#seq_shot_list = ['0003', '0004', '0005']

rev_shot_list=[]
for x in range(54,57):
    if x<10:
        shot_str = '000%i' % (x)
    elif x<100:
        shot_str = '00%i' % (x)
    elif x<1000:
        shot_str = '0%i' % (x)
    rev_shot_list.append(shot_str)

seq_shot_list=[]
for x in range(58,83):
    if x<10:
        shot_str = '000%i' % (x)
    elif x<100:
        shot_str = '00%i' % (x)
    elif x<1000:
        shot_str = '0%i' % (x)
    seq_shot_list.append(shot_str)

print(seq_shot_list)
NTplot(sys_path, rev_shot_list, seq_shot_list)
#Effplot(sys_path, rev_shot_list, seq_shot_list)

seq_shots, seq_nthermal, seq_tx, seq_tz = ParamRead(sys_path, seq_shot_list)
print(seq_nthermal)

