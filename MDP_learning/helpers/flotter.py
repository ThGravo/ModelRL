import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('GTK3Cairo', warn=False, force=True)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some logs.')
parser.add_argument('--csvdir', type=str, default='../', help='Folder with all the csv files')
args = parser.parse_args()
print(args)
maxcount = 200
# plt.style.use('fivethirtyeight')
'''
count = 0
for root, dirs, files in os.walk(args.csvdir):
    for file in files:
        if file.endswith(".csv"):
            count += 1

fig, axarr = plt.subplots(int(min(maxcount,count)/4)+1,4)
count = 0
for root, dirs, files in os.walk(args.csvdir):
    for i, file in enumerate(files):
        if file.endswith(".csv") and count < maxcount:
            f = os.path.join(root, file)
            axarr[int(count/4),count%4].set_title(f)
            # data = np.genfromtxt(f, delimiter=',', skip_header=1)
            df = pd.read_csv(f)
            axarr[int(count/4),count%4].plot(df.values[:,1], df.values[:,2])
            # plt.plotfile(f, ('step', 'value'))  # 'wall_time',
            count += 1

plt.show()
'''
for root, dirs, files in os.walk(args.csvdir):
    for i, file in enumerate(files):
        if file.endswith(".csv"):
            f = os.path.join(root, file)
            plt.title(f)
            plt.plotfile(f, ('step', 'value'))  # 'wall_time',
plt.show()
