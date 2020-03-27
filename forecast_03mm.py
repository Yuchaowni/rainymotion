###!/usr/bin/env conda run -n rainymotion python

# written by Yuchao Jiang on 2020.3.13
# based on forecast_one_hour.py

# use rainymotion library, make 12 predictions in every 5 minutes for the next hour
# download the kakuho data, extract and compare the threat score with rainymotion

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import glob
from datetime import datetime,timedelta
from time import time
from rainymotion import models, metrics, utils
import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
import joblib 
from netCDF4 import Dataset
import sys

# 1. get radar data from 
data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
bin_files.sort()
if len(bin_files) < 14:
    print("No enough data, please check folder:")
    print(data_foler)
    sys.exit(0)
now_files = bin_files[-14:-12]
truth_files = bin_files[-12:]
datetime_str = now_files[-1].split("/")[-1].split(".")[0]
datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")

print("--------------- using now time as: ---------------")
print(datetime_object)

#print("--------------- using these files for rainymotion prediction and verification ---------------")
#print(*now_files, sep = "\n")
#print(*truth_files, sep = "\n")

threshold = 0.29
inputs = np.zeros(shape = (2,3360,2560), dtype = np.float32)
for i, bin_file in enumerate(now_files):
    inputs[i] = np.fromfile(bin_file, dtype = "float32").reshape((3360,2560))
mask = inputs[0] < 0
inputs[inputs < 0] = 0

model = models.Dense()
model.input_data = inputs
model.lead_steps = 13
prediciton = model.run()


# --------------------------------------------------------------------------------
# 2. get kakuho prediction data
# 2.1 get kakuho now time and file
dt_now = datetime_object
base_URL  = "http://stock1.wni.co.jp/stock_hdd/411024220/0200600011024220"
source_folder = os.path.join(base_URL, dt_now.strftime("%Y/%m/%d"))
kakuho_file = dt_now.strftime('%Y%m%d_%H%M00.000')
source_path = os.path.join(source_folder, kakuho_file)
print("---- using kakuho source file : -----------------------")
print(source_path)
if not os.path.exists(kakuho_file):
    cmd = f"wget {source_path}"
    os.system(cmd)

# 2.2 convert wgrib2 to nc file and extract
#var = ":60 min fcst"
varlist = [":{} min".format(i) for i in range(5,65,5)]
var = '|'.join(varlist)
nc_file = "temp.nc"
cmd = f"wgrib2 {kakuho_file} -s | egrep '({var})'|wgrib2 -i {kakuho_file} -netcdf {nc_file}"
os.system(cmd)

#rain_reduced = Dataset(nc_file, "r")['APCP_surface'][0]
#rain_reduced.fill_value = 0.0 
#kakuho = rain_reduced.filled() * 6  # mm/10 min to mm/h
root = Dataset(nc_file, "r")  
os.system(f"rm -r {nc_file}")
os.system(f"rm -r {kakuho_file}")

# 3. compare and print

rainy_threat_13 = [1]
kakuho_threat_13 = [1]
persis_threat_13 = [1]
for (i,truth_file) in enumerate(truth_files):
    truth_data = np.fromfile(truth_file, dtype = "float32").reshape((3360,2560))
    rainy_threat_13.append(metrics.CSI(truth_data, prediciton[i+1], threshold = threshold))

    kakuho_data = root['APCP_surface'][i]
    kakuho_data.fill_value = 0.0
    kakuho_data = kakuho_data.filled().astype('float32') * 6 # mm/10 min to mm/h
    kakuho_threat_13.append(metrics.CSI(truth_data, kakuho_data, threshold = threshold))
    persis_threat_13.append(metrics.CSI(truth_data, inputs[-1], threshold = threshold))

coverage = np.sum(truth_data >= 0.1)/ np.sum(~mask)
#kakuho_threat = metrics.CSI(gt_in_60_min, kakuho, threshold = 0.1)
print("--------------- Result: ---------------")
print(f"rain coverage = {coverage:.3f}")
print(f"threat score of rainymotion = {rainy_threat_13[-1]:.2f}")
print(f"threat score of rain kakuho = {kakuho_threat_13[-1]:.2f}")
print(f"threat score of presist model = {persis_threat_13[-1]:.2f}")

# 4.1 plot 5 min head-to-head comparison
lead_time = list(range(0,65,5))
plt.figure(dpi=100)
plt.plot(lead_time, rainy_threat_13,'o-',label = "Optical Flow (rainymotion)")
plt.plot(lead_time, kakuho_threat_13,'s-',label = "Rain Kakuho")
plt.plot(lead_time, persis_threat_13,'*-',label = "Persistence")
plt.legend()
plt.ylim([0.2, 0.8])
plt.xlim([0, 62])
plt.yticks(np.arange(0.2, 0.81, step=0.1)) 
plt.xticks(np.arange(0, 61, step=5)) 
plt.grid()
plt.ylabel("Threat score")
plt.xlabel("Minutes after now")
plt.title(f"Now = {datetime_object.strftime('%Y-%m-%d %H:%M')} UTC, rain coverage = {coverage*100:.1f} %, threshold = {threshold} mm/h")
plt.savefig("./03mm_threat_compare.png",format = "png",bbox_inches='tight')




