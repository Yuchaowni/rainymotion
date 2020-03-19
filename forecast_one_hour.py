# written by Yuchao Jiang on 2020.3.19
# based on rainy_forecast_one_hour.py
# based on radar_kakuho_evaluation.py
# use rainymotion library, only make one prediction for the next one hour
# download the kakuho data, extract and compare the threat score with rainymotion

# additional notes:
# 1. may have duplicated datetime due to repeated run


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

# 1. get radar data from 
data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
bin_files.sort()
if len(bin_files) < 14:
    print("No enough data, please check folder:")
    print(data_foler)
    sys.exit(0)
now_files = bin_files[-14:-12]
truth_file = bin_files[-1]
datetime_str = now_files[-1].split("/")[-1].split(".")[0]
datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")

print("--------------- using now time as: ---------------")
print(datetime_object)

print("--------------- using these files for rainymotion prediction and verification ---------------")
print(*now_files, sep = "\n")
print(truth_file)

threshold = 0.1
inputs = np.zeros(shape = (2,3360,2560), dtype = np.float32)
for i, bin_file in enumerate(now_files):
    inputs[i] = np.fromfile(bin_file, dtype = "float32").reshape((3360,2560))
mask = inputs[0] < 0
inputs[inputs < 0] = 0

gt_in_60_min = np.fromfile(truth_file, dtype = "float32").reshape((3360,2560))
model = models.Dense60()
model.input_data = inputs
prediciton = model.run()
rainy_threat = metrics.CSI(gt_in_60_min, prediciton, threshold = threshold)

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
var = ":60 min fcst"
nc_file = "temp2.nc"
cmd = f"wgrib2 {kakuho_file} -s | egrep '({var})'|wgrib2 -i {kakuho_file} -netcdf {nc_file}"
os.system(cmd)

rain_reduced = Dataset(nc_file, "r")['APCP_surface'][0]
rain_reduced.fill_value = 0.0 
kakuho = rain_reduced.filled() * 6  # mm/10 min to mm/h
os.system(f"rm -r {nc_file}")
os.system(f"rm -r {kakuho_file}")

# 3. compare and print
coverage = np.sum(gt_in_60_min >= 0.1)/ np.sum(~mask)
kakuho_threat = metrics.CSI(gt_in_60_min, kakuho, threshold = 0.1)
print("--------------- Result: ---------------")
print(f"rain coverage = {coverage:.3f}")
print(f"threat score of rainymotion = {rainy_threat:.2f}")
print(f"threat score of rain kakuho = {kakuho_threat:.2f}")

# 4. store to file and plot comparison curve
threat_file = "all_threat.csv"
datetime_current = now_files[-1].split("/")[-1].split(".")[0]  # store as string
data = pd.DataFrame({"datetime":datetime_current,
                    "rain_coverage":round(coverage,3),
                   "rainy_threat": round(rainy_threat,3),
                  "kakuho_threat": round(kakuho_threat,3)},
                  index = [0])
if os.path.exists(threat_file):
    old_data = pd.read_csv(threat_file)
    data = old_data.append(data, ignore_index = False)
    # plot comparison
    plt.figure(dpi=100)
    dates = data.datetime.apply(lambda x:datetime.strptime(str(x),"%Y%m%d%H%M"))
    
    plt.plot(dates, data.rainy_threat, 'o-', label = "score of rainymotion")
    plt.plot(dates, data.kakuho_threat, 's-',label = "score of rain kakuho")
    plt.plot(dates, data.rain_coverage*2, '^-',label = "rain_coverage * 2")
    plt.legend()
    plt.title("Threat score comparison",fontsize= 15)
    plt.xlabel("datetime",fontsize= 20)
    plt.ylabel("threat score",fontsize= 20)
    plt.ylim([0,1])
    plt.grid()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:')) # 格式化时间轴标注
    plt.gcf().autofmt_xdate() # 优化标注（自动倾斜）

    #plt.show()
    plt.savefig("1hour_threat_compare.png",format = "png",bbox_inches='tight')
data.to_csv(threat_file, index = False)
print(f"threat scores have been added to {threat_file}")
