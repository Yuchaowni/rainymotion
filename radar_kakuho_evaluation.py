# written by Yuchao on 2020.3.12
# check and download the latest available radar401300210, parse the datetime
# backward 1 hour and get download the kakuho data
# extract the kakuho data and make comparison

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import urllib.request
from bs4 import BeautifulSoup
from rainymotion import models, metrics, utils
from netCDF4 import Dataset

print(os.getcwd())
dt = datetime.utcnow()
print("UTC now:",dt.strftime("%Y-%m-%d %H:%M"))

base_page = "http://stock1.wni.co.jp/stock/401300210/0000300100200012"
base_URL = "http://stock1.wni.co.jp/stock/401300210/0000300100200012"
base_goal = "/Users/jiang/data/jma_radar"

# 1. get the latest radar file and data
# 1.1 obtain the radar file name
page_URL = os.path.join(base_page,dt.strftime("%Y/%m/%d"))
with urllib.request.urlopen(page_URL) as response:
	html = response.read()
soup = BeautifulSoup(html, 'html.parser')
file_list = [a.string for a in soup.find_all('a') if ".000" in a.string]
file_list.sort()
latest = file_list[-1]

source_folder = os.path.join(base_URL,dt.strftime("%Y/%m/%d"))
goal_folder = os.path.join(base_goal, dt.strftime("%Y/%m/%d"))
if not os.path.exists(goal_folder):
    os.makedirs(goal_folder)

# 1.2. download radar data
file_string = latest
source_path = os.path.join(source_folder, file_string)
download_radar_path = os.path.join(goal_folder, file_string)
try: 
    urllib.request.urlretrieve(source_path, download_radar_path)
except: 
    print(f"not exist:{source_path}")

# 1.3 convert wgrib2 file to nc and extract
nc_file = "temp1.nc"
cmd = f"wgrib2 {download_radar_path} -s | egrep 'surface'|wgrib2 -i {download_radar_path} -netcdf {nc_file}"
fail = os.system(cmd)
if fail:
    print("wgrib2 wrong at ",download_radar_path)
rain_reduced = Dataset(nc_file, "r")['var0_1_203_surface'][0]
rain_reduced.fill_value = 0.0 
ground_truth = rain_reduced.filled()
os.system(f"rm -r {nc_file}")
jma_mask = rain_reduced.mask

# --------------------------------------------------------------------------------
# 2. get kakuho prediction data
# 2.1 get kakuho file name
dt_12 = datetime.strptime(file_string.split(".")[0], '%Y%m%d_%H%M%S')
print("latest available time:", dt_12)
time_step = 5 * 60 # seconds
dt_now = dt_12 - timedelta(seconds = time_step * 12)

# 2.2 download kakuho file
base_URL  = "http://stock1.wni.co.jp/stock_hdd/411024220/0200600011024220"
base_goal = "/Users/jiang/data/rain_kakuho"
file_string = dt_now.strftime('%Y%m%d_%H%M00.000')
source_folder = os.path.join(base_URL, dt.strftime("%Y/%m/%d"))
goal_folder = os.path.join(base_goal, dt.strftime("%Y/%m/%d"))
if not os.path.exists(goal_folder):
    os.makedirs(goal_folder)
source_path = os.path.join(source_folder, file_string)
goal_path = os.path.join(goal_folder, file_string)
try: 
    urllib.request.urlretrieve(source_path,goal_path)
except: 
    print(f"not exist:{source_path}")

# 2.3 convert wgrib2 to nc file and extract
var = "60 min fcst"
nc_file = "temp2.nc"
cmd = f"wgrib2 {goal_path} -s | egrep '({var})'|wgrib2 -i {goal_path} -netcdf {nc_file}"
fail = os.system(cmd)
if fail:
    print("wgrib2 wrong at ",source_path)
rain_reduced = Dataset(nc_file, "r")['APCP_surface'][0]
rain_reduced.fill_value = 0.0 
kakuho = rain_reduced.filled() * 6  # mm/10 min to mm/h
os.system(f"rm -r {nc_file}")


# 3. compare
coverage = np.sum(ground_truth >= 0.1)/ np.sum(~jma_mask)
threat = metrics.CSI(ground_truth, kakuho, threshold = 0.1)
print(f"rain coverage = {coverage:.3f}")
print(f"threat = {threat:.2f}")
