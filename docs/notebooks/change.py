import glob
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())

data_folder = "/Users/jiang/data/radar20190101"

from netCDF4 import Dataset
from datetime import datetime,timedelta
from rainymotion import models, metrics, utils
from collections import OrderedDict
import h5py
import wradlib.ipol as ipol

nc_files = glob.glob(os.path.join(data_folder,"*.nc"))  # 288 = 12*24
nc_files.sort()

all_data = [] # 288 arrays with shape (1000,1000)
for nc_file in nc_files:
    root = Dataset(nc_file, "r")
    rain = root['var0_1_203_surface'][0,:,:] # masked array, shape(3360,3560)
    rain_reduced = rain[1500:2500,1000:2000].copy()
    rain_reduced.fill_value = 0.0
    rain_filled = rain_reduced.filled()
    all_data.append(rain_filled) 

now = 1
threshold = 1 

hh = str(now * 5 // 60).zfill(2) 
mm = str(now * 5  % 60).zfill(2)
print(f"now = 2019.1.1-{hh}:{mm}")

inputs = np.array([all_data[now-1],all_data[now]])

# SparseSD mode
model = models.SparseSD()
model.input_data = inputs
model.lead_steps = 13
nowcast = model.run()  # shape (12, 900, 900)