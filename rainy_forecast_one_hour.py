# written by Yuchao Jiang on 2020.3.3
# use rainymotion library, only make one prediction for the next one hour

import os
import numpy as np
import glob
from datetime import datetime,timedelta
from rainymotion import models, metrics, utils
from time import time
import cv2
import matplotlib.pyplot as plt
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
import joblib 

t0 = time()
#print(os.getcwd())
data_folder = "/Users/jiang/data/kakuho_dev"  ## in local computer 
output_folder = "./output_vector"
if not os.path.exists(data_folder):
	data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
bin_files.sort()
if len(bin_files) < 14:
	print("No enough data, please check folder:")
	print(data_foler)
	sys.exit(0)
now_files = bin_files[-14:-12]
truth_file = bin_files[-1]
print("--------------- using these files ---------------")
print(*now_files, sep = "\n")
print(truth_file)

datetime_str = now_files[-1].split("/")[-1].split(".")[0]
datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")
print("--------------- now time is: ---------------")
print(datetime_object)

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
threat = metrics.CSI(gt_in_60_min, prediciton, threshold = threshold)
print("--------------- Result: ---------------")
print(f"rain coverage = {np.sum(gt_in_60_min>= threshold)/np.sum(~mask):.3f}")
print(f"threat = {threat:.2f}")
print(f"time cost: {time()-t0:.2f} seconds")