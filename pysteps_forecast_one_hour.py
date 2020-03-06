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
import pysteps_plus
from pysteps.motion.lucaskanade import dense_lucaskanade

t0 = time()
#print(os.getcwd())
data_folder = "/Users/jiang/data/kakuho_dev"  ## in local computer 
output_folder = "./output_vector"
if not os.path.exists(data_folder):
	data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
bin_files.sort()
now_files = bin_files[-15:-12]
truth_file = bin_files[-1]
print("--------------- using these files ---------------")
print(*now_files, sep = "\n")
print(truth_file)

datetime_str = now_files[-1].split("/")[-1].split(".")[0]
datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")
print("--------------- now time is: ---------------")
print(datetime_object)

threshold = 0.1
zerovalue = -15.0
n_ens_members = 10
time_seperation_in_minute = 5 
seed = 24
p_threshold = 0.5 # probability threshold

inputs = np.zeros(shape = (3,3360,2560), dtype = np.float32)
for i, bin_file in enumerate(now_files):
	inputs[i] = np.fromfile(bin_file, dtype = "float32").reshape((3360,2560))
mask = inputs[0] < 0
# inputs[inputs < 0] = 0
zeros = inputs < threshold
inputs[~zeros] = 10.0 * np.log10(inputs [~zeros] )
inputs[zeros] = zerovalue
V = dense_lucaskanade(inputs)
R_f = pysteps_plus.forecast12(inputs, V, n_timesteps = 12, n_ens_members=n_ens_members, n_cascade_levels=6, 
    R_thr=-10.0, kmperpixel=2.0, timestep = time_seperation_in_minute, decomp_method="fft",
    bandpass_filter_method="gaussian", noise_method="nonparametric", vel_pert_method="bps",
    mask_method="incremental", seed=seed)

# Back-transform to rain rates
R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]  # (20, 1000,1000)
# compute the exceedance probability of 0.1 mm/h from the ensemble
P_f = ensemblestats.excprob(R_f, threshold, ignore_nan=True) 
   
gt_in_60_min = np.fromfile(truth_file, dtype = "float32").reshape((3360,2560))

hits   = np.sum(np.logical_and(P_f >= p_threshold, gt_in_60_min >= threshold))    
misses = np.sum(np.logical_and(P_f <  p_threshold, gt_in_60_min >= threshold))  
false_alarms = np.sum(np.logical_and(P_f >= p_threshold, gt_in_60_min < threshold))
threat = hits/(hits + false_alarms + misses)

print("--------------- Result: ---------------")
print(f"rain coverage = {np.sum(gt_in_60_min>= threshold)/np.sum(~mask):.3f}")
print(f"threat = {threat:.2f}")
print(f"time cost: {time()-t0:.2f} seconds")