# written by Yuchao Jiang on 2020.3.27
# use rainymotion library
# based on rainy_forecast_one_hour.py
# based on forecast_every_5min.py
# based on kakuhoOF/read_radar_wgrib_nc_mask_basemap.ipynb
# based on rainymotion_check_specific_cases_animation.ipynb
# based on local_data_to_motion_vector.py

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
import imageio
import warnings
warnings.simplefilter("ignore")
from rainy_functions import *

def main():
    t0 = time()
    data_folder = "/Users/jiang/data/kakuho_dev"  ## in local computer, for testing purpose
    log_folder = "./"
    output_folder = "./"
    if not os.path.exists(data_folder):
        data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
    bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
    if len(bin_files) < 2:
        print_log(f"sorry, only {len(bin_files)} files, please check.",log_folder)
        return

    bin_files.sort()
    now_files = bin_files[-2:]
    date_string = now_files[-1].split("/")[-1].split(".")[0]
    dt = datetime.strptime(date_string,"%Y%m%d%H%M")
    print_log(dt.strftime(f"using time: %Y-%m-%d %H:%M, available files: {len(bin_files)}"),log_folder)
    print_log("using these 2 files for optical flow:",log_folder)
    for file in now_files:
        print_log(file,log_folder)

    prediction, mask = get_prediction(now_files, lead_steps=25)

    lon = np.linspace(118.00625, 149.99375, 2560)
    lat = np.linspace(20.004167, 47.995833, 3360)
    lons, lats = np.meshgrid(lon, lat)

    pred_pngs = []
    step = 2  # use seperate of 2 to reduce time cost
    for i in range(0,25,step):
        pred_pngs.append(plot_radar(lons, lats, prediction[i], mask,dt,i,output_folder))
    print_log(f'-- output {len(pred_pngs)} of pngs in {step*5} minutes interval',log_folder)
    output_gif = dt.strftime(f'realtime_prediction_%Y_%m_%d.%H:.%M.gif')
    make_gif(pred_pngs, output_folder, output_gif, keep_png = False)
    print_log(f'-- output_gif: {output_folder}/{output_gif}', log_folder)
    print_log(f"time cost: {time()-t0}", log_folder)  # 195 s = 3 min for 13 images

if __name__ == '__main__':
    main()