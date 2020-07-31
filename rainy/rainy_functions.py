# written by Yuchao Jiang on 2020.7.31
# python 3.7.6
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

def print_log(string, log_folder):
    log_file = f"{log_folder}/log.txt"
    with open (log_file, 'a') as text_file:
        print(string)
        print(string, file = text_file)

def get_prediction(now_files, lead_steps):
    # data processing
    inputs = np.zeros(shape = (2,3360,2560), dtype = np.float32)
    for i, bin_file in enumerate(now_files):
        inputs[i] = np.fromfile(bin_file, dtype = "float32").reshape((3360,2560))
    mask = inputs[0] < 0
    inputs[inputs < 0] = 0

    # call rainymotion and make prediction
    model = models.Dense()
    model.input_data = inputs
    model.lead_steps = lead_steps # 13 for one hour, 25 for 2 hour
    return model.run(), mask

def plot_radar(lons, lats, pred, mask, dt, i, output_folder):
    plt.figure(dpi=150)
    m = Basemap(llcrnrlat=20,urcrnrlat=48, llcrnrlon=118, urcrnrlon=150,resolution = "i")
    m.contourf(lons, lats, pred, levels=list(range(1,40)), cmap='jet' )
    m.drawcoastlines(color='black')
    m.drawcountries()
    m.drawmeridians(np.arange(118, 150, 5), labels=[1,0,0,1]) # left, right, top or bottom
    m.drawparallels(np.arange(20, 48, 5), labels=[1,0,1,0])
    plt.colorbar(label='mm/h')
    m.pcolormesh(lons, lats, mask, cmap= "binary", alpha = 0.002)
    plt.title(f"now = {dt.strftime('%Y-%m-%d %H:%M')} UTC,+{str(i*5).zfill(2)} min")
    output_png = f"{output_folder}/realtime_prediction_{str(i).zfill(2)}.png"
    plt.savefig(output_png,format = "png",bbox_inches='tight')
    plt.close()
    return output_png

def make_gif(png_files, output_folder, output_gif, keep_png = False):
    images = []
    for png_file in png_files:
        images.append(imageio.imread(png_file))
        if not keep_png:
            os.system(f"rm {png_file}")  # keep the space clean
    imageio.mimsave(f'{output_folder}/{output_gif}', images, duration = 1,loop = 4)