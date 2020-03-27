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


data_folder = "/Users/jiang/data/kakuho_dev"  ## in local computer, for testing purpose
if not os.path.exists(data_folder):
	data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
bin_files.sort()
now_files = bin_files[-2:]

print("--------------- using these files ---------------")
print(*now_files, sep = "\n")

datetime_str = now_files[-1].split("/")[-1].split(".")[0]
datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")
print("--------------- now time is: ---------------")
print(datetime_object)

# data processing
inputs = np.zeros(shape = (2,3360,2560), dtype = np.float32)
for i, bin_file in enumerate(now_files):
	inputs[i] = np.fromfile(bin_file, dtype = "float32").reshape((3360,2560))
mask = inputs[0] < 0
inputs[inputs < 0] = 0

# call rainymotion and make prediction
model = models.Dense()
model.input_data = inputs
model.lead_steps = 25 # 13 for one hour, 25 for 2 hour
prediciton = model.run()

# draw prediction and overlap with basemap
# use seperate of 2 to reduce time cost
for i in range(0,25,2):
	plt.figure(dpi=150)
	m = Basemap(llcrnrlat=20,urcrnrlat=48, llcrnrlon=118, urcrnrlon=150,resolution = "i")
	lon = np.linspace(118.00625, 149.99375, 2560)
	lat = np.linspace(20.004167, 47.995833, 3360)
	lons, lats = np.meshgrid(lon, lat)
	m.contourf(lons, lats, prediciton[i], levels=list(range(1,40)), cmap='jet' )
	m.drawcoastlines(color='black')
	m.drawcountries()
	m.drawmeridians(np.arange(118, 150, 5), labels=[1,0,0,1]) # left, right, top or bottom
	m.drawparallels(np.arange(20, 48, 5), labels=[1,0,1,0])
	plt.colorbar(label='mm/h')

	m.pcolormesh(lons, lats, mask, cmap= "binary", alpha = 0.002)
	plt.title(f"now = {datetime_object.strftime('%Y-%m-%d %H:%M')} UTC,+{str(i*5).zfill(2)} min")
	plt.savefig(f"realtime_prediction_basemap_{str(i).zfill(2)}.png",format = "png",bbox_inches='tight')
	plt.close()

# make gif 
png_files = glob.glob("./realtime_prediction_basemap*.png")
png_files.sort()
images = []
for filename in png_files:
    images.append(imageio.imread(filename))
    os.system(f"rm -r {filename}")
output_file = f'realtime_prediction_basemap.gif'
imageio.mimsave(output_file, images,duration = 1)

#print("time cost: ", time()-t0)  # 195 s = 3 min for 13 images