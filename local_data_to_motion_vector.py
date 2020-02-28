# written by Yuchao Jiang on 2020.2.27

import os
import numpy as np
import glob
from datetime import datetime,timedelta
#from rainymotion import models, metrics, utils
from time import time
import cv2
import matplotlib.pyplot as plt
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
import joblib 

t0 = time()
print(os.getcwd())
data_folder = "/Users/jiang/data/kakuho_dev"
bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
bin_files.sort()
now_files = bin_files[-2:]
print(*now_files, sep = "\n")
datetime_str = now_files[-1].split("/")[-1].split(".")[0]
datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")
print(datetime_object)

time_step = 5 * 60 # seconds
inputs = np.zeros(shape = (2,3360,2560), dtype = np.float32)
for i, bin_file in enumerate(now_files):
	inputs[i] = np.fromfile(bin_file, dtype = "float32").reshape((3360,2560))

mask = inputs[0] < 0
inputs[inputs < 0] = 0
X_dbz = np.log10(inputs + 0.01)
c1 = X_dbz.min()
c2 = X_dbz.max()
data_scaled = ((X_dbz - c1) / (c2 - c1) * 255).astype(np.uint8)

delta = cv2.optflow.createOptFlow_DIS().calc(data_scaled[0], data_scaled[1], None)

u = ma.array(data = delta[:,:,0], mask = mask)
v = ma.array(data = delta[:,:,0], mask = mask)

# plot motion vector
plt.figure(dpi = 150)
plt.imshow(data_scaled[-1], cmap = "gray_r",origin = "lower")
X,Y = range(2560),range(3360)
skip = 50
plt.quiver(X[::skip], Y[::skip], u[::skip,::skip], v[::skip,::skip],
           scale = 30,color='red',headwidth = 3)
plt.savefig(f"{datetime_str}_motion_vector.png",format = "png",bbox_inches='tight')
plt.close()

# plot base map 
lon, lat = joblib.load("JMA_radar_lon_lat.joblib")
plt.figure(dpi=100)
m = Basemap(llcrnrlat=20,urcrnrlat=48, llcrnrlon=118, urcrnrlon=150,resolution = "i")
lons, lats = np.meshgrid(lon, lat)
#m.contourf(lons, lats, data_scaled[-1], levels=list(range(1,65)),cmap='jet' )
m.drawcoastlines(color='black')
m.drawcountries()
m.drawmeridians(np.arange(118, 150, 5), labels=[1,0,0,1])# left, right, top or bottom
m.drawparallels(np.arange(20, 48, 5), labels=[1,0,1,0])
#plt.colorbar()
m.quiver(X[::skip], Y[::skip], u[::skip,::skip], v[::skip,::skip],scale = 30,color='red',headwidth = 3)
m.pcolormesh(lons, lats,mask,cmap= "binary",alpha=0.002)
plt.title(f"{datetime_str}_basemap_motion_vector.png")
plt.savefig(f"{datetime_str}_basemap_motion_vector.png",format = "png",bbox_inches='tight')




print(f"time cost: {time()-t0:.2f} seconds")