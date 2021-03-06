# written by Yuchao Jiang on 2020.2.27
# opencv 3.4.9
# python 3.7.6
# generate u,v binary files: python **.py
# generate u,v binary files + png: python **.py --output_figure
# use the lastest radar data for prediction

import argparse
import os
import cv2
import numpy as np
import numpy.ma as ma
import glob
from datetime import datetime,timedelta
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def main(output_figure = False):
    t0 = time()
    data_folder = "/Users/jiang/data/kakuho_dev"  ## in local computer 
    output_folder = "./output_vector"
    if not os.path.exists(data_folder):
        data_folder = "../../../usr/amoeba/pub/rain_kakuho/hres.jma_nowcast/out"
    bin_files = glob.glob(os.path.join(data_folder,"*ints.1km.bin"))  # 288 = 12*24
    bin_files.sort()
    now_files = bin_files[-2:]
    print("--------------- using these two files ---------------")
    print(*now_files, sep = "\n")
    datetime_str = now_files[-1].split("/")[-1].split(".")[0]
    datetime_object = datetime.strptime(datetime_str,"%Y%m%d%H%M")
    print("--------------- now time is: ---------------")
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

    u = ma.array(data = delta[:,:,0], mask = mask,fill_value = 0)
    v = ma.array(data = delta[:,:,1], mask = mask,fill_value = 0)
    # save data to binary 
    u.data.tofile(f"{output_folder}/{datetime_str}_u.bin")
    v.data.tofile(f"{output_folder}/{datetime_str}_v.bin")

    if output_figure == True:
        # plot motion vector on base map
        plt.figure(dpi=150)
        m = Basemap(llcrnrlat=20,urcrnrlat=48, llcrnrlon=118, urcrnrlon=150,resolution = "i")
        lon = np.linspace(118.00625, 149.99375, 2560)
        lat = np.linspace(20.004167, 47.995833, 3360)
        lons, lats = np.meshgrid(lon, lat)
        m.contourf(lons, lats, data_scaled[-1], levels=list(range(1,255)),cmap='jet' )
        m.drawcoastlines(color='black')
        m.drawcountries()
        m.drawmeridians(np.arange(118, 150, 5), labels=[1,0,0,1])# left, right, top or bottom
        m.drawparallels(np.arange(20, 48, 5), labels=[1,0,1,0])
        plt.colorbar()
        x,y = m(lon, lat)
        skip = 60
        plt.quiver(x[::skip], y[::skip], u[::skip,::skip], v[::skip,::skip], scale = 100,  headwidth = 3, alpha=0.4)
        m.pcolormesh(lons, lats, mask, cmap= "binary", alpha=0.002)
        plt.title(f"{datetime_object.strftime('%Y_%m_%d %H:%M ')}basemap_motion_vector")
        plt.savefig(f"{output_folder}/{datetime_str}_basemap_motion_vector.png",format = "png",bbox_inches='tight')

    print(f"time cost: {time()-t0:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "input 2 radar images and output u,v vectors ")
    parser.add_argument('--output_figure', action = "store_true",
                        help='If output (basemap + vector) figure, will takes 30 seconds')
    args = parser.parse_args()
    main(args.output_figure)
