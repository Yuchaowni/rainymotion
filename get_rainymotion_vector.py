# note on 2020.2.21
# 1. download 2 images from data (http://data-catalog.wni.co.jp/data_catalog/view.cgi?tagid=400220382)
# 2. process and make it ready for use in opencv
# 3. output motion vector in specified format

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import urllib.request
from bs4 import BeautifulSoup

print(os.getcwd())
dt = datetime.utcnow()
print("UTC now:",dt.strftime("%Y-%m-%d %H:%M"))

base_page = "http://stock1.wni.co.jp/cgi-bin/list.cgi?path=/stock1m/400220382"
base_URL = "http://stock1.wni.co.jp/stock1m/400220382"
base_goal = "/Users/jiang/data/radar400220382"

page_URL = os.path.join(base_page,dt.strftime("%Y/%m/%d"))
source_folder = os.path.join(base_URL,dt.strftime("%Y/%m/%d"))
with urllib.request.urlopen(page_URL) as response:
	html = response.read()

soup = BeautifulSoup(html, 'html.parser')
file_list = [a.string for a in soup.find_all(target = '_blank')]
file_list.sort()
#print(file_list)
last_two = file_list[-2:]
print(last_two)

goal_folder = os.path.join(base_goal, dt.strftime("%Y/%m/%d"))
if not os.path.exists(goal_folder):
    os.makedirs(goal_folder)

for file_string in last_two:
	source_path = os.path.join(source_folder, file_string)
	goal_path = os.path.join(goal_folder, file_string)
	try: 
	    urllib.request.urlretrieve(source_path, goal_path)
	except: 
	    print(f"not exist:{source_path}")
	    continue

inputs = np.zeros(shape = (2,1000,1000), dtype = np.float32)

for file_string in last_two:
	ru_file = os.path.join(goal_folder, file_string)
	gz_file = ru_file + ".bin.gz"
	cmd = f"perl cutruhead.pl {ru_file} > {gz_file}"
	os.system(cmd)

	cmd = f"gzip -d {gz_file}"
	os.system(cmd)

	bin_file = ru_file + ".bin"
	data = np.fromfile(bin_file,dtype = "float32")
	os.system(f"rm -r {bin_file}")  
	#print (data.shape)  # (8601600,) = 3360 * 2560
	print(f"max ={np.max(data)}, min = {np.min(data)}") 
	data = data.reshape((3360,2560))[1500:2500,1000:2000]
	data[data<0] = 0

from rainymotion import models, metrics, utils
