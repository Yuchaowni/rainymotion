Author: Yuchao Jiang

Last update: 2020.4.3

Every thing in one page:  http://172.16.232.61/jiang/rainy/Optical_flow_nowcasting.html

Source code in GitHub: https://github.com/Yuchaowni/rainymotion

Purpose: future maintenance, improvement and new comers to take over.

[TOC]

## 1. environment setup

Python virtual environment is setup as follows:

```shell
sudo yum update
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
# after uploading sh files
bash Anaconda3-2019.10-Linux-x86_64.sh
exit
# re-connect
conda env list
# base                  *  /home/jiang/anaconda3
python --version
# Python 3.7.4
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -n rainymotion python=3 wradlib opencv=3.4.9 basemap numpy scipy h5py matplotlib scikit-learn scikit-image beautifulsoup4 basemap-data-hires
conda activate rainymotion 
# conda install nb_conda  # for local computer 
# after uploading rainymotion source code folder and setup.py
python setup.py install
```

## 2. how to run python codes?

First, need to activate the python virtual environment

```shell
conda activate rainymotion
```

Then run python for once or continuously using `nohup`

All these codes will automatically fetch the latest data in the same server. 

| code                                           | Output                                                       | Note                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| local_data_to_motion_vector.py                 | xxx_u.bin, xxx_v.bin                                         | each shape (3360,2560), float32, masked value is filled with 0 |
| local_data_to_motion_vector.py --output_figure | xxx_u.bin, xxx_v.bin, xxx_basemap_motion_vector.png          | u,v binary files + motion vector plot overlap basemap. Generating basemap need 30 s |
| forecast_one_hour.py                           | all_threat.csv, 1hour_threat_compare.png                     | validate the single prediction using 2 radar data in the previous hour, accumulate threat score and plot. |
| forecast_every_5min.py                         | all_threat.csv, 5min_threat_compare.png　1hour_threat_compare.png | including above, make prediction every 5 min for the past hour |
| forecast_every_5min.py --threshold 0.29        | same as above                                                | pass threshold parameter into main source code               |
| forecast_beyond_1hour.py                       | beyond1hour_threat_compare.png                               | Use all the available current data for verification          |
| forecast_realtime_basemap.py                   | realtime_prediction_basemap.gif                              | use latest 2 images for real-time prediction and visualize   |
| scheduler_forecast_one_hour.py                 |                                                              | Run python codes every 10 minutes                            |

## 3 regular maintenance

```shell
ssh pt-atky-kakuho00.wni.co.jp
conda activate rainymotion
cd rainy
nohup python scheduler_forecast_one_hour.py &
xxx
tail -n 10 all_threat.csv
ps ax | grep scheduler_forecast_one_hour.py
kill xxx
```

