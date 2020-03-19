# updated on 2019.6.11: add human edited data
# written by Yuchao on 2019.4.1
# purpose: run own_ml_cn_realtime.py every hour and delete outdated data
# how to run: python scheduler_china.py xx, xx is stock1 or asap2

import sys
import schedule
import time
import datetime
import os



def job():
	dt = datetime.datetime.utcnow()
	print("UTC now:",dt)
	cmd = "python forecast_one_hour.py"
	os.system(cmd)  # 0 indicate success, others indicate fail
	print("wait for next update in 10 minutes  --------------------------------------------")
	return 

job()
#schedule.every().hour.at(":30").do(job)  # every hour at 10 min
schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1) # 