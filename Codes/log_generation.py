#
#    log_generation.py
#
#    This is a demo for API traffic analysis, implemented in TensorFlow.

import tensorflow as tf
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt


#  generation some log for normal app traffic
num_device = 50
num_message_per_device = 1000
message_interval_min = 15
base_time = dt.datetime.strptime('2017-11-27 00:00:00','%Y-%m-%d %H:%M:%S')
np.random.seed(16)
wakeup_time_min = np.random.randint(low=0, high=30, size=num_device)
wakeup_time_sec = np.random.randint(low=0, high=60, size=num_device)


with open('C://Ericsson//Projects//tensorflow//apigw//normal_log.csv', 'wt') as f:
    #Generate MO message Log
    for i in range(num_device):
        mt_time_stamp = base_time + dt.timedelta(minutes=int(wakeup_time_min[i])) + dt.timedelta(seconds=int(wakeup_time_sec[i]))
        for j in range(num_message_per_device):
            mt_time_stamp += dt.timedelta(minutes=message_interval_min) 
            mt_time_stamp += dt.timedelta(seconds=np.random.randint(0, 3))
            mo_time_stamp = mt_time_stamp + dt.timedelta(minutes=3)
            print('D{0},NIDD-MT,{1}'.format(i, mt_time_stamp), file=f)
            print('D{0},NIDD-MO,{1}'.format(i, mo_time_stamp), file=f)
            mo_time_stamp += dt.timedelta(seconds=20)
            print('D{0},NIDD-MO,{1}'.format(i, mo_time_stamp), file=f)


# generation of some log for abnormal app traffic
num_device = 100
num_message_per_device = 1000
message_interval_min = 15
base_time = dt.datetime.strptime('2017-12-20 00:00:00','%Y-%m-%d %H:%M:%S')
np.random.seed(16)
wakeup_time_min = np.random.randint(low=0, high=30, size=num_device)
wakeup_time_sec = np.random.randint(low=0, high=60, size=num_device)


with open('C://Ericsson//Projects//tensorflow//apigw//abnormal_log.csv', 'wt') as f:
    #Generate MO message Log
    for i in range(num_device):
        mt_time_stamp = base_time + dt.timedelta(minutes=int(wakeup_time_min[i])) + dt.timedelta(seconds=int(wakeup_time_sec[i]))
        for j in range(num_message_per_device):
            mt_time_stamp += dt.timedelta(minutes=message_interval_min) 
            mt_time_stamp += dt.timedelta(seconds=np.random.randint(0, 3))
            mo_time_stamp = mt_time_stamp + dt.timedelta(minutes=3)
            print('D{0},NIDD-MT,{1}'.format(i, mt_time_stamp), file=f)
            print('D{0},NIDD-MO,{1}'.format(i, mo_time_stamp), file=f)
            mo_time_stamp += dt.timedelta(seconds=20)
            print('D{0},NIDD-MO,{1}'.format(i, mo_time_stamp), file=f)

    base_time = dt.datetime.strptime('2017-12-20 12:00:00','%Y-%m-%d %H:%M:%S')
    time_stamp = base_time + dt.timedelta(minutes=int(wakeup_time_min[49])) + dt.timedelta(seconds=int(wakeup_time_sec[49]))
    for k in range(4000):
        time_stamp += dt.timedelta(minutes=1) 
        time_stamp += dt.timedelta(seconds=np.random.randint(0, 3))
        print('D50,NIDD-MO,{1}'.format(i, time_stamp), file=f)
