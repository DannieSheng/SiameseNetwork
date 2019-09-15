# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:01:27 2019
The code is used to generate summary of number of pixels belong to 6 classes of the switchgrass, for every image
@author: hdysheng
"""
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import re

gt_path = r'T:\AnalysisDroneData\groundTruth\CLMB STND 2019 Flight Data\100085_2019_07_18_15_54_58\gt_processed'
filelist = [f for f in os.listdir(os.path.join(gt_path)) if f.endswith('.mat')]

# get the correct order of files
list_frame_idx_hyper = []
for f in filelist:
    hyper_cube_name = re.findall('\d+', f)[0]
    list_frame_idx_hyper.append(int(hyper_cube_name))
index_temp = np.argsort(list_frame_idx_hyper)
file_list = [filelist[i] for i in index_temp]

col = ['class {}'.format(i) for i in range(0,7)]
row = [re.findall('\d+', f)[0] for f in file_list]
df_summary = pd.DataFrame(columns = col, index = row)

class_names = np.arange(0,7)
for f in file_list:
    count_final = np.zeros(7)
    hyper_cube_name = re.findall('\d+', f)[0]
    loaded = sio.loadmat(os.path.join(gt_path, f), squeeze_me = True)
    gt     = loaded['gt_final']
    [c_name,counts] = np.unique(gt, return_counts = True)

    if len(c_name) == 7:
        df_summary.loc[hyper_cube_name,:] = counts
    else:
        for idx, i_name in enumerate(c_name):
            count_final[i_name] = counts[idx]
        df_summary.loc[hyper_cube_name,:] = count_final
       
df_summary.to_csv(os.path.join(gt_path, 'summary.csv'), index = True, encoding = 'latin-1')
            
