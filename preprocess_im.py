# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:01:01 2019
Script to preprocess test images to avoid repeated calculation
@author: hdysheng
"""
import os
os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/GitHub/SiameseNetwork')
import numpy as np
import hdf5storage
import scipy.io as sio
import pickle
import re
import pdb

datapath    = r'T:\Box2\Drone Flight Data and Reference Files\Flight Data - All Sites\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06'
hyperpath   = datapath.replace(r'Box2\Drone Flight Data and Reference Files\Flight Data - All Sites', r'AnalysisDroneData\dataPerClass')
hyperimpath = hyperpath.replace('dataPerClass', r'ReflectanceCube\MATdataCube')
labelpath   = hyperpath.replace('dataPerClass', 'groundTruth')
labelpath   = r'{}\gt_processed'.format(labelpath)

savepath    = hyperpath.replace('dataPerClass', 'Siamese')
savepath    = r'{}\testdata'.format(savepath)
if not os.path.exists(savepath):
    os.makedirs(savepath)
pdb.set_trace()   

flag             = sio.loadmat(os.path.join(hyperimpath, 'flagGoodWvlen.mat'), squeeze_me = True)
goodWvlengthFlag = flag['flag']

list_temp   = os.listdir(hyperpath)
list_file_all = []
for f in list_temp:
    cube_name = re.findall('\d+', f)[0]
    list_file_all.append(int(cube_name))
list_file = np.unique(list_file_all)

for f in list_file:
    filename = str(f)

    # load hyperspectral images
    data     = hdf5storage.loadmat(os.path.join(hyperimpath, 'raw_{}_rd_rf.mat'.format(filename)))
    hyper_im = data['data']
    hyper_im = hyper_im[:,:,np.where(goodWvlengthFlag == 1)[0]]
    label    = sio.loadmat(os.path.join(labelpath, 'ground_truth_{}.mat'.format(filename)))    

    map_target = np.zeros(np.shape(label['gt_final']), dtype = int)
    for i in range(1,7):
        map_target[np.where(label['gt_final'] == i)] = 1

    hyper_im = hyper_im*map_target[:,:,None]
    label_im = label['gt_final']*map_target
    
    spectra = np.reshape(hyper_im, [-1, np.shape(hyper_im)[2]])
    gt      = np.reshape(label_im, [-1, 1])
    
    idx_ntarget  = np.where(gt == 0)[0]
    spectra = np.delete(spectra, idx_ntarget, 0)
    gt      = np.delete(gt, idx_ntarget, 0)

    idx_all = np.array(range(0, np.shape(label_im)[0]*np.shape(label_im)[1]))
    idx_target = np.setdiff1d(idx_all, idx_ntarget)

    test_data = {'spectra': spectra,
                 'gt': gt,
                 'idx_target': idx_target,
                 'label_im': label_im}
    pickle.dump(test_data, open(os.path.join(savepath, 'test_data_{}.pkl'.format(filename)), 'wb'))

    