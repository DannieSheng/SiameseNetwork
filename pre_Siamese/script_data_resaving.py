# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:09:44 2019
Save the hyperspectral data in a way that data from every class are saved together, ready for the Siamese network

@author: hdysheng
"""
import hdf5storage
import scipy.io as sio
import numpy as np
import os
import pickle

hyperpath = r'T:\AnalysisDroneData\ReflectanceCube\MATdataCube\CLMB STND 2019 Flight Data\100084_2019_06_25_16_39_57'

savepath = hyperpath.replace('ReflectanceCube\MATdataCube', 'dataPerClass')
if not os.path.exists(savepath):
    os.makedirs(savepath)

labelpath = hyperpath.replace('ReflectanceCube\MATdataCube', 'groundTruth')
labelpath = labelpath + '\\gt_processed'
list_file = [f for f in os.listdir(labelpath) if f.endswith('.mat')]

for idx_f, f in enumerate(list_file):
    label = sio.loadmat(os.path.join(labelpath, f))['gt_final']
    gt    = np.reshape(label, [-1, 1])

    fn = os.path.splitext(f)[0] 
    fn = fn.replace('ground_truth', 'raw')
    fn = fn + '_rd_rf.mat'

    hyper_im = hdf5storage.loadmat(os.path.join(hyperpath, fn))['data']
    spectra = np.reshape(hyper_im, [-1, np.shape(hyper_im)[2]])

    list_gt = np.unique(gt)

    for lb in list_gt:
        idx = np.where(np.squeeze(gt, axis=1) == lb)
        spectra_lb = spectra[idx[0],:]
        
        savename = fn.replace('_rd_rf.mat', '_' + str(lb) +'.pkl')
#        pdb.set_trace()
        with open(os.path.join(savepath, savename), 'wb') as fs:
            pickle.dump(spectra_lb, fs)
        fs.close()

        print('Image ' + f + ' label ' + str(lb) + ' done!!')
