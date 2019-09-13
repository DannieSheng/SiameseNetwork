# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:01:01 2019
Script to preprocess:
    Fold definition and save train, validation, and test data
@author: hdysheng
"""
import torch
import os
os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/GitHub/SiameseNetwork')
#import random
import numpy as np
# from lib.helper_funcs_Siamese_train import EarlyStopping
import scipy.io as sio
from sklearn.model_selection import train_test_split
import pickle
import random
from imblearn.over_sampling import SMOTE
import pandas as pd

import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters = {
    'hyperpath': r'T:\AnalysisDroneData\dataPerClass\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06',
    'flagname': 'flagGoodWvlen.mat',
    'use_all_class':1, # indicator of "all classes" method or "one-vs.-all"
    'name_class': [1, 2, 3, 4, 5, 6],
    'sample_per_class': 200000,
    'grass_names': ['Liberty', 'Blackwell', 'Alamo', 'Kanlow', 'CIR', 'Carthage']
}
parameters['flagpath']  = parameters['hyperpath'].replace('dataPerClass', r'ReflectanceCube\MATdataCube')
parameters['labelpath'] = parameters['hyperpath'].replace('dataPerClass', 'grounTruth')
parameters['labelpath'] = parameters['labelpath'] + r'\gt_processed'
path_temp               = parameters['hyperpath'].replace('dataPerClass', 'Siamese')
if parameters['use_all_class'] == 1:
    path = path_temp + r'\use_all_classes'
else:
    path = path_temp + r'\one_vs_all'
parameters['savepath_data'] = path + r'\data'
parameters['savepath_para'] = path + r'\parameters'

if not os.path.exists(parameters['savepath_data']):
    os.makedirs(parameters['savepath_data'])
if not os.path.exists(parameters['savepath_para']):
    os.makedirs(parameters['savepath_para'])

paras = parameters

# extract list of files containing specific classes from the summary file
df_summary        = pd.read_csv(os.path.join(paras['labelpath'], 'summary.csv'), index_col = False,  encoding = 'Latin-1')
paras['filename'] = df_summary[df_summary.columns[0]].tolist()
for classn in paras['name_class']:
    idx                                = df_summary[df_summary['class {}'.format(classn)] >0].index
    paras['filename{}'.format(classn)] = [str(paras['filename'][i]) for i in idx]
    
list_file_all = {}
len_all = []
for classn in paras['name_class']:
    list_file_all[str(classn)] = paras['filename'+ str(classn)]
    len_all.append(len(list_file_all[str(classn)]))

    # "good" wavelengths flag 
flag             = sio.loadmat(os.path.join(paras['flagpath'], paras['flagname']), squeeze_me = True)
goodWvlengthFlag = flag['flag']
wavelength       = flag['wavelength']

    # fold for cross validation
idx_fold  = 0
count_all = 0
while all(l>=2 for l in len_all): #& count_all<10:
    try:
        del spectra
    except:
        pass
    try:
        del gt
    except:
        pass
    savepath_data_fold = paras['savepath_data'] + r'\fold{}'.format(idx_fold)
    savepath_para_fold = paras['savepath_para'] + r'\fold{}'.format(idx_fold)
    if not os.path.exists(savepath_data_fold):
        os.makedirs(savepath_data_fold)
    if not os.path.exists(savepath_para_fold):
        os.makedirs(savepath_para_fold)     

    count_all += 1
    len_all = []
    if not os.path.exists(os.path.join(savepath_data_fold, 'data.pkl')):
        paras['selected_file'] = []
        for idx, classn in enumerate(paras['name_class']):
            list_file = list_file_all[str(classn)]
            if idx >0:
                list_file = list(set(list_file)-set(paras['selected_file'])) # if file has been selected before, regard the file for this time 
            # selected_file = random.choice(list_file)
            selected_file = random.sample(list_file, 2)
            # if the file has already been selected, remove it (it cannot be selected even for another class)
            for f in selected_file:
                list_file_all[str(classn)].remove(f)
                paras['selected_file'].append(f)
        for classn in paras['name_class']:
            len_all.append(len(list_file_all[str(classn)]))
        spectra_all = {}
        label_all   = {}
        for (idxf, file) in enumerate(paras['selected_file']):
            for (idx_c, classn) in enumerate(paras['name_class']):
                if os.path.exists(os.path.join(paras['hyperpath'], 'raw_{}_{}.pkl'.format(file, classn))):
                    spectra_class = pickle.load(open(os.path.join(paras['hyperpath'], 'raw_{}_{}.pkl'.format(file, classn)), 'rb'))
                    spectra_class = spectra_class[:,np.where(goodWvlengthFlag == 1)[0]]
                    label_class   = np.ones(np.shape(spectra_class)[0])*classn 
                    try:
                        spectra_all[str(classn)] = np.concatenate((spectra_all[str(classn)], spectra_class), axis = 0) 
                        label_all[str(classn)]   = np.concatenate((label_all[str(classn)], label_class), axis = 0) 
                    except:
                        spectra_all[str(classn)] = spectra_class
                        label_all[str(classn)] = label_class 

        paras['num_class'] = len(spectra_all)
        paras['inputsize'] = np.shape(spectra_class)[1]
        print('Data loading done!!')
        
                    # save parameters in a txt file and a pickle file 
        with open(os.path.join(savepath_para_fold, 'parameters.txt'), 'w') as f:
            for key, value in paras.items():
                f.write(key + ': ' + str(value) + '\n')
        f.close()
        pickle.dump(paras, open(os.path.join(savepath_para_fold, 'parameters.pkl'), 'wb'))
        
        # if class numbers greater than paras['sample_per_class'], reduce to paras['sample_per_class']
        for idx_c, classn in enumerate(paras['name_class']):
            num_sample = len(label_all[str(classn)])
#            pdb.set_trace()
            if num_sample > paras['sample_per_class']:
                selected_id = np.array(random.sample(list(np.arange(0, num_sample)), paras['sample_per_class']))
                selected_X  = spectra_all[str(classn)][selected_id,:]
                selected_y  = label_all[str(classn)][selected_id]
            else:
                selected_X = spectra_all[str(classn)]
                selected_y  = label_all[str(classn)]
            try:
                spectra = np.concatenate((spectra, selected_X), axis = 0) 
                gt      = np.concatenate((gt, selected_y), axis = 0)	
            except:
                spectra = selected_X
                gt      = selected_y
        print('Downsampled those with more than {} samples!\n'.format(paras['sample_per_class']))  
        
        # if there is one class with number smaller than 100000, upsampling
        v, count = np.unique(gt, return_counts = True)          
        if len(np.where(count < 150000)[0])>0:
            sm = SMOTE(sampling_strategy = 'not majority', random_state = 2, n_jobs = 16)
            print('Oversampled those longer than 150000!\n')               
            X_train, y_train = sm.fit_sample(spectra, gt)
        else:
            print('All classes have greater than {} samples!'.format(paras['sample_per_class']))

            ## train-test split
        X_train_all, X_test, y_train_all, y_test, idx_train, idx_test = train_test_split(spectra, gt, range(0, len(gt)), test_size = 0.1, random_state = 0)
        #	pdb.set_trace()
            ## train-validation split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size = 0.11, random_state = 0)

        # if there is one class with number smaller than 100000, upsampling
        # if all class numbers greater than 150000, reduce to 100000
        

        all_data = {'X_train': X_train,
                    'y_train': y_train,
                    'X_valid': X_valid,
                    'y_valid': y_valid,
                    'X_test': X_test,
                    'y_test': y_test}
        pickle.dump(all_data, open(os.path.join(savepath_data_fold, 'data.pkl'), 'wb'))

    idx_fold += 1
    print('Fold ' + str(idx_fold) + ' finished!')
    print('====================================================')
    pdb.set_trace()
    
    