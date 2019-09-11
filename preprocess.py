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
from torch.utils.data import DataLoader
import lib.helper_funcs_Siamese_train as trainlib
# from lib.helper_funcs_Siamese_train import EarlyStopping
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
#from sklearn.metrics import roc_curve, auc
# from lib.tools import EarlyStopping
import lib.tools as tools
import random
from imblearn.over_sampling import SMOTE
import pandas as pd

import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


plt.close('all')
paras = trainlib.parameters

# extract list of files containing specific classes from the summary file
df_summary = pd.read_csv(os.path.join(paras['labelpath'], 'summary.csv'), index_col = False,  encoding = 'Latin-1')
paras['filename'] = df_summary[df_summary.columns[0]].tolist()
for classn in paras['name_class']:
#    pdb.set_trace()
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

#     # patience for early stopping
# patience = 30

    # fold for cross validation
idx_fold  = 0
count_all = 0
while all(l>0 for l in len_all): #& count_all<10:
    savepath_fold      =  paras['savepath']  + r'\fold' + str(idx_fold)
    savepath_data_fold = paras['savepath_data']  + r'\fold' + str(idx_fold)
    if not os.path.exists(savepath_fold):
        os.makedirs(savepath_fold)     
    if not os.path.exists(savepath_data_fold):
        os.makedirs(savepath_data_fold)
    
    
    count_all += 1
    len_all = []
    if not os.path.exists(os.path.join(savepath_data_fold, 'data.pkl')):
        paras['selected_file'] = []
        for idx, classn in enumerate(paras['name_class']):
            pdb.set_trace()
            list_file = list_file_all[str(classn)]
            if idx >0:
                list_file = list(set(list_file)-set(paras['selected_file'])) # if file has been selected before, regard the file for this time 
            # selected_file = random.choice(list_file)
            selected_file = random.sample(list_file, 2)
            # if the file has already been selected, remove it (it cannot be selected even for another class)
            for f in selected_file:
                list_file_all[str(classn)].remove(f)
                paras['selected_file'].append(f)
#            pdb.set_trace()
        for classn in paras['name_class']:
            len_all.append(len(list_file_all[str(classn)]))    
#        pdb.set_trace()
        spectra_all = {}
        label_all   = {}
        for (idxf, file) in enumerate(paras['selected_file']):
            for (idx_c, classn) in enumerate(paras['name_class']):
                if os.path.exists(os.path.join(paras['hyperpath'], 'raw_{}_{}.pkl'.format(file, classn))):
                    spectra_class = pickle.load(open(os.path.join(paras['hyperpath'], 'raw_{}_{}.pkl'.format(file, classn)), 'rb'))
                    spectra_class = spectra_class[:,np.where(goodWvlengthFlag == 1)[0]]
                    label_class   = np.ones(np.shape(spectra_class)[0])*classn 
#                    pdb.set_trace()
#                    if idx+idx_c == 0:
#                        spectra = spectra_class
#                        gt      = label_class
#                    else:
#                        spectra = np.concatenate((spectra, spectra_class), axis = 0)
#                        gt      = np.concatenate((gt, label_class), axis = 0)	
                else:
                    pdb.set_trace()
                if idxf == 0:
                    spectra_all[str(classn)] = spectra_class
                    label_all[str(classn)] = label_class  
                else:
                    spectra_all[str(classn)] = np.concatenate((spectra_all[str(classn)], spectra_class), axis = 0) 
                    label_all[str(classn)]   = np.concatenate((label_all[str(classn)], label_class), axis = 0) 
                  
        
        paras['num_class'] = len(spectra_all)
        paras['inputsize'] = np.shape(spectra_class)[1]
        pdb.set_trace()

        print('Data loading done!!')
        pdb.set_trace()
            # save parameters in a txt file and a pickle file 
        with open(os.path.join(savepath_fold, 'parameters.txt'), 'w') as f:
            for key, value in paras.items():
                f.write(key + ': ' + str(value) + '\n')
        f.close()
        pickle.dump(paras, open(os.path.join(savepath_fold, 'parameters.pkl'), 'wb'))

            ## train-test split
        X_train_all, X_test, y_train_all, y_test, idx_train, idx_test = train_test_split(spectra, gt, range(0, len(gt)), test_size = 0.1, random_state = 0)
        #	pdb.set_trace()
            ## train-validation split
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size = 0.11, random_state = 0)

        # if there is one class with number smaller than 100000, upsampling
        # if all class numbers greater than 150000, reduce to 100000 (not finished)
        v, count = np.unique(y_train, return_counts = True)  
        if len(np.where(count > 100000)[0]) > 0:
#            print('Downsampled those longer than 100000!\n')   
            for idx in np.where(count > 100000)[0]:
            #        idx = random.choice(np.where(count > 100000)[0])
                index = np.where(y_train == v[idx])[0]
                selected_id = np.array(random.sample(list(index), 100000))
                selected_X  = X_train[selected_id,:]
                selected_y  = y_train[selected_id]
                X_train = np.delete(X_train, index, axis = 0)
                y_train = np.delete(y_train, index)
                X_train = np.concatenate((X_train, selected_X), axis = 0)
                y_train = np.concatenate((y_train, selected_y), axis = 0)
            print('Downsampled those longer than 100000!\n')           
        if len(np.where(count < 100000)[0])>0:
            if len(np.where(count < 100000)[0])==2:  
                sm = SMOTE(random_state = 2, n_jobs = 16)
            else:
                sm = SMOTE(sampling_strategy = 'minority', random_state = 2, n_jobs = 16)
            print('Oversampled those longer than 100000!\n')               
            X_train, y_train = sm.fit_sample(X_train, y_train)
        else:
            print('All classes have greater than 100000 samples!')
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
    
    