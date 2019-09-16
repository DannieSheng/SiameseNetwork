# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:29:12 2019
Script for applying trained Simese model on new data
@author: hdysheng
"""

import os
os.chdir(r'\\ece-azare-nas1.ad.ufl.edu\ece-azare-nas\Profile\hdysheng\Documents\GitHub\SiameseNetwork') # set work directory

import lib.helper_funcs_Siamese_train as trainlib
import lib.helper_funcs_Siamese_test as testlib
import lib.tools as tools
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import re
import pdb

plt.close('all')
device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_all = r'T:\AnalysisDroneData\Siamese\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06'

NUMFOLD       = 1
exp           = 1
end_dim       = 30
use_all_class = 1
if use_all_class == 1:
    path_exp  = r'{}\use_all_class\experiments\exp{}\{}'.format(path_all, exp, end_dim)
path_testdata = r'{}\use_all_class\testdata'.format(path_all)

for idx_fold in range(0, NUMFOLD):
    path_fold  = path_exp + r'\fold{}'.format(idx_fold)
    parameters = pickle.load(open(os.path.join(path_fold, 'parameters.pkl'), 'rb'))
    list_file = os.listdir(path_testdata)

    ## load trained model
    model = testlib.SiameseNetwork(parameters['inputsize'], end_dim)
    model.load_state_dict(torch.load(os.path.join(path_fold, '_model.pth')))
    
    ## load the classifier
    with open(os.path.join(path_fold, 'classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    f.close()

    ## loop over all files
    for f in list_file:
        plt.close('all')
        filename = re.findall('\d+',f)[0]

        test_data  = pickle.load(open(os.path.join(path_testdata, f), 'rb'))
        spectra    = test_data['spectra']
        gt         = test_data['gt']
        idx_target = test_data['idx_target']
        label_im   = test_data['label_im']

        ## make a transformation on the whole dataset
        data_iter   = trainlib.create_iterator_single(spectra, gt)
#        data_loader = DataLoader(data_iter, batch_size = len(spectra), shuffle = False, num_workers = 0)
        data_loader = DataLoader(data_iter, batch_size = parameters['train_batch_size'], shuffle = False, num_workers = 0)
        
        with torch.no_grad():
            outputs_, labels_ = tools.evaluate_single(data_loader, model)

        output_tosave = {
                'output': outputs_,
                'label': labels_,
                'index_in_im': idx_target}
    
        classifier, output_tosave['predicted'], output_tosave['accuracy'], prob = trainlib.knn_on_output(5, np.nan_to_num(outputs_), np.squeeze(labels_, axis = 1), classifier, path_fold, filename)

        f = open(os.path.join(path_fold, filename+'_results.pkl'), 'wb')
        pickle.dump(output_tosave, f)
        f.close()

            ## plot confusion matrix
        tools.plot_confu(np.squeeze(output_tosave['label'], axis = 1), output_tosave['predicted'], path_fold, filename, parameters['grass_names']) 

            ## show the output in image
        tools.show_gt_output(idx_target, output_tosave['label'], output_tosave['predicted'], label_im, filename, path_fold)

        ## visualization of output
#        tools.output_visualize(parameters, output_tosave['output'], output_tosave['label'], filename, test_savepath)
        print(filename + ' done!!')
        print('==================================================================')
        pdb.set_trace()
        try:
            label_all = np.concatenate((label_all, labels_), axis = 0)
            prob_all = np.concatenate((prob_all, prob), axis = 0)
            predicted_all = np.concatenate((predicted_all, output_tosave['predicted']), axis = 0)
        except:
            label_all = labels_
            prob_all  = prob
            predicted_all = output_tosave['predicted']

         ## plot overall ROC curves on classification result for all test data
    tools.ROC_classifier(parameters['name_class'], np.squeeze(label_all, axis = 1), prob_all, path_fold, 'overall_knn')
    
        ## plot overall confusion matrix
    tools.plot_confu(np.squeeze(label_all, axis = 1), predicted_all, path_fold, 'overall', parameters['grass_names']) 

