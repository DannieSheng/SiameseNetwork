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

NUMFOLD      = 1
exp          = 0
end_dim      = 30
device       = torch.device("cuba:0" if torch.cuda.is_available() else "cpu")
# path_exp     = r'T:\AnalysisDroneData\Siamese\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06\use_all_class\experiments\exp{}\{}'.format(exp, end_dim)
path_exp      = r'T:\AnalysisDroneData\Siamese\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06\use_all_class\30\exp0'
path_testdata = r'T:\AnalysisDroneData\Siamese\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06\use_all_class\testdata'

for idx_fold in range(0, NUMFOLD):
    path_fold  = path_exp + r'\fold{}'.format(idx_fold)
    parameters = pickle.load(open(os.path.join(path_fold, 'parameters.pkl'), 'rb'))
    # parameters['modelpath'] = modelpath
    
    # parameters['modelname'] = '_model.pth'

    # parameters['classifiername'] = 'classifier_' + str(idx_fold) + '.pkl'
    # list_all   = parameters['filename']
    # list_train = parameters['selected_file']
    # list_file  = list(set(list_all) - set(list_train))
    # list_file  = list_all
    list_file = os.listdir(path_testdata)

    ## load trained model
    model = testlib.SiameseNetwork(parameters['inputsize'], end_dim)
    model.load_state_dict(torch.load(os.path.join(path_fold, '_model.pth')))

    test_savepath = parameters['savepath']
    test_savepath = test_savepath.replace('use_all_classes', 'use_all_class')
    
    ## load the classifier
    with open(os.path.join(path_fold, 'classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    f.close()

    ## get the list of all files
    # hyperim_path = parameters['hyperpath'].replace('per_class', 'newflag')
	# list_file_temp = [f for f in os.listdir(hyperim_path) if f.endswith('.mat') and f.startswith('mapped')]

	#     ## get the correct order of files
	# list_file = []
	# for f in list_file_temp:
	#     cube_name = re.findall('\d+', f)[0]
	#     list_file.append(int(cube_name))
	# index_temp = np.argsort(list_file)
	# list_file      = [list_file[i] for i in index_temp]
	# list_file_temp = [list_file_temp[i] for i in index_temp]
	# list_file = [list_file[i] for i in [0, 3, 4, 11, 15]]

	    ## loop over all files
    
    for f in list_file:
        plt.close('all')
        filename = re.findall('\d+',f)[0]

        test_data  = pickle.load(open(os.path.join(path_testdata, f), 'rb'))
        spectra    = test_data['spectra']
        gt         = test_data['gt']
        idx_target = test_data['idx_target']
        label_im   = test_data['label_im']
        
        savepath_fold = test_savepath + r'\fold{}'.format(idx_fold)
        
	#    pdb.set_trace()
	#    for (idx_c, classn) in enumerate(parameters['name_class']):
	#        spectra_class = pickle.load(open(os.path.join(parameters['hyperpath'], 'mappedhyper_' + filename + '_' + str(int(classn)) + '.pkl'), 'rb'))
	#        index_class   = pickle.load(open(os.path.join(parameters['hyperpath'], 'mappedhyper_' + filename + '_' + str(int(classn)) + '_index.pkl'), 'rb'))
	#        pdb.set_trace()

        # data     = hdf5storage.loadmat(os.path.join(path_hyperim, 'raw_{}_rd_rf.mat'.format(filename)))
        # hyper_im = data['hyper_im']
        # hyper_im = hyper_im[:,:,np.where(goodWvlengthFlag == 1)[0]]
        # label    = sio.loadmat(os.path.join(parameters['labelpath'], 'ground_truth_{}.mat'.format(filename)))

        # map_target = np.zeros(np.shape(label['gt']), dtype = int)
        # for i in parameters['name_class']:
        #     map_target[np.where(label['gt'] == i)] = 1

        # hyper_im = hyper_im*map_target[:,:,None]
        # label_im = label['gt']*map_target
        # plt.imshow(label_im)

        # spectra          = np.reshape(hyper_im, [-1, np.shape(hyper_im)[2]])
        # gt               = np.reshape(label_im, [-1, 1])

        # idx_ntarget  = np.where(gt == 0)[0]
        # spectra = np.delete(spectra, idx_ntarget, 0)
        # gt      = np.delete(gt, idx_ntarget, 0)
	    
        # idx_all = np.array(range(0, np.shape(label_im)[0]*np.shape(label_im)[1]))
        # idx_target = np.setdiff1d(idx_all, idx_ntarget)

	    ## visualization of original spectra
	#    rand_idx = lib.plotspectra(parameters, spectra, gt, filename)

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
    
        classifier, output_tosave['predicted'], output_tosave['accuracy'], prob = testlib.knn_on_output(5, np.nan_to_num(outputs_), np.squeeze(labels_, axis = 1), classifier, savepath_fold, filename)

        f = open(os.path.join(savepath_fold, filename+'_results.pkl'), 'wb')
        pickle.dump(output_tosave, f)
        f.close()
	    
	        ## plot confusion matrix
        tools.plot_confu(np.squeeze(output_tosave['label'], axis = 1), output_tosave['predicted'], savepath_fold, filename, parameters['grass_names']) 
	        
            ## show the output in image
        tools.show_gt_output(idx_target, output_tosave['label'], output_tosave['predicted'], label_im, filename, savepath_fold)

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
    tools.ROC_classifier(parameters['name_class'], np.squeeze(label_all, axis = 1), prob_all, savepath_fold, 'overall_knn')
    
        ## plot overall confusion matrix
    tools.plot_confu(np.squeeze(label_all, axis = 1), predicted_all, savepath_fold, 'overall', parameters['grass_names']) 

