# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:29:12 2019
Script for applying trained Simese model on new data
@author: hdysheng
"""

import os
os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/Python Scripts/DOEdrone/Siamese/SiameseUpdated/final_code - Copy') # set work directory

import lib.helper_funcs_Siamese_train as trainlib
import lib.helper_funcs_Siamese_test as testlib
import lib.tools as tools
import hdf5storage
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import random
import pdb

plt.close('all')
    ## assign parameters
#parameters = lib.parameters
#parameters['modelpath'] = 'T:/Results/Analysis CLMB 2018 drone data/orthorectification/Hyperspectral_reflectance/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/parameter1/Siamese/usegt/' + str(parameters['end_dim']) +'/exp' + str(parameters['exp']) + '/'

exp = '5'
end_dim = 30
normalization = 1
use_all_classes = 1 
MAXFOLD = 2


if use_all_classes == 1:
    modelpath = 'T:/Results/Analysis CLMB 2018 drone data/Siamese/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/usegt/use_all_classes/'
else: 
    modelpath = 'T:/Results/Analysis CLMB 2018 drone data/Siamese/Maria_Bradford_Switchgrass_Standplanting/100071_2018_10_31_16_29_53/usegt/one_vs_all/'
if normalization == 1:
    modelpath = modelpath + 'normedSpectra/' + str(end_dim) + '/exp' + exp
else:
    modelpath = modelpath + str(end_dim) + '/exp' + exp
    

for idx_fold in range(0, MAXFOLD):
    parameters  = pickle.load(open(os.path.join(modelpath+ '/fold' + str(idx_fold), 'parameters.pkl'), 'rb'))

    parameters['modelpath'] = modelpath
    
    parameters['modelname'] = '_model.pth'

    parameters['classifiername'] = 'classifier_' + str(idx_fold) + '.pkl'

    list_all   = parameters['filename']
    list_train = parameters['selected_file']
    list_file  = list(set(list_all) - set(list_train))

    flag             = sio.loadmat(os.path.join(parameters['flagpath'], parameters['flagname']), squeeze_me = True)
    goodWvlengthFlag = flag['goodWvlengthFlag']

	## load trained model
    model = testlib.SiameseNetwork(parameters['inputsize'], parameters['end_dim'])
    try:
        model.load_state_dict(torch.load(os.path.join(parameters['modelpath'], parameters['modelname'])))
    except:
        model.load_state_dict(torch.load(os.path.join(parameters['modelpath'] + '/fold' + str(idx_fold), parameters['modelname'])))    
        parameters['modelpath'] = parameters['modelpath'] + '/fold'+ str(idx_fold)
    
    test_savepath = parameters['modelpath']
    
	## load the classifier
    with open(os.path.join(parameters['modelpath'], 'classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    f.close()

	########

	#     ## get the list of all files
    hyperim_path = parameters['hyperpath'].replace('per_class', 'newflag')
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
        filename = str(f)
	#    pdb.set_trace()
	#    for (idx_c, classn) in enumerate(parameters['name_class']):
	#        spectra_class = pickle.load(open(os.path.join(parameters['hyperpath'], 'mappedhyper_' + filename + '_' + str(int(classn)) + '.pkl'), 'rb'))
	#        index_class   = pickle.load(open(os.path.join(parameters['hyperpath'], 'mappedhyper_' + filename + '_' + str(int(classn)) + '_index.pkl'), 'rb'))
	#        pdb.set_trace()

        data             = hdf5storage.loadmat(os.path.join(hyperim_path, 'mappedhyper_'+filename+'.mat'))
        hyper_im         = data['hyper_im']
        hyper_im         = hyper_im[:,:,np.where(goodWvlengthFlag == 1)[0]]
        label            = sio.loadmat(os.path.join(parameters['labelpath'], 'ground_truth_'+filename+'.mat'))
	    
        map_target = np.zeros(np.shape(label['gt']), dtype = int)
        for i in parameters['name_class']:
            map_target[np.where(label['gt'] == i)] = 1
	        
        yper_im = hyper_im*map_target[:,:,None]
        label_im = label['gt']*map_target
        plt.imshow(label_im)

        spectra          = np.reshape(hyper_im, [-1, np.shape(hyper_im)[2]])
        gt               = np.reshape(label_im, [-1, 1])
	    
        idx_ntarget  = np.where(gt == 0)[0]
        spectra = np.delete(spectra, idx_ntarget, 0)
        gt      = np.delete(gt, idx_ntarget, 0)
	    
        idx_all = np.array(range(0, np.shape(label_im)[0]*np.shape(label_im)[1]))
        idx_target = np.setdiff1d(idx_all, idx_ntarget)

	    ## visualization of original spectra
	#    rand_idx = lib.plotspectra(parameters, spectra, gt, filename)

	    ## make a transformation on the whole dataset
        data_iter   = trainlib.create_iterator_single(spectra, gt)
        data_loader = DataLoader(data_iter, batch_size = len(spectra), shuffle = False, num_workers = 0)
	 
	    
        for idx, (inputs, labels) in enumerate(data_loader):
            labels_  = labels.numpy()
            outputs_ = model.forward_once(inputs).detach().numpy()
            output_tosave = {
	                'output': outputs_,
	                'label': labels_,
	                'index_in_im': idx_target}    

	    
        classifier, output_tosave['predicted'], output_tosave['accuracy'], prob = tools.knn_on_output(5, np.nan_to_num(outputs_), np.squeeze(labels_, axis = 1), classifier, test_savepath, filename)
#        prob_ = np.zeros((len(prob), 3))
#        if prob.shape[1] <3:
#            list_lb = np.unique(labels_)
#            for idx, lb in enumerate(list_lb):
#                prob_[:,int(lb)-1] = prob[:, idx]
#        else:
#            prob_ = prob
        f = open(os.path.join(test_savepath, filename+'_results.pkl'), 'wb')
        pickle.dump(output_tosave, f)
        f.close()
	    

	        ## plot confusion matrix
        tools.plot_confu(np.squeeze(output_tosave['label'], axis = 1), output_tosave['predicted'], test_savepath, filename) 
	        ## show the output in image
        tools.show_gt_output(idx_target, output_tosave['label'], output_tosave['predicted'], label_im, filename, test_savepath)

	    ## visualization of output
        tools.output_visualize(parameters, output_tosave['output'], output_tosave['label'], filename, test_savepath)
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
    tools.ROC_classifier(parameters['name_class'], np.squeeze(label_all, axis = 1), prob_all, test_savepath, 'overall_knn')
    
        ## plot overall confusion matrix
    tools.plot_confu(np.squeeze(label_all, axis = 1), predicted_all, test_savepath, 'overall') 

