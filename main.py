# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:01:01 2019
Script of applying Siamese Network on DOE hyperspectral dataset
Reference:
    https://github.com/delijati/pytorch-siamese/blob/master/train_mnist.py
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
for n in paras['name_class']:
    idx                                = df_summary[df_summary['class {}'.format(n)] >0].index
    paras['filename{}'.format(n)] = [str(paras['filename'][i]) for i in idx]
    
list_file_all = {}
len_all = []
for classn in paras['name_class']:
    list_file_all[str(classn)] = paras['filename'+ str(classn)]
    len_all.append(len(list_file_all[str(classn)]))

    # "good" wavelengths flag 
flag             = sio.loadmat(os.path.join(paras['flagpath'], paras['flagname']), squeeze_me = True)
goodWvlengthFlag = flag['flag']
wavelength       = flag['wavelength']

    # patience for early stopping
patience = 30

	# fold for cross validation
idx_fold  = 0
count_all = 0
while all(l>0 for l in len_all) & count_all<10:

	savepath_fold =  paras['savepath']  + '/fold' + str(idx_fold)
	if not os.path.exists(savepath_fold):
		os.makedirs(savepath_fold)     
    
	early_stopping = trainlib.EarlyStopping(patience=patience, verbose=True)
    
	count_all += 1
	len_all = []

    if os.path.exists(os.path.join(savepath_fold, 'data.pkl')):
        paras    = pickle.load(open(os.path.join(savepath_fold, 'parameters.pkl'), 'rb'))
        all_data = pickle.load(open(os.path.join(savepath_fold, 'data.pkl'), 'rb'))
    else:
    	paras['selected_file'] = []
    	for idx, classn in enumerate(paras['name_class']):
    		list_file = list_file_all[str(classn)]
    		if idx >0:
    			list_file = list(set(list_file)-set(paras['selected_file'])) # if file has been selected before, regard the file for this time 
    		# selected_file = random.choice(list_file)
    		selected_file = random.sample(list_file, 3)
    		for f in selected_file:
    		    list_file_all[str(classn)].remove(f)
    		    paras['selected_file'].append(f)
    	    
    	for classn in paras['name_class']:
    		len_all.append(len(list_file_all[str(classn)]))    

    	for (idx, file) in enumerate(paras['selected_file']):
    		for (idx_c, classn) in enumerate(paras['name_class']):
    			if os.path.exists(os.path.join(paras['hyperpath'], 'raw_{}_{}.pkl'.format(file, classn))):
    				spectra_class = pickle.load(open(os.path.join(paras['hyperpath'], 'raw_{}_{}.pkl'.format(file, classn)), 'rb'))
    				spectra_class = spectra_class[:,np.where(goodWvlengthFlag == 1)[0]]
    				label_class   = np.ones(np.shape(spectra_class)[0])*classn           
    				if idx+idx_c == 0:
    					spectra = spectra_class
    					gt      = label_class
    				else:
    					spectra = np.concatenate((spectra, spectra_class), axis = 0)
    					gt      = np.concatenate((gt, label_class), axis = 0)	     

    	paras['num_class']  = len(np.unique(gt))
    	paras['inputsize']  = np.shape(spectra)[1]

        
    	print('Data loading done!!')

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
    		print('Downsampled those longer than 100000!\n')   
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
    		print('Downsampled those longer than 150000!\n')           
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
        pickle.dump(all_data, open(os.path.join(savepath_fold, 'data.pkl'), 'wb'))

################### model definition#################
	model = trainlib.SiameseNetwork(paras['inputsize'], paras['end_dim'])	

	    # loss function
	criterion = trainlib.ConstrastiveLoss()
	
	if torch.cuda.is_available():
		model.to(device)
		criterion.to(device)

	    # optimizer
	        # SGD
	#optimizer  = torch.optim.SGD(model.parameters(), lr = parameters.learning_rate, momentum = parameters.momentum)
	        # Adam
	optimizer = torch.optim.Adam(model.parameters(), lr = paras['learning_rate'])	    

        ## data train_lib
	train_iter              = trainlib.create_iterator(X_train, y_train, paras['name_class'])
	valid_iter, valid_iter0 = trainlib.create_iterator(X_valid, y_valid, paras['name_class'], return_all = True)

	train_loader  = DataLoader(train_iter, batch_size = paras['train_batch_size'], shuffle = True, num_workers = 0)
	valid_loader  = DataLoader(valid_iter, batch_size = paras['train_batch_size'], shuffle = True, num_workers = 0)
	valid_loader0 = DataLoader(valid_iter0, batch_size = paras['train_batch_size'], shuffle = False, num_workers = 0)

	# loss after each epoch
	loss_all = {'train':[], 'valid': []}
	accu_all= {'train': [], 'valid': []}

	model, loss_all, accu_all = trainlib.train(model, criterion, optimizer, paras, train_loader, valid_loader, loss_all, accu_all, early_stopping)

	## learning curve
	minposs   = loss_all['valid'].index(min(loss_all['valid'])) + 1
	fig1, axs = plt.subplots(1,2)
	list_label = ['train', 'validation']
	axs = axs.ravel()
	for (idx, tv) in enumerate(['train', 'valid']):
		lb = list_label[idx]
		axs[0].plot(range(1, len(loss_all[tv])+1), loss_all[tv])
		axs[1].plot(range(1, len(accu_all[tv])+1), accu_all[tv], label = lb)
	axs[0].axvline(minposs, linestyle = '--', color = 'r', label = 'Early Stopping Checkpoint')
	axs[1].legend(loc = 'best')
	axs[0].set_title('loss history')
	axs[0].set_xlabel('training epoch')
	axs[0].set_ylabel('loss')
	axs[1].set_title('accuracy history')
	axs[1].set_xlabel('training epoch')
	axs[1].set_ylabel('accuracy')
	plt.suptitle('Learning curve')
	plt.savefig(os.path.join(savepath_fold, '_train_hist_epoch.png'))
	plt.show()

	## test: transform all data at a time
	test_iter   = trainlib.create_iterator(X_test, y_test, paras['name_class'])
	test_loader = DataLoader(test_iter, batch_size = paras['train_batch_size'], shuffle = True, num_workers = 0)
	test_loss_temp = []
	test_accu_temp = []
	with torch.no_grad():    
		_, test_accu = trainlib.evaluate(test_loader, test_loss_temp, test_accu_temp, model, criterion, optimizer)

		print('=========================================================')
		print('Test accuracy {}\n'.format(test_accu))
    
	## make a transformation on the whole dataset
	train_iter_all   = trainlib.create_iterator_single(X_train, y_train)
	valid_iter_all   = trainlib.create_iterator_single(X_valid, y_valid)
	test_iter_all    = trainlib.create_iterator_single(X_test, y_test)
	train_loader_all = DataLoader(train_iter_all, batch_size = paras['train_batch_size'], shuffle = False, num_workers = 0)
	valid_loader_all = DataLoader(valid_iter_all, batch_size = paras['train_batch_size'], shuffle = False, num_workers = 0)
	test_loader_all  = DataLoader(test_iter_all, batch_size = paras['train_batch_size'], shuffle = False, num_workers = 0)

	with torch.no_grad():
		outputs_train, labels_train = tools.evaluate_single(train_loader_all, model)
		outputs_valid, labels_valid = tools.evaluate_single(valid_loader_all, model)
		outputs_test, labels_test   = tools.evaluate_single(test_loader_all, model)

	    ## train a knn classifier on train data
	classifier = None
	classifier = trainlib.run_classifier(classifier, 5, outputs_train, labels_train, paras, 'knn_train', idx_fold)
	_          = trainlib.run_classifier(classifier, 5, outputs_valid, labels_valid, paras, 'knn_valid', idx_fold)
	_          = trainlib.run_classifier(classifier, 5, outputs_test, labels_test, paras, 'knn_test', idx_fold)

	# classifier, predicted_vali0, accuracy_vali0, prob_vali0 = trainlib.knn_on_output(5, outputs_vali0, labels_vali0.ravel(), classifier, savepath_fold, 'knn_valiROC_accuracy')
	# labels_vali0_ = [parameters['grass_names'][i] for i in labels_vali0]
	# tools.plot_confu(labels_vali0_, predicted_vali0, savepath_fold, 'knn_valiROC') 
	# tools.ROC_classifier(parameters['name_class'], parameters['grass_names'], labels_vali0.ravel(), prob_vali0, savepath_fold, 'knn_valiROC')

	## ROC curves on classification result
	idx_fold += 1
	print('Fold ' + str(idx_fold) + ' finished!')
	print('====================================================')
