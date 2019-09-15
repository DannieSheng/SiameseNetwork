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
from torch.utils.data import DataLoader
import lib.helper_funcs_Siamese_train as trainlib
import matplotlib.pyplot as plt
import pickle
#from sklearn.metrics import roc_curve, auc
import lib.earlystop as es
import lib.tools as tools

import pdb


device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

paras_exp = trainlib.parameters # parameters for Siamese network

    # patience for early stopping
patience = 30

for idx_fold in range(0,8):
        
    path_para_fold = paras_exp['path_all'] + r'\parameters\fold{}'.format(idx_fold)
    path_data_fold = paras_exp['path_all'] + r'\data\fold{}'.format(idx_fold)  
        # load pre-saved parameters
    paras      = pickle.load(open(os.path.join(path_para_fold, 'parameters.pkl'), 'rb'))
    parameters = {**paras_exp, **paras}
    
        # definition of save paths
    path_save_fold = parameters['path_all'] + r'\experiments\exp{}\{}\fold{}'.format(parameters['exp'], parameters['end_dim'], idx_fold)
    
    if not os.path.exists(path_save_fold):
        os.makedirs(path_save_fold)
        
    parameters['savepath_fold'] = path_save_fold
            
                    # save parameters in a txt file and a pickle file 
    with open(os.path.join(path_save_fold, 'parameters.txt'), 'w') as f:
        for key, value in parameters.items():
            f.write(key + ': ' + str(value) + '\n')
    f.close()
    pickle.dump(parameters, open(os.path.join(path_save_fold, 'parameters.pkl'), 'wb')) 
    
        # load pre-saved data
    all_data = pickle.load(open(os.path.join(path_data_fold, 'data.pkl'), 'rb'))
    X_train  = all_data['X_train']
    y_train  = all_data['y_train']
    X_valid  = all_data['X_valid']
    y_valid  = all_data['y_valid']
    X_test   = all_data['X_test']
    y_test   = all_data['y_test']

    #     # "good" wavelengths flag 
    # flag             = sio.loadmat(os.path.join(paras['flagpath'], paras['flagname']), squeeze_me = True)
    # goodWvlengthFlag = flag['flag']
    # wavelength       = flag['wavelength']

        # definition of early stopping
    if parameters['early_stop_mtd'] == 1:
        early_stopping = es.EarlyStopping1(patience=patience, verbose=True)
    elif parameters['early_stop_mtd'] == 2:
        early_stopping = es.EarlyStopping2(patience=patience, verbose=True)

################### model definition#################
    model = trainlib.SiameseNetwork(parameters['inputsize'], parameters['end_dim'])	

        # loss function
    criterion = trainlib.ConstrastiveLoss()

    if torch.cuda.is_available():
        model.to(device)
        criterion.to(device)

        # optimizer
            # SGD
    #optimizer  = torch.optim.SGD(model.parameters(), lr = parameters.learning_rate, momentum = parameters.momentum)
            # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr = parameters['learning_rate'])	    

        ## data train_lib
    train_iter              = trainlib.create_iterator(X_train, y_train, parameters['name_class'])
    valid_iter, valid_iter0 = trainlib.create_iterator(X_valid, y_valid, parameters['name_class'], return_all = True)

    train_loader  = DataLoader(train_iter, batch_size  = parameters['train_batch_size'], shuffle = True, num_workers = 0)
    valid_loader  = DataLoader(valid_iter, batch_size  = parameters['train_batch_size'], shuffle = True, num_workers = 0)
    valid_loader0 = DataLoader(valid_iter0, batch_size = parameters['train_batch_size'], shuffle = False, num_workers = 0)

    # loss after each epoch
    loss_all = {'train':[], 'valid': []}
    accu_all= {'train': [], 'valid': []}

    model, loss_all, accu_all = trainlib.train(model, criterion, optimizer, parameters, train_loader, valid_loader, loss_all, accu_all, early_stopping)

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
    plt.savefig(os.path.join(path_save_fold, '_train_hist_epoch.png'))
    plt.show()

    ## test: transform all data at a time
    test_iter   = trainlib.create_iterator(X_test, y_test, parameters['name_class'])
    test_loader = DataLoader(test_iter, batch_size = parameters['train_batch_size'], shuffle = True, num_workers = 0)
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
    train_loader_all = DataLoader(train_iter_all, batch_size = parameters['train_batch_size'], shuffle = False, num_workers = 0)
    valid_loader_all = DataLoader(valid_iter_all, batch_size = parameters['train_batch_size'], shuffle = False, num_workers = 0)
    test_loader_all  = DataLoader(test_iter_all, batch_size  = parameters['train_batch_size'], shuffle = False, num_workers = 0)

    with torch.no_grad():
        outputs_train, labels_train = tools.evaluate_single(train_loader_all, model)
        outputs_valid, labels_valid = tools.evaluate_single(valid_loader_all, model)
        outputs_test, labels_test   = tools.evaluate_single(test_loader_all, model)

        ## train a knn classifier on train data
    classifier = None
    classifier = trainlib.run_classifier(classifier, 5, outputs_train, labels_train, parameters, 'knn_train', idx_fold)
    _          = trainlib.run_classifier(classifier, 5, outputs_valid, labels_valid, parameters, 'knn_valid', idx_fold)
    _          = trainlib.run_classifier(classifier, 5, outputs_test, labels_test, parameters, 'knn_test', idx_fold)

    # classifier, predicted_vali0, accuracy_vali0, prob_vali0 = trainlib.knn_on_output(5, outputs_vali0, labels_vali0.ravel(), classifier, savepath_fold, 'knn_valiROC_accuracy')
    # labels_vali0_ = [parameters['grass_names'][i] for i in labels_vali0]
    # tools.plot_confu(labels_vali0_, predicted_vali0, savepath_fold, 'knn_valiROC') 
    # tools.ROC_classifier(parameters['name_class'], parameters['grass_names'], labels_vali0.ravel(), prob_vali0, savepath_fold, 'knn_valiROC')

    ## ROC curves on classification result
    idx_fold += 1
    print('Fold ' + str(idx_fold) + ' finished!')
    print('====================================================')
    pdb.set_trace()
