# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:10:39 2019
Parameters and helper functions for the Siamese Network for the hyperspectral dataset
@author: hdysheng
"""
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier
import os

from sklearn.linear_model import LinearRegression
import pickle
import lib.tools as tools
from torch.nn.utils import clip_grad_norm_

import scipy.sparse as sp
from joblib import Parallel, delayed, cpu_count

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters = {
    'exp': '1',
    'end_dim': 30,
    'normalization': 0,
    'train_batch_size': 128, # 16, 3, 13
    'valid_batch_size': 64,
    'train_num_epochs': 500,
    'margin': 1.0,
    'thres_dist': 0.5,
    'learning_rate': 5e-4,
    'momentum': 0.9,
    'hyperpath': r'T:\AnalysisDroneData\dataPerClass\CLMB STND 2019 Flight Data\100085_2019_07_18_15_54_58',
    'use_all_class':1, # indicator of "all classes" method or "one-vs.-all"
    'early_stop_mtd': 1 #1: the one from internet, 2: the one created by my self
    }
if parameters['use_all_class'] == 1:
    parameters['path_all'] = parameters['hyperpath'].replace('dataPerClass', 'Siamese') + r'\use_all_class'
else:
    pass

def run_classifier(classifier, k, outputs, labels, parameters, save_name, idx_fold):
    classifier, predicted, accuracy, prob = knn_on_output(k, outputs, labels, classifier, parameters['savepath_fold'], save_name)
    if 'train' in save_name:
        pickle.dump(classifier, open(os.path.join(parameters['savepath_fold'], 'classifier_'+str(idx_fold) + '.pkl'), 'wb'))
    labels_ = [parameters['grass_names'][int(i-1)] for i in labels]
    predicted_ = [parameters['grass_names'][int(i-1)] for i in predicted]
    tools.plot_confu(labels_, predicted_, parameters['savepath_fold'], save_name, parameters['grass_names']) 
    tools.ROC_classifier(parameters['name_class'], parameters['grass_names'], labels, prob, parameters['savepath_fold'], save_name)
    return classifier

def _predict(estimator, X, method, start, stop):
    return getattr(estimator, method)(X[start:stop])
def parallel_predict(estimator, X, n_jobs=1, method='predict_proba', batches_per_job=3):
    n_jobs = max(cpu_count() + 1 + n_jobs, 1)  # XXX: this should really be done by joblib
    n_batches  = batches_per_job * n_jobs
    n_samples = len(X)
    batch_size = int(np.ceil(n_samples / n_batches))
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(delayed(_predict)(estimator, X, method, i, i + batch_size)
                       for i in range(0, n_samples, batch_size))
    if sp.issparse(results[0]):
        return sp.vstack(results)
    return np.concatenate(results)

def knn_on_output(k, outputs, labels, classifier = None, path_result = None, filename = None):
    if classifier is None:
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(outputs, labels)
    # prob      = classifier.predict_proba(outputs)
    prob = parallel_predict(classifier, outputs, n_jobs=1, method='predict_proba', batches_per_job=100)
#    predicted = classifier.predict(outputs)

    predicted = np.argmax(prob, axis = 1) + 1

    accuracy  = (predicted == labels).mean() 
    
    if (path_result is not None) and (filename is not None):
        with open(os.path.join(path_result, 'accuracy_' + filename +'_' + str(k) + 'nn.txt'), 'w') as f:
            f.write('test accuracy for file ' + filename + ': ' + str(accuracy) + '\n')
    return classifier, predicted, accuracy, prob

def evaluate(d_loader, loss_temp, accu_temp, model, criterion, optimizer):
    for idx_batch, (x0, x1, validlabels) in enumerate(d_loader):
        validlabels = validlabels.float()
        if torch.cuda.is_available():
            validlabels = validlabels.to(device)       
            model.to(device)
            x0 = x0.to(device)
            x1 = x1.to(device)            
        outputs0, outputs1 = model(x0, x1)
        distances_valid    = model.predict(outputs0, outputs1)
        loss_temp.append(criterion(outputs0, outputs1, validlabels).detach().cpu().numpy())
        accu_temp.append(tools.compute_accuracy(validlabels.detach().cpu().numpy(), distances_valid.detach().cpu().numpy(), parameters['thres_dist']))
    loss_epoch = np.average(loss_temp)
    accu_epoch = np.average(accu_temp)
    return loss_epoch, accu_epoch


def train(model, criterion, optimizer, parameters, train_loader, valid_loader, loss_all, accu_all, early_stopping):
    for idx_epoch in range(0, parameters['train_num_epochs']):
        model.train()
        # loss_temp and accu_temp: loss and accuracy recorded for every epoch
        loss_temp = {'train': [], 'valid':[]}
        accu_temp = {'train': [], 'valid': []}

        for idx_batch, (x0, x1, trainlabel) in enumerate(train_loader):
            trainlabel = trainlabel.float()
            if torch.cuda.is_available():
                x0 = x0.to(device)
                x1 = x1.to(device)
                trainlabel = trainlabel.to(device)
                model.to(device)
                criterion.to(device)

            output0, output1 = model(x0, x1)
            loss             = criterion(output0, output1, trainlabel)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            loss_temp['train'].append(loss.detach().cpu().numpy())
            distances = model.predict(output0, output1)
            accu      = tools.compute_accuracy(trainlabel.detach().cpu().numpy(), distances.detach().cpu().numpy(), parameters['thres_dist'])
            accu_temp['train'].append(accu)

            if idx_batch%500 == 0:
                print('=========================================================')
                print('Epoch number {}\n Current batch {}\n Current loss {}\n'.format(idx_epoch+1, idx_batch+1, loss.item()))
                print('Current accuracy {}\n'.format(accu))

        loss_epoch, accu_epoch = np.average(loss_temp['train']), np.average(accu_temp['train'])
        loss_all['train'].append(loss_epoch)
        accu_all['train'].append(accu_epoch)

        # validation
        with torch.no_grad():
            loss_temp['valid'], accu_temp['valid'] = evaluate(valid_loader, loss_temp['valid'], accu_temp['valid'], model, criterion, optimizer)
            loss_epoch_valid, accu_epoch_valid = np.average(loss_temp['valid']), np.average(accu_temp['valid'])
            loss_all['valid'].append(loss_epoch_valid)
            accu_all['valid'].append(accu_epoch_valid)

            # early-stopping
            if idx_epoch >= 180:
                early_stopping(idx_epoch, loss_epoch_valid, model, parameters['end_dim'])
                if early_stopping.early_stop:
                    print("Early stopping at epoch " + str(idx_epoch))
                    break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./' + str(parameters['end_dim'])+'/checkpoint.pt'))

    # save the final model
    torch.save(model.state_dict(), os.path.join(parameters['savepath_fold'], '_model.pth'))

    return model, loss_all, accu_all



  ## dataset for dataloader
  # used for paired input
class SiameseDataset(Dataset):
    
    def __init__(self, x0, x1, label):
        self.size  = label.shape[0]
        self.x0    = torch.from_numpy(x0)
        self.x1    = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return(self.x0[index],
               self.x1[index],
               self.label[index])
        
    def __len__(self):
        return self.size

  # used for single input
class SiameseDataset_single(Dataset):
    
    def __init__(self, x, label):
        self.size  = label.shape[0]
        self.x    = torch.from_numpy(x)
        self.label = torch.from_numpy(label)
    
    def __getitem__(self, index):
        return(self.x[index],
               self.label[index])
        
    def __len__(self):
        return self.size

	## function to create pairs for Siamese Network    
#def create_pairs(data, digit_indices):
def create_pairs(data, digit_indices, name_class, return_all = False):

    x0_data = []
    x1_data = []
    label   = []
    if return_all is True:
        label_all = []
        data_all  = []
        label_all1 = []
        label_all2 = []

#    n = min([len(digit_indices[d]) for d in range(Config.num_class)])-1 # get the minimum sample size from n classes
    n = min([len(digit_indices[d]) for d in range(len(name_class))])-1 # get the minimum sample size from n classes
#    n = round(n/len(parameters['filename']))

    for d in range(len(name_class)):
        for i in range(n):
            
            # generate pairs from same class: label 0
            z0, z1 = digit_indices[d][i], digit_indices[d][i+1]
            # normalization to be added
            x0_data.append(data[z0])
            x1_data.append(data[z1])
            label.append(0)
            if return_all is True:
                label_all1.append(name_class[d])
                label_all2.append(name_class[d])    
            
            # generate pairs from different classes: label 1
#            inc    = random.randrange(1,Config.num_class)
            inc    = random.randrange(1,len(name_class))
#            dn     = (d+inc)%Config.num_class
            dn     = (d+inc)%len(name_class)
            z0, z1 = digit_indices[d][i], digit_indices[dn][i]
            x0_data.append(data[z0])
            x1_data.append(data[z1])
            label.append(1)
            if return_all == True:  
                label_all1.append(name_class[d])
                label_all2.append(name_class[dn])

    x0_data = np.array(x0_data, dtype = np.float32)
    x1_data = np.array(x1_data, dtype = np.float32)
    label   = np.array(label, dtype = np.int32)
    if return_all == True:
        label_all1 = np.array(label_all1, dtype = np.int32)
        label_all2 = np.array(label_all2, dtype = np.int32)
        data_all = np.reshape(np.concatenate((x0_data, x1_data), axis = 0), (-1,x0_data.shape[1]))
        label_all = np.reshape(np.concatenate((label_all1, label_all2), axis = 0), (data_all.shape[0], -1))
    if return_all == True:
        return x0_data, x1_data, label, data_all, label_all
    else:
        return x0_data, x1_data, label

	## function to create iterable objects 
	# used for paired inputs
def create_iterator(data, label, name_class, shuffle = False, return_all = False):
#def create_iterator(data, label, name_class, shuffle = True):
#    digit_indices = [np.where(label == i)[0] for i in range(Config.num_class)]
#    digit_indices = [np.where(label == i)[0] for i in np.nditer(name_class)]
    digit_indices = [np.where(label == i)[0] for i in name_class]
    
    if return_all == True:
        x0, x1, label, data_all, label_all = create_pairs(data, digit_indices, name_class, return_all = True)
        ret_all = create_iterator_single(data_all, label_all)
    else:
        x0, x1, label = create_pairs(data, digit_indices, name_class, return_all = False)
    ret           = SiameseDataset(x0, x1, label)
    if return_all == True:
        return ret, ret_all
    else:
        return ret

    # used for single inputs
def create_iterator_single(data, label, shuffle = False):
#def create_iterator_single(data, label, shuffle = True):
    x  = []
    label_ = []
    for i in range(len(label)):
        x.append(data[i])
        label_.append(label[i])
    x  = np.array(x, dtype = np.float32)
    label_ = np.array(label_, dtype = np.float32)
    ret    = SiameseDataset_single(x, label_)
    return ret

    ## loss function: contrastive loss
class ConstrastiveLoss(nn.Module):
    def __init__(self, margin = parameters['margin']):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_constrastive  = torch.mean((1-label)*torch.pow(euclidean_distance, 2)+
                                        (label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min = 0.0),2))
        return loss_constrastive

    ## Siamese network definition
class SiameseNetwork(nn.Module):
    def __init__(self, inputsize, end_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1        = nn.Linear(inputsize, 180)
        self.fc2        = nn.Linear(180, 110)
        self.fc3        = nn.Linear(110, 55)
        self.fc4        = nn.Linear(55, 30)
        self.fc5        = nn.Linear(30, 15)
        self.fc6        = nn.Linear(15, 8)
        self.fc7        = nn.Linear(8, 4)
        self.fc8        = nn.Linear(4, end_dim)
        self.dropout    = nn.Dropout(0.1)
        
#        self.activation = nn.ReLU(inplace = False)
        self.activation = nn.PReLU()
        
        # forward pass for single input
    def forward_once(self, x):
#        output  = F.relu(self.fc1(x))
#        output  = self.dropout(output)
#        output  = F.relu(self.fc2(output))
#        output  = self.dropout(output)
#        output  = self.fc3(output)
        
        output = self.fc1(x)
        
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc3(output)
  
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc4(output)
#        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc5(output)
        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc6(output)
#        
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc7(output)
#    
#        output = self.activation(output)
#        output = self.dropout(output)
#        output = self.fc8(output)
        
        return output
    
        # forward pass for the whole network
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
        
        # predict 
    def predict(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return euclidean_distance
    

    
