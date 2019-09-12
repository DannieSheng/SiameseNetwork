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
os.chdir('//ece-azare-nas1.ad.ufl.edu/ece-azare-nas/Profile/hdysheng/Documents/GitHub/SiameseNetwork')

from sklearn.linear_model import LinearRegression
import pickle
import lib.tools as tools
from torch.nn.utils import clip_grad_norm_
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters = {
    'exp': '1',     
#    'hyperpath': 'T:/Results/AnalysisDroneData/dataPerClass/CLMB STND 2019 Flight Data/100081_2019_06_11_17_57_06',
    'hyperpath': r'T:\AnalysisDroneData\dataPerClass\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06',
    # 'flagpath': r'T:\Results\AnalysisDroneData\ReflectanceCube\MATdataCube\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06',
    'flagname': 'flagGoodWvlen.mat',
    # 'labelpath': r'T:\Results\AnalysisDroneData\grounTruth\CLMB STND 2019 Flight Data\100081_2019_06_11_17_57_06\gt_processed',)
    'use_gt':1,
    'use_all_class':1, # indicator of "all classes" method or "one-vs.-all"
    # 'end_dim': 30,
    'normalization': 0,
    # 'train_batch_size': 128, # 16, 3, 13
    # 'valid_batch_size': 64,
    # 'train_num_epochs': 500,
    'margin': 1.0,
    'thres_dist': 0.5,
    'learning_rate': 5e-4,
    'name_class': [1, 2, 3, 4, 5, 6],
    'momentum': 0.9,
    'grass_names': ['Liberty', 'Blackwell', 'Alamo', 'Kanlow', 'CIR', 'Carthage']}
parameters['flagpath']      = parameters['hyperpath'].replace('dataPerClass', r'ReflectanceCube\MATdataCube')
parameters['labelpath']     = parameters['hyperpath'].replace('dataPerClass', 'grounTruth')
parameters['labelpath']     = parameters['labelpath'] + r'\gt_processed'
parameters['savepath_data'] = parameters['hyperpath'].replace('dataPerClass', 'Siamese')
if parameters['use_all_class'] == 1:
#    path = parameters['savepath_data'] + '/usegt/use_all_classes/'
    path = parameters['savepath_data'] + r'\use_all_classes'
else:
#    path = parameters['savepath_data'] + '/usegt/one_vs_all/'
    path = parameters['savepath_data'] + r'\one_vs_all'
if parameters['normalization'] == 1:
    parameters['savepath'] = path + r'normedSpectra\{}\exp{}'.format(parameters['end_dim'],parameters['exp'])
else:
    parameters['savepath'] = path + r'\{}\exp{}'.format(parameters['end_dim'], parameters['exp'])
parameters['savepath_data'] = parameters['savepath_data'] + r'\data\exp{}'.format(parameters['exp'])
if not os.path.exists(parameters['savepath_data']):
    os.makedirs(parameters['savepath_data'])
    
if not os.path.exists(parameters['savepath']):
    os.makedirs(parameters['savepath'])

def run_classifier(classifier, k, outputs, labels, parameters, save_name, idx_fold):
    savepath_fold = parameters['savepath']  + '/fold' + str(idx_fold)
    classifier, predicted, accuracy, prob = knn_on_output(k, outputs, labels, classifier, savepath_fold, save_name)
    pickle.dump(classifier, open(os.path.join(savepath_fold, 'classifier_'+str(idx_fold) + '.pkl'), 'wb'))
    labels_ = [parameters['grass_names'][int(i-1)] for i in labels]
    predicted_ = [parameters['grass_names'][int(i-1)] for i in predicted]
    tools.plot_confu(labels_, predicted_, savepath_fold, 'knn_training') 
    tools.ROC_classifier(parameters['name_class'], parameters['grass_names'], labels, prob, savepath_fold, 'knn_training')
    return classifier

def evaluate(d_loader, loss_temp, accu_temp, model, criterion, optimizer):
    for idx_batch, (x0, x1, validlabels) in enumerate(d_loader):
        validlabels = validlabels.float()
        if torch.cuda.is_available():
            validlabels = validlabels.to(device)            
            x0 = x0.to(device)
            x1 = x1.to(device)            
        outputs0, outputs1 = model(x0, x1)
        distances_valid     = model.predict(outputs0, outputs1)
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
    torch.save(model.state_dict(), os.path.join(parameters['savepath'], '_model.pth'))

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
    
def knn_on_output(k, outputs, labels, classifier = None, path_result = None, filename = None):
    if classifier is None:
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(outputs, labels)
    
    predicted = classifier.predict(outputs)
    prob      = classifier.predict_proba(outputs)

    accuracy  = (predicted == labels).mean() 
    
    if (path_result is not None) and (filename is not None):
        with open(os.path.join(path_result, 'accuracy_' + filename +'_' + str(k) + 'nn.txt'), 'w') as f:
            f.write('test accuracy for file ' + filename + ': ' + str(accuracy) + '\n')
    return classifier, predicted, accuracy, prob
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Using the slope of the fitted line as the measurement"""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.early_stop = False
        self.thres   = 0.0001
        self.y = np.zeros(patience)
        self.score = np.Inf
        self.count = 0
        self.x = np.arange(0,patience)

    def __call__(self, epoch, val_loss, model, end_dim):
        if self.count < self.patience:
            self.y[self.count] = val_loss
            self.count += 1
#            self.save_checkpoint(epoch, val_loss, model, end_dim)
            self.early_stop = False
        else:
#            pdb.set_trace()
            regressor = LinearRegression()  
            regressor.fit(np.reshape(self.x, (-1,1)), self.y)
            self.score = regressor.coef_
            if self.score > -self.thres:
                self.early_stop = True
                print(f'EarlyStopping epoch: {epoch}')
            else:
                self.early_stop = False
                self.save_checkpoint(epoch, val_loss, model, end_dim)
                self.count = 0
            

    def save_checkpoint(self, epoch, val_loss, model, end_dim):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Saving model ...')
#            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists('./' + str(end_dim)):
            os.makedirs('./' + str(end_dim))
        torch.save(model.state_dict(), './' + str(end_dim)+'/checkpoint.pt')
