# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:38:12 2019

@author: hdysheng
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## function to compute the accuracy
def compute_accuracy(y_true, y_pred, thres):
    '''Compute classification accuracy with a fixed threshold on distances
    '''
    pred = y_pred.ravel() > thres
    return np.mean(pred == y_true)

    ## function to generate scatter plots
#def scatterplot(outputs, labels, name_class, name_title):
#    fig = plt.figure()
#    if parameters['end_dim'] !=2:
#        ax1 = fig.add_subplot(1,1,1, projection = '3d')
#    else:
#        ax1 = fig.add_subplot(1,1,1)
#    for classname in name_class:
#        outputs_i = outputs[np.where(labels == classname)[0],:]
#        if parameters['end_dim']!=2:
#            ax1.plot(outputs_i[:,0], outputs_i[:,1], outputs_i[:,2],'.', label = str(classname))
#        else:
#            ax1.plot(outputs_i[:,0], outputs_i[:,1], '.', label = str(classname))
#    ax1.legend(loc = 'best')
#    plt.title('Scatter plot for '+name_title)
#    plt.savefig(os.path.join(parameters['savepath'], 'scatter_' + name_title+'.jpg'))
#    plt.show() 

def evaluate_single(d_loader):
    outputs_all = []
    labels_all  = []
    for idx_batch, (inputs, labels) in enumerate(d_loader):        
        labels  = labels.float()
        if torch.cuda.is_available():
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs = model.forward_once(inputs).detach().cpu().numpy()
        else:
            outputs = model.forward_once(inputs)
        labels_all.append(labels)
        outputs_all.append(outputs)
    return outputs_all, labels_all

def plot_confu(labels, predicted, path_result, filename):
    
    confu = confusion_matrix(labels, predicted, labels=None, sample_weight=None)
    confu_percent = confu / confu.astype(np.float).sum(axis=1)
    df_cm = pd.DataFrame(confu_percent)
    
        # plot confusion matrix
    fig101, axs = plt.subplots(1,1)
    sn.heatmap(df_cm, annot = True, ax = axs)
    plt.title('Confusion matrix')
    plt.savefig(os.path.join(path_result, 'cM_' + filename + '.png'))  
    
def ROC_classifier(name_class, name_grass, labels, score, path_result, filename):
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    fig102, axs = plt.subplots(1,1)
    labels = label_binarize(labels, classes = name_class)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, color in zip(name_class, colors):
        fpr[i-1], tpr[i-1], _ = roc_curve(labels[:,i-1], score[:,i-1])
        roc_auc[i-1] = auc(fpr[i-1], tpr[i-1])
        axs.plot(fpr[i-1], tpr[i-1], lw = 2, label = 'ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(name_grass[i-1], roc_auc[i-1]))
    plt.legend(loc="lower right")
    plt.title('ROC curves for classification result')
    plt.savefig(os.path.join(path_result, 'ROC_' + filename + '.png'))
        
def show_gt_output(idx_target, labels, predicted_, label_im, filename, path_result):
                ## show the output in image
    label_sorted  = labels[idx_target.argsort()]
    index_sorted  = idx_target[idx_target.argsort()]
    predicted_sorted = predicted_[idx_target.argsort()]
    predicted_sorted = np.expand_dims(predicted_sorted, axis = 1)
   
    im_gt = np.reshape(np.zeros(np.shape(label_im)), (-1,1))
    im_predicted = np.reshape(np.zeros(np.shape(label_im)), (-1,1))
    im_gt[index_sorted, :] =  label_sorted#np.expand_dims(label_sorted, axis = 1)
    im_predicted[index_sorted, :] =  predicted_sorted
    im_gt = np.reshape(im_gt, np.shape(label_im))
    im_predicted = np.reshape(im_predicted, np.shape(label_im))

    fig100, axs = plt.subplots(1,2)
    axs.ravel()

    axs[0].imshow(im_gt)
    axs[0].set_title('Ground truth image')

    im = axs[1].imshow(im_predicted)
    fig100.colorbar(im)
    axs[1].set_title('Predicted result image')

    # fig100 = plt.figure()
    # ax1 = fig100.add_subplot(1,2,1)
    # plt.imshow(im_gt)
    # plt.colorbar()
    # plt.title('Ground truth image')
    
    # ax2 = fig100.add_subplot(1,2,2)
    # plt.imshow(im_predicted)
    # plt.colorbar()
    # plt.title('Predicted result image')
    plt.savefig(os.path.join(path_result, filename + '_predicted_result_im.jpg'))
    
def output_visualize(parameters, outputs_, labels_, filename, path_result):
    fig1 = plt.figure()
    if parameters['end_dim'] == 3:
        ax1 = fig1.add_subplot(1,1,1, projection = '3d')
    else:
        ax1 = fig1.add_subplot(1,1,1)
    color_code = ['r', 'g', 'b']
    count = 0
    for i in parameters['name_class']:
        output_i = outputs_[np.where(labels_ == i)[0],:]
        if parameters['end_dim'] == 2:
            ax1.plot(output_i[:,0], output_i[:,1], '.', label = str(i))
        elif parameters['end_dim'] == 3: 
            ax1.plot(output_i[:,0], output_i[:,1], output_i[:,2], '.', label = str(i))
        else:
            temp = np.arange(1, parameters['end_dim']+1)
            temp = np.expand_dims(temp, axis = 1)
            mean_output_i = np.mean(output_i, axis = 0)
            std_output_i  = np.std(output_i, axis = 0)
#            idx = rand_idx[count]
#            ax1.plot(temp, np.transpose(output_i[idx[0],:]), color_code[count], label = 'class ' + str(parameters['name_class'][count]))
#            ax1.plot(temp, np.transpose(output_i[idx[1:],:]), color_code[count], label = str())

            ax1.errorbar(temp, mean_output_i, yerr=std_output_i, color = color_code[count], label = 'class ' + str(parameters['name_class'][count]))
            count += 1
    ax1.legend()
    plt.title('Visualization of dimensionality reduction rusult')
    plt.savefig(os.path.join(path_result, filename + '_visualization.jpg'))
    