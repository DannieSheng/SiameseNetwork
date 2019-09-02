# -*- coding: utf-8 -*-
"""
Created on Thrus. July 4 12:15:39 2019
Parameters and helper functions for the Siamese Network for the hyperspectral dataset
@author: hdysheng
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import pdb
    
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

def plot_confu(labels, predicted, path_result, filename):
    
    confu = confusion_matrix(labels, predicted, labels=None, sample_weight=None)
    confu_percent = confu / confu.astype(np.float).sum(axis=1)
    df_cm = pd.DataFrame(confu_percent)
    
        # plot confusion matrix
    fig101 = plt.figure()
    ax1    = fig101.add_subplot(1,1,1)
    sn.heatmap(df_cm, annot = True)
    plt.title('Confusion matrix')
    plt.savefig(os.path.join(path_result, 'cM_' + filename + '.jpg'))  
