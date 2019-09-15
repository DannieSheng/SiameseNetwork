# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:35:45 2019
Parameters and helper functions for applying trained Siamese Network on new data
(Should be able to use the "helper_funcs_Siamese" directly, however using a separate one enables running two experiments simultaneously
@author: hdysheng
"""



import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import os


    ### helper functions

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
        
        # output = self.activation(output)
        # output = self.dropout(output)
        # output = self.fc5(output)
        
        # output = self.activation(output)
        # output = self.dropout(output)
        # output = self.fc6(output)
        
        # output = self.activation(output)
        # output = self.dropout(output)
        # output = self.fc7(output)
    
        # output = self.activation(output)
        # output = self.dropout(output)
        # output = self.fc8(output)
        
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

    
