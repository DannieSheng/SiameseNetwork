from sklearn.linear_model import LinearRegression
import numpy as np
import os
import torch

class EarlyStopping1:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an imporvement
        """
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.Inf
        self.delta        = delta

    def __call__(self, epoch, val_loss, model, end_dim):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, end_dim)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping epoch: {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, end_dim)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, end_dim):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists('./{}'.format(end_dim)):
            os.makedirs('./{}'.format(end_dim))
        torch.save(model.state_dict(), './{}/checkpoint.pt'.format(end_dim))
        self.val_loss_min = val_loss


class EarlyStopping2:
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
