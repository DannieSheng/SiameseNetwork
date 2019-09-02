from sklearn.linear_model import LinearRegression

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
        self.thres   = 0.0005
        self.y = np.zeros(patience)
        self.score = np.Inf
        self.count = 0
        self.x = np.arange(0,patience)

    def __call__(self, epoch, val_loss, model, end_dim):
        if self.count < self.patience:
            self.y[count] = val_loss
            self.count += 1
            self.save_checkpoint(val_loss, model, end_dim)
            self.early_stop = False
        else:
            regressor = LinearRegression()  
            regressor.fit(self.x, self.y)
            self.score = regressor.coef_
            if self.score > self.thres:
                self.early_stop = True
                print(f'EarlyStopping epoch: {epoch}')
            else:
                self.early_stop = False
                self.save_checkpoint(val_loss, model, end_dim)
                self.count = 0

    def save_checkpoint(self, epoch, val_loss, model, end_dim):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Saving model ...')
#            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists('./' + str(end_dim)):
            os.makedirs('./' + str(end_dim))
        torch.save(model.state_dict(), './' + str(end_dim)+'/checkpoint.pt')
#        self.score = val_loss