class EarlyStopper:
    def __init__(self,patience = 5): # we should note that patience in this case is based on the number of evals (not epochs). So 5 would mean 5*eval_every
        self.patience = patience
        self.counter = 0
        self.best_loss = 99999
        self.best_curr_model = None
    
    def __call__(self, val_loss, curr_model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_curr_model = curr_model
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True, self.best_curr_model
        else:
            return False, None
    