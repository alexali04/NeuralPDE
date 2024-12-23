# Define loss functions in this file depending on type of PDE.
import torch
# Generic loss function class. Can be implemented by specific loss functions
# Pretty much useless, just wrote this to have something to import
# If we need a class for loss, this might be useful, otherwise just leave it alone and delete it if it turns out it's unneeded.
class Generic_Loss():
    def __init__(self):
        return
    
    def compute_loss(self):
        return

# Mean Squared Error
def mse_loss(x, y, reduction='mean'):
    return torch.nn.MSELoss(x, y, reduction=reduction)

# Since GP does not batch, the gradient loss must be the sum (or average) of 
# residuals between all gradients for each input x and y.
def pde_loss(pde_type, x, y):
    if pde_type == 'laplacian':
        # TODO
        pass
    pass
    