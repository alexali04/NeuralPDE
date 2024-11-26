# Define loss functions in this file depending on type of PDE.
import torch
# Generic loss function class. Can be implemented by specific loss functions
# Pretty much useless, just wrote this to have something to import
# If we need a class for loss, this might be useful, otherwise just leave it alone and delete it if it turns out it's unneeded.

from PDEs.Laplacian import(
    compute_laplacian,
    boundary_loss,
    compute_loss_square,
    
)

# We want to take the mean function (neural network model for PDEs) and optimize
# the marginal log-likelihood over the kernel hyperparameters. 
# Will have to define 
class ModifiedMllLoss():
    def __init__(self, mean_function, ):
        return
    
    def compute_loss(self):
        return

# Mean Squared Error
def mse_loss(x, y, reduction='mean'):
    return torch.nn.MSELoss(x, y, reduction=reduction)

# -------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------#
# Since GP does not batch, the gradient loss (I think must) be the sum (or average) of 
# residuals between all gradients for each input x and y, the boundary loss, AND the MLL.

# This will calculate the sum or average (depending) of the whole datasets
# grad loss
def pde_grad_loss(pde_type, model, u, x, y):
    if pde_type == 'laplacian':
        # TODO
        # x will be a vector dim(n): input
        # y will be a vector dim(n): input
        # u will be a vector dim(n): model output
        
        pass
    pass
# This will calculate the sum or average (depending) of the whole datasets
# boundary loss
def pde_boundary_loss(pde_type, model, x, y):
    if pde_type == 'laplacian':
        # TODO
        pass
    pass

    