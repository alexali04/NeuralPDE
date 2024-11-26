#TODO: THIS IS ONLY SUITED FOR LINEAR REGRESSION (as of now). WILL NEED TO BE CHANGED FOR PDES! Also try for higher dimensional inputs; only has been tested on 1-dimensional inputs
#TODO: Need to separate out some things to make it more modular

# Create and train the model
# Needs to import from loss_utils.py to get the loss functions
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt

import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from tqdm import tqdm   # For progress bars

from GP import(
    GP,
)
from utils.loss_utils import(
    Generic_Loss,
    mse_loss,
)
from utils.data_utils import(
    gen_interior_points_rectangle,
    data_square,
    data_loader_square,
)
from PDEs.Laplacian import(
    rectangle_loss,
    compute_loss_square,
)
from Regression.regression_data import(
    gen_regression_data,
)

# Currently testing linear regression
# Using mll, take negative to minimize
# TODO: Delete batch_size parameter, no batching for Gaussian process
def train(model, train_data, val_data, optimizer, loss_fn, epochs=20):
    iterator = tqdm(range(epochs))
    for epoch in iterator:
        x, y = train_data
        output = model(x)
        loss = -loss_fn(output, y)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss = loss.detach().clone()
        
        # val_x, val_y = val_data
        # output = model(val_x)
        # loss = -loss_fn(output, val_y)
        # val_loss = loss.mean().detach().clone()    # TODO: Print val_loss (first make sure training works)
        iterator.set_description('EPOCH %d - TRAIN LOSS = %d' % (epoch, epoch_loss))
    return

def test(model, likelihood, test_data, loss_fn, batch_size=1):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        x, y = test_data
        output = model(x)
        samples = output.sample(sample_shape=torch.Size([10]))
        # print('PREDS:', preds)
        # loss = -loss_fn(preds, y)
        # test_loss = loss.detach().clone()
        # print('TEST LOSS = %d' % (test_loss))
    return output, samples

# Generate Data
# Generates data for either regression or PDEs, specification must be passed to function
def gen_data(problem='regression', datapoints=1000, train=True,
             boundary_condition={"bottom": 1.0, "top": 0, "left": 1.0, "right": 0},
             domain_bounds={"bottom": 0, "left": 0, "top": 1, "right": 1}):
    if problem == 'regression':
        data = gen_regression_data(datapoints=datapoints, train=train)
    elif problem == 'laplacian':
        # So far, laplacian only generates training data and no validation set.
        data = gen_interior_points_rectangle(5, domain_bounds)
    return data

# -------------------------------------------------------------------------------------

def main(problem, data, epochs=100):
    # Set kernel and likelihood
    if problem == 'regression':
        train_x, train_y, val_x, val_y = data
    else:
        train_x, train_y = data
        val_x, val_y = None, None
    
    mean = gpytorch.means.ConstantMean()
    covar = gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                    num_dims=2, grid_size=100)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    distribution = gpytorch.distributions.MultivariateNormal
    # Define model and log marginal likelihood (loss function for optimization)
    model = GP(train_x=train_x, train_y=train_y, mean=mean, covar=covar, likelihood=likelihood, distribution=distribution)
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # Set optimizer
    optimizer = torch.optim.Adam([
        {'params': model.transform.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.02)
    # Train the model
    train(model=model,
        train_data=(train_x, train_y),
        val_data=(val_x, val_y),
        optimizer=optimizer,
        loss_fn=mll,
        epochs=epochs,
        )

    # Generate regression test points
    test_x, test_y = gen_data(problem='regression', datapoints=100, train=False)

    # preds, samples = test(model=model,
    #                       likelihood=likelihood,
    #                       test_data=(test_x, test_y),
    #                       loss_fn=mse_loss)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        plt.figure(figsize=(10, 6))
        # observed data
        plt.plot(train_x.squeeze().numpy(), train_y.squeeze().numpy(), 'k*', label='Observed Data')
        
        # Get predictions
        observed_pred = likelihood(model(test_x))
        # sampled functions
        plt.plot(test_x.squeeze().numpy(), observed_pred.mean.numpy(), 'b')   

        # confidence region
        lower, upper = observed_pred.confidence_region()
        plt.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), color='mediumpurple', alpha=0.5)
        plt.ylim([-3.5, 3.5])
        plt.legend(['Observed Data', 'Prediction', 'Confidence Region'])
        plt.title("GP Function Interpolation")
        plt.show()
# Set type of problem we want to generate data for here
problem = 'regression'
# If PDE, need to set boundary conditions and domain.
boundary_condition={"bottom": 1.0, "top": 0, "left": 1.0, "right": 0}
domain_bounds={"bottom": 0, "left": 0, "top": 1, "right": 1}
data = gen_data(problem=problem, datapoints=1000,
                boundary_condition=boundary_condition,
                domain_bounds=domain_bounds)

main(problem=problem, data=data, epochs=50)