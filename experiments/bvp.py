import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

USE_CUDA = True
if USE_CUDA and torch.cuda.is_available():
    torch.set_default_device("cuda")

def sample_batch(d, bs):
    X = torch.rand((bs, d)) # X ~ Uniform[0,1] X.shape = (batch_size, dimension)
    return X

def batch_residuals(nn, pde_f, X):
    f = pde_f(nn, X)
    return torch.mean(f**2)

def optimize_loss(loss, model, init_params, sampler, epochs=10000, lr=1e-3, bs=10000):
    opt = torch.optim.Adam(init_params, lr)
    progress = tqdm(range(epochs), desc="LS solve for initial conditions")
    model.train()
    for step in progress:
        X, dX = sampler(bs)
        X.requires_grad = True
        opt.zero_grad()
        l, fit_mse, pde_mse = loss(model, X, dX)
        l.backward()
        opt.step()
        progress.set_description(f"epoch:{step}, loss:{l.item():.5f}, ({fit_mse:0.5f},{pde_mse:0.5f})")

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, act=nn.Tanh):
        super(MLP, self).__init__()
        self.act = act
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), self.act(),
            nn.Linear(hidden_dim, hidden_dim), self.act(),
            nn.Linear(hidden_dim, hidden_dim), self.act(),
            nn.Linear(hidden_dim, hidden_dim), self.act(),
            nn.Linear(hidden_dim, out_dim))
    
    def forward(self, x):
        y = self.fc(x)
        return y

class BVPSolver:
    def __init__(self, model, epochs=1000, alpha=1):
        self.epochs = epochs
        self.nn = model
        self.alpha = alpha

    @staticmethod
    def pde_f(nn, x):
        # all terms in the PDE are on the left and should equal 0.
        del nn, x
        raise NotImplementedError
    
    def sampler(self, bs):
        # returns X and dX (corresponding to the points in the boundary)
        del bs
        raise NotImplementedError

    @staticmethod
    def bcs(dX):
        del dX
        raise NotImplementedError
    
    def solve(self, model, epochs=50_000, lr=3e-4, bs=10_000):
        u0 = self.__class__.bcs
        def combined_loss(model, X, dX):
            # dX : spatial points in the boundary
            fit_mse = ((model(dX).squeeze() - u0(dX))**2).mean()
            pde_mse = batch_residuals(model, self.__class__.pde_f, X)
            return fit_mse + self.alpha *  pde_mse, fit_mse, pde_mse

        optimize_loss(combined_loss, model, model.parameters(), self.sampler, epochs, lr, bs)
