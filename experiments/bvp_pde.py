import torch
import bvp

class LinearSolver(bvp.BVPSolver):
    def pde_f(nn, x):
        u = nn(x)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] - 1.0
        res = du_dx
        return res
    
    def sampler(self, bs):
        return bvp.sample_batch(1, bs), torch.zeros((bs,1))
    
    def bcs(dX):
        return torch.full((dX.shape[0],), 3.0)
    
class LaplaceSolver(bvp.BVPSolver):
    def pde_f(nn, x):
        u = nn(x)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        res = d2u_dx2.sum(axis=-1)
        return res
    
    def sampler(self, bs):
        # rectangular boundary
        boundary_size = bs // 4
        left = torch.zeros((boundary_size,2))
        right = torch.ones((boundary_size,2))
        bottom = torch.zeros((boundary_size,2))
        top = torch.ones((boundary_size,2))
        left[:,1] = torch.rand((boundary_size,))
        right[:,1] = torch.rand((boundary_size,))
        bottom[:,0] = torch.rand((boundary_size,))
        top[:,0] = torch.rand((boundary_size,))
        dX = 2*torch.pi*torch.cat([left, right, bottom, top], dim=0)
        rp = torch.randperm(dX.shape[0])
        dX = dX[rp,:]
        return bvp.sample_batch(2, bs), dX
    
    def bcs(dX):
        # u0 = torch.where(torch.logical_or(dX[:,1] == 0., dX[:,1] == 2*torch.pi), 
        #                  torch.sin(dX[:,0]),
        #                  torch.sin(dX[:,1]))
        u0 = torch.where(dX[:,1] == 0,
                         torch.sin(dX[:,0]-torch.pi/2)+1,
                         torch.zeros(dX.shape[0]))
        return u0
