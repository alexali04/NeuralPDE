import bvp as bvp
import torch
from matplotlib import pyplot as plt
import bvp_pde as bvp_pde

bvp.USE_CUDA = False

def main():
    model = bvp.MLP(1, 1, 10)
    solver = bvp_pde.LinearSolver(model)
    solver.solve(model, epochs=1000, lr=3e-3, bs=1000)
    test_x = torch.linspace(0., 1., 100).view(-1,1)
    with torch.no_grad():
        output = model(test_x)
        y = output.detach().cpu().numpy()
        plt.plot(test_x.detach().cpu().numpy(), y)
        plt.show()

if __name__ == "__main__":
    main()