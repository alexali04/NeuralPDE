import bvp as bvp
import torch
from matplotlib import pyplot as plt
import bvp_pde as bvp_pde

def main():
    model = bvp.MLP(2, 1, 1024)
    model = model.cuda()
    solver = bvp_pde.LaplaceSolver(model)
    solver.solve(model, epochs=1000, lr=1e-4, bs=512)
    test_x1 = torch.linspace(0., 2*torch.pi, 100)
    test_x2 = torch.linspace(0., 2*torch.pi, 100)
    X1, X2 = torch.meshgrid(test_x1, test_x2, indexing="ij")
    X = torch.cat((X1.reshape(-1,1), X2.reshape(-1,1)), dim=-1)
    print(X.shape)
    with torch.no_grad():
        output = model(X)
        y = output.reshape(X1.shape).detach().cpu().numpy()
        print(y.shape)
        plt.imshow(y, cmap='plasma', extent=[torch.min(X1).item(),torch.max(X1).item(),torch.min(X2).item(),torch.max(X2).item()])
        plt.show()

if __name__ == "__main__":
    main()