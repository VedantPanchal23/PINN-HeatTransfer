import torch

def pde_residual(model, x, y, t, alpha=1.0):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    input_tensor = torch.cat([x, y, t], dim=1)
    u = model(input_tensor)

    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    residual = u_t - alpha * (u_xx + u_yy)
    return residual
