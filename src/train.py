import torch
from src.model import FCN
from src.pde import pde_residual

def loss_function(model, x_f, y_f, t_f, x_b, y_b, t_b, u_b):
    f_pred = pde_residual(model, x_f, y_f, t_f)
    u_pred = model(torch.cat([x_b, y_b, t_b], dim=1))

    mse_f = torch.mean(f_pred ** 2)
    mse_u = torch.mean((u_pred - u_b) ** 2)
    return mse_f + mse_u

def train(model, optimizer, x_f, y_f, t_f, x_b, y_b, t_b, u_b, epochs=10000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, x_f, y_f, t_f, x_b, y_b, t_b, u_b)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
