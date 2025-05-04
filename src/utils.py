import numpy as np
import matplotlib.pyplot as plt

def plot_solution(model, x_vals, y_vals, t_val):
    X, Y = np.meshgrid(x_vals, y_vals)
    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
    t_tensor = torch.tensor(np.full_like(x_tensor, t_val), dtype=torch.float32)

    with torch.no_grad():
        input_tensor = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
        u_pred = model(input_tensor).numpy().reshape(X.shape)

    plt.contourf(X, Y, u_pred, 100, cmap='inferno')
    plt.colorbar()
    plt.title(f"Temperature Distribution at t = {t_val}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
