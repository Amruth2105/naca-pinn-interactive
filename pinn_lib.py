import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def naca4(number, n_points=100):
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0

    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    yc = np.where(x < p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))

    dyc_dx = np.where(x < p,
                      2 * m / p**2 * (p - x),
                      2 * m / (1 - p)**2 * (p - x))

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    X = np.concatenate((xu, xl[::-1]))
    Y = np.concatenate((yu, yl[::-1]))
    
    return X, Y

def get_dataset(naca_number, N_b, N_c):
    # Generate Airfoil Points
    X_airfoil, Y_airfoil = naca4(naca_number, n_points=200)

    # --- BOUNDARY POINTS ---
    # Inlet (x=-1)
    xy_inlet = np.hstack([np.full((N_b // 4, 1), -1.0), np.random.uniform(-1, 1, (N_b // 4, 1))])
    u_inlet = np.ones((N_b // 4, 1))
    v_inlet = np.zeros((N_b // 4, 1))

    # Top boundary (y=1)
    xy_top = np.hstack([np.random.uniform(-1, 2, (N_b // 4, 1)), np.full((N_b // 4, 1), 1.0)])
    u_top = np.ones((N_b // 4, 1))
    v_top = np.zeros((N_b // 4, 1))
    
    # Bottom boundary (y=-1)
    xy_bot = np.hstack([np.random.uniform(-1, 2, (N_b // 4, 1)), np.full((N_b // 4, 1), -1.0)])
    u_bot = np.ones((N_b // 4, 1))
    v_bot = np.zeros((N_b // 4, 1))

    # Airfoil Surface (No-slip)
    # Resample airfoil points to N_b // 4
    idx = np.random.choice(len(X_airfoil), N_b // 4)
    xy_foil = np.hstack([X_airfoil[idx].reshape(-1, 1), Y_airfoil[idx].reshape(-1, 1)])
    u_foil = np.zeros((N_b // 4, 1))
    v_foil = np.zeros((N_b // 4, 1))

    # Combine Boundary Data
    X_b = np.concatenate([xy_inlet[:,0:1], xy_top[:,0:1], xy_bot[:,0:1], xy_foil[:,0:1]])
    Y_b = np.concatenate([xy_inlet[:,1:2], xy_top[:,1:2], xy_bot[:,1:2], xy_foil[:,1:2]])
    U_b = np.concatenate([u_inlet, u_top, u_bot, u_foil])
    V_b = np.concatenate([v_inlet, v_top, v_bot, v_foil])

    # --- COLLOCATION POINTS ---
    # Uniform sampling in [-1, 2] x [-1, 1]
    X_c = np.random.uniform(-1, 2, (N_c, 1))
    Y_c = np.random.uniform(-1, 1, (N_c, 1))
    
    # Filter points OUTSIDE airfoil
    path = Path(np.hstack([X_airfoil.reshape(-1, 1), Y_airfoil.reshape(-1, 1)]))
    mask = ~path.contains_points(np.hstack([X_c, Y_c]))
    X_c = X_c[mask]
    Y_c = Y_c[mask]

    # Convert to PyTorch Tensors
    X_b = torch.tensor(X_b, dtype=torch.float32)
    Y_b = torch.tensor(Y_b, dtype=torch.float32)
    U_b = torch.tensor(U_b, dtype=torch.float32)
    V_b = torch.tensor(V_b, dtype=torch.float32)
    X_c = torch.tensor(X_c.reshape(-1, 1), dtype=torch.float32)
    Y_c = torch.tensor(Y_c.reshape(-1, 1), dtype=torch.float32)

    return X_b, Y_b, U_b, V_b, X_c, Y_c, X_airfoil, Y_airfoil

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3) # u, v, p
        )

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.hidden_layers(inputs)

def physics_loss(model, x, y, Re):
    x.requires_grad = True
    y.requires_grad = True
    
    outputs = model(x, y)
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]
    
    # First derivatives
    grads_u = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y = grads_u[0], grads_u[1]
    
    grads_v = torch.autograd.grad(v, [x, y], grad_outputs=torch.ones_like(v), create_graph=True)
    v_x, v_y = grads_v[0], grads_v[1]
    
    grads_p = torch.autograd.grad(p, [x, y], grad_outputs=torch.ones_like(p), create_graph=True)
    p_x, p_y = grads_p[0], grads_p[1]
    
    # Second derivatives
    grads_u_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    grads_u_y = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_xx = grads_u_x
    u_yy = grads_u_y
    
    grads_v_x = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    grads_v_y = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_xx = grads_v_x
    v_yy = grads_v_y
    
    # Navier-Stokes Residuals
    f_cont = u_x + v_y
    f_u = (u * u_x + v * u_y) + p_x - (1/Re) * (u_xx + u_yy)
    f_v = (u * v_x + v * v_y) + p_y - (1/Re) * (v_xx + v_yy)
    
    return torch.mean(f_cont**2) + torch.mean(f_u**2) + torch.mean(f_v**2)
