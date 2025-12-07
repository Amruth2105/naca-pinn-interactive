import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pinn_lib
import time

st.set_page_config(page_title="Interactive NACA PINN (PyTorch)", layout="wide")

st.title("Interactive PINN: Laminar Flow over NACA Airfoil (PyTorch)")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
naca_number = st.sidebar.text_input("NACA 4-Digit Series", "6412")
re_number = st.sidebar.number_input("Reynolds Number (Re)", value=1000.0, min_value=1.0, max_value=100000.0)

st.sidebar.header("Training Hyperparameters")
epochs = st.sidebar.number_input("Epochs", value=1000, min_value=100, step=100)
learning_rate = st.sidebar.selectbox("Learning Rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4], index=2)
n_boundary = st.sidebar.number_input("Boundary Points", value=1000, step=100)
n_collocation = st.sidebar.number_input("Collocation Points", value=5000, step=500)

if st.sidebar.button("Start Simulation"):
    # 1. Geometry & Data
    with st.spinner("Generating Grid and Data..."):
        try:
            X_b, Y_b, U_b, V_b, X_c, Y_c, X_airfoil, Y_airfoil = pinn_lib.get_dataset(naca_number, n_boundary, n_collocation)
            st.success(f"Geometry Generated: NACA {naca_number}")
        except Exception as e:
            st.error(f"Error generating geometry: {e}")
            st.stop()

    # 2. Model Initialization
    model = pinn_lib.PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop with Progress
    st.subheader("Training Progress")
    progress_bar = st.progress(0)
    loss_chart = st.empty()
    history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Boundary Prediction
        inputs_b = torch.cat([X_b, Y_b], dim=1)
        out_b = model(X_b, Y_b)
        
        loss_b = torch.mean((out_b[:, 0:1] - U_b)**2) + torch.mean((out_b[:, 1:2] - V_b)**2)
        
        # Physics Prediction
        loss_p = pinn_lib.physics_loss(model, X_c, Y_c, re_number)
        
        loss = loss_b + loss_p
        
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
        if epoch % 10 == 0:
            progress_bar.progress((epoch + 1) / epochs)
            loss_chart.line_chart(history)
            
    end_time = time.time()
    st.success(f"Training Complete! Time taken: {end_time - start_time:.2f}s")
    st.info(f"Final Loss: {history[-1]:.6f}")

    # 4. Visualization
    st.subheader("Flow Visualization")
    
    n_vis = 100
    x_vis = np.linspace(-1, 2, n_vis)
    y_vis = np.linspace(-1, 1, n_vis)
    X_grid, Y_grid = np.meshgrid(x_vis, y_vis)
    X_flat = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32)
    Y_flat = torch.tensor(Y_grid.flatten()[:, None], dtype=torch.float32)
    
    with torch.no_grad():
        preds = model(X_flat, Y_flat)
        u_vis = preds[:, 0].numpy().reshape(n_vis, n_vis)
        v_vis = preds[:, 1].numpy().reshape(n_vis, n_vis)
        p_vis = preds[:, 2].numpy().reshape(n_vis, n_vis)
    
    speed = np.sqrt(u_vis**2 + v_vis**2)
    
    # Simple masking for visualization
    from matplotlib.path import Path
    path = Path(np.hstack([X_airfoil.reshape(-1, 1), Y_airfoil.reshape(-1, 1)]))
    X_flat_np = X_flat.numpy()
    Y_flat_np = Y_flat.numpy()
    mask_vis = path.contains_points(np.hstack([X_flat_np, Y_flat_np])).reshape(n_vis, n_vis)
    
    speed = np.where(mask_vis, np.nan, speed)
    p_vis = np.where(mask_vis, np.nan, p_vis)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Velocity
    c1 = ax[0].contourf(X_grid, Y_grid, speed, 100, cmap='jet')
    ax[0].fill(X_airfoil, Y_airfoil, 'k')
    ax[0].set_title(f"Velocity Magnitude (Re={re_number})")
    ax[0].axis('equal')
    plt.colorbar(c1, ax=ax[0])
    
    # Pressure
    c2 = ax[1].contourf(X_grid, Y_grid, p_vis, 100, cmap='RdBu_r')
    ax[1].fill(X_airfoil, Y_airfoil, 'k')
    ax[1].set_title("Pressure Field")
    ax[1].axis('equal')
    plt.colorbar(c2, ax=ax[1])
    
    st.pyplot(fig)
else:
    st.info("Adjust parameters and click 'Start Simulation' to begin.")
