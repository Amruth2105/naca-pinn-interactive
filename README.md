# Interactive NACA 6412 PINN (PyTorch)

An interactive Physics-Informed Neural Network (PINN) simulation for laminar flow over NACA airfoils, built with **PyTorch** and **Streamlit**.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

[**🚀 Try the Live Demo Here**](https://naca-pinn-interactive-brcjxevvb4ayctqvh3lg3y.streamlit.app/)

## Overview

This application solves the Navier-Stokes equations for 2D flow over a NACA 4-digit airfoil (default: NACA 6412). It allows users to:

1.  **Modify Geometry**: Input any 4-digit NACA series (e.g., 0012, 2412, 4412).
2.  **Adjust Physics**: Change the Reynolds number ($Re$).
3.  **Train in Real-Time**: Watch the PINN learn the flow field physics.
4.  **Visualize Results**: View Velocity Magnitude and Pressure fields.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Amruth2105/naca-pinn-interactive.git
    cd naca-pinn-interactive
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Features

-   **Geometry Generator**: `naca4` function creates airfoil coordinates on the fly.
-   **Physics Loss**: Custom loss function incorporating Navier-Stokes residuals (Continuity + Momentum).
-   **Interactive UI**: Sidebar controls for hyperparameters (Epochs, Learning Rate, Point Density).
-   **PyTorch Backend**: Fast and flexible automatic differentiation for physics constraints.

## License

MIT License
