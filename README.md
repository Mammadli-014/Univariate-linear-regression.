# 🏠 Linear Regression with Gradient Descent

This project implements **Linear Regression** from scratch using **Gradient Descent** to predict housing prices based on living area (sqft).

It demonstrates the mathematical intuition behind machine learning algorithms without relying on high-level frameworks like Scikit-learn.

# 📁 Project Overview

-   **Data Loading:** Reads housing data (Area vs. Price) from `data.txt`.
-   **Normalization:** Scales the target prices for better convergence.
-   **Core Algorithms:**
    -   `compute_cost()`: Calculates the Mean Squared Error cost function $J(w,b)$.
    -   `compute_gradient()`: Computes partial derivatives for optimization.
    -   `gradient_descent()`: Optimizes parameters $w$ and $b$ iteratively.
-   **Visualization:** Plots the original data and the final regression line using **Matplotlib**.

# ⚙️ Requirements

-   Python 3.x
-   NumPy
-   Matplotlib

You can install dependencies via pip:
```bash
pip install numpy matplotlib