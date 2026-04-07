import numpy as np
import matplotlib.pyplot as plt
from typing import Any
def print_linear_equation(beta: Any):
    print(f"Linear Model: y = {beta[0]:.4f}*T + {beta[1]:.4f}*I + {beta[2]:.4f}")

def print_quadratic_equation(beta: Any):
    print(f"Quadratic Model: y = {beta[0]:.4f}*T^2 + {beta[1]:.4f}*I^2 + {beta[2]:.4f}*T*I + {beta[3]:.4f}*T + {beta[4]:.4f}*I + {beta[5]:.4f}")

def predict_linear(beta: Any, T_val, I_val):
    return beta[0]*T_val + beta[1]*I_val + beta[2]

def predict_quadratic(beta: Any, T_val, I_val):
    return beta[0]*T_val**2 + beta[1]*I_val**2 + beta[2]*T_val*I_val + beta[3]*T_val + beta[4]*I_val + beta[5]
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch



# Data
# ========================
y = np.array([270,362,162,45,91,233,372,305,234,122,25,210,450,325,52])
T = np.array([40,27,40,73,65,65,10,9,24,65,66,41,22,40,60])
I = np.array([4,4,10,6,7,40,6,10,10,4,10,6,4,4,10])

# ========================
# R2 function
# ========================
def r2_score(y, y_pred):
    return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

# ========================
# Helper: create surface
# ========================
def create_surface(T_data, I_data, beta_lin, beta_quad):
    T_range = np.linspace(min(T_data), max(T_data), 30)
    I_range = np.linspace(min(I_data), max(I_data), 30)
    T_grid, I_grid = np.meshgrid(T_range, I_range)

    T_flat = T_grid.flatten()
    I_flat = I_grid.flatten()

    # Linear
    X_lin = np.column_stack((T_flat, I_flat, np.ones(len(T_flat))))
    y_lin = (X_lin @ beta_lin).reshape(T_grid.shape)

    # Quadratic
    X_quad = np.column_stack((T_flat**2, I_flat**2, T_flat*I_flat, T_flat, I_flat, np.ones(len(T_flat))))
    y_quad = (X_quad @ beta_quad).reshape(T_grid.shape)

    return T_grid, I_grid, y_lin, y_quad

# ========================
# BEFORE CLEANING
# -------- Plot: Data with Outlier Highlighted
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Identify outliers (I == 40)
outlier_mask = (I == 40)
normal_mask = ~outlier_mask
# Plot normal points
ax.scatter(T[normal_mask], I[normal_mask], y[normal_mask], color='#0072B2', edgecolor='k', s=70, label='Normal')
# Plot outliers
ax.scatter(T[outlier_mask], I[outlier_mask], y[outlier_mask], color='#D55E00', edgecolor='k', s=90, label='Outlier')
ax.set_title('Data with Outlier Highlighted', fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.legend()
ax.grid(True)
# Set axis limits to start from min values
ax.set_xlim([T.min(), T.max()])
ax.set_ylim([I.min(), I.max()])
ax.set_zlim([y.min(), y.max()])
plt.tight_layout()
plt.show()
# ========================
X_lin = np.column_stack((T, I, np.ones(len(T))))
X_quad = np.column_stack((T**2, I**2, T*I, T, I, np.ones(len(T))))

# Fit models
beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
beta_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
# Create surfaces for plotting
Tg, Ig, y_lin_surf, y_quad_surf = create_surface(T, I, beta_lin, beta_quad)

# --- BEFORE CLEANING ---
print("\n=== BEFORE CLEANING ===")
print_linear_equation(beta_lin)
print_quadratic_equation(beta_quad)
y_pred_lin = X_lin @ beta_lin
y_pred_quad = X_quad @ beta_quad
r2_lin = r2_score(y, y_pred_lin)
r2_quad = r2_score(y, y_pred_quad)
print(f"R^2 (Linear): {r2_lin:.4f}")
print(f"R^2 (Quadratic): {r2_quad:.4f}")


# -------- Plot 1: Linear only (Before)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T, I, y, color='#0072B2', edgecolor='k', s=70, label='Data')
ax.plot_surface(Tg, Ig, y_lin_surf, alpha=0.5, cmap='viridis', linewidth=0, antialiased=True, label='Linear Surface')
ax.set_title("Linear Model (Before Cleaning)", fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.legend()
ax.grid(True)
ax.set_xlim([T.min(), T.max()])
ax.set_ylim([I.min(), I.max()])
ax.set_zlim([y.min(), y.max()])
plt.tight_layout()
plt.show()

# -------- Plot 2: Quadratic only (Before)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T, I, y, color='#0072B2', edgecolor='k', s=70, label='Data')
ax.plot_surface(Tg, Ig, y_quad_surf, alpha=0.5, cmap='plasma', linewidth=0, antialiased=True, label='Quadratic Surface')
ax.set_title("Quadratic Model (Before Cleaning)", fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.legend()
ax.grid(True)
ax.set_xlim([T.min(), T.max()])
ax.set_ylim([I.min(), I.max()])
ax.set_zlim([y.min(), y.max()])
plt.tight_layout()
plt.show()

# -------- Plot 3: Both (Before)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T, I, y, color='#0072B2', edgecolor='k', s=70, label='Data')
surf1 = ax.plot_surface(Tg, Ig, y_lin_surf, alpha=0.4, cmap='viridis', linewidth=0, antialiased=True)
surf2 = ax.plot_surface(Tg, Ig, y_quad_surf, alpha=0.5, cmap='plasma', linewidth=0, antialiased=True)
legend_elements = [
    Patch(facecolor='#56B4E9', label='Linear Surface'),
    Patch(facecolor='#F0E442', label='Quadratic Surface'),
    Patch(facecolor='#0072B2', label='Data')
]
ax.legend(handles=legend_elements)
ax.set_title("Linear vs Quadratic (Before Cleaning)", fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.grid(True)
ax.set_xlim([T.min(), T.max()])
ax.set_ylim([I.min(), I.max()])
ax.set_zlim([y.min(), y.max()])
plt.tight_layout()
plt.show()

# ========================
# AFTER CLEANING
# ========================
mask = I != 40

T_c = T[mask]
I_c = I[mask]
y_c = y[mask]

X_lin_c = np.column_stack((T_c, I_c, np.ones(len(T_c))))
X_quad_c = np.column_stack((T_c**2, I_c**2, T_c*I_c, T_c, I_c, np.ones(len(T_c))))

beta_lin_c = np.linalg.lstsq(X_lin_c, y_c, rcond=None)[0]
beta_quad_c = np.linalg.lstsq(X_quad_c, y_c, rcond=None)[0]

# --- AFTER CLEANING ---
print("\n=== AFTER CLEANING ===")
print_linear_equation(beta_lin_c)
print_quadratic_equation(beta_quad_c)
y_pred_lin_c = X_lin_c @ beta_lin_c
y_pred_quad_c = X_quad_c @ beta_quad_c
r2_lin_c = r2_score(y_c, y_pred_lin_c)
r2_quad_c = r2_score(y_c, y_pred_quad_c)
print(f"R^2 (Linear): {r2_lin_c:.4f}")
print(f"R^2 (Quadratic): {r2_quad_c:.4f}")


Tg_c, Ig_c, y_lin_surf_c, y_quad_surf_c = create_surface(T_c, I_c, beta_lin_c, beta_quad_c)

# -------- Plot 4: Linear only (After)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T_c, I_c, y_c, color='#009E73', edgecolor='k', s=70, label='Data (Cleaned)')
ax.plot_surface(Tg_c, Ig_c, y_lin_surf_c, alpha=0.5, cmap='viridis', linewidth=0, antialiased=True, label='Linear Surface')
ax.set_title("Linear Model (After Cleaning)", fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.legend()
ax.grid(True)
ax.set_xlim([T_c.min(), T_c.max()])
ax.set_ylim([I_c.min(), I_c.max()])
ax.set_zlim([y_c.min(), y_c.max()])
plt.tight_layout()
plt.show()

# -------- Plot 5: Quadratic only (After)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T_c, I_c, y_c, color='#009E73', edgecolor='k', s=70, label='Data (Cleaned)')
ax.plot_surface(Tg_c, Ig_c, y_quad_surf_c, alpha=0.5, cmap='plasma', linewidth=0, antialiased=True, label='Quadratic Surface')
ax.set_title("Quadratic Model (After Cleaning)", fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.legend()
ax.grid(True)
ax.set_xlim([T_c.min(), T_c.max()])
ax.set_ylim([I_c.min(), I_c.max()])
ax.set_zlim([y_c.min(), y_c.max()])
plt.tight_layout()
plt.show()

# -------- Plot 6: Both (After)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T_c, I_c, y_c, color='#009E73', edgecolor='k', s=70, label='Data (Cleaned)')
surf1 = ax.plot_surface(Tg_c, Ig_c, y_lin_surf_c, alpha=0.4, cmap='viridis', linewidth=0, antialiased=True)
surf2 = ax.plot_surface(Tg_c, Ig_c, y_quad_surf_c, alpha=0.5, cmap='plasma', linewidth=0, antialiased=True)
legend_elements = [
    Patch(facecolor='#56B4E9', label='Linear Surface'),
    Patch(facecolor='#F0E442', label='Quadratic Surface'),
    Patch(facecolor='#009E73', label='Data (Cleaned)')
]
ax.legend(handles=legend_elements)
ax.set_title("Linear vs Quadratic (After Cleaning)", fontsize=13)
ax.set_xlabel('T', fontsize=11)
ax.set_ylabel('I', fontsize=11)
ax.set_zlabel('y', fontsize=11)
ax.grid(True)
ax.set_xlim([T_c.min(), T_c.max()])
ax.set_ylim([I_c.min(), I_c.max()])
ax.set_zlim([y_c.min(), y_c.max()])
plt.tight_layout()
plt.show()

if r2_lin_c > r2_quad_c:
    print("Best fit: Linear Model") 
    pred_val_15_5 = predict_linear(beta_lin_c, 15, 5)
    print(f"Prediction for T=15, I=5: {pred_val_15_5:.2f}")
else:
    print("Best fit: Quadratic Model")
    pred_val_15_5 = predict_quadratic(beta_quad_c, 15, 5)
    print(f"Prediction for T=15, I=5: {pred_val_15_5:.2f}")