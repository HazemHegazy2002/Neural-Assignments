import numpy as np
import matplotlib.pyplot as plt

# ========================
# Regressing Problems 1 & 2
# ========================

# Problem 1: Insured Persons in a Country

# ========================
# Data (ACTUAL YEARS)
# ========================
years = np.array([1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996])
y = np.array([12400, 10900, 10000, 1050, 9500, 8900, 8000, 7800, 7600, 7200])

# ========================
# R^2 function
# ========================
def r2_score(y, y_pred):
    return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

# ========================
# Function to print equations
# ========================
def print_equation(p):
    degree = len(p) - 1
    eq = "y = "
    for i, coef in enumerate(p):
        power = degree - i
        if power > 1:
            eq += f"{coef:.6e}*Year^{power} + "
        elif power == 1:
            eq += f"{coef:.6e}*Year + "
        else:
            eq += f"{coef:.2f}"
    return eq

# ========================
# BEFORE CLEANING
# ========================
p1 = np.polyfit(years, y, 1)
p2 = np.polyfit(years, y, 2)
p3 = np.polyfit(years, y, 3)

y1 = np.polyval(p1, years)
y2 = np.polyval(p2, years)
y3 = np.polyval(p3, years)

print("=========== BEFORE CLEANING ===========")
print(f"Linear R² = {r2_score(y, y1):.4f}")
print(f"Quadratic R² = {r2_score(y, y2):.4f}")
print(f"Cubic R² = {r2_score(y, y3):.4f}")

print("\nFinal Models (Before Cleaning):")
print("Linear:    ", print_equation(p1))
print("Quadratic: ", print_equation(p2))
print("Cubic:     ", print_equation(p3))

# ------------------------
# Individual plots BEFORE
# ------------------------
titles = ["Linear", "Quadratic", "Cubic"]
models = [y1, y2, y3]

for i in range(3):
    plt.figure()
    plt.scatter(years, y, label="Data")
    plt.plot(years, models[i], label=f"{titles[i]} Fit")
    plt.title(f"{titles[i]} Fit (Before Cleaning)")
    plt.legend()
    plt.grid()
    plt.show()

# Combined BEFORE
plt.figure()
plt.scatter(years, y, label="Data")
plt.plot(years, y1, label="Linear")
plt.plot(years, y2, label="Quadratic")
plt.plot(years, y3, label="Cubic")
plt.title("All Models (Before Cleaning)")
plt.legend()
plt.grid()
plt.show()

# ========================
# AFTER CLEANING
# ========================
mask = years != 1990
years_c = years[mask]
y_c = y[mask]

p1_c = np.polyfit(years_c, y_c, 1)
p2_c = np.polyfit(years_c, y_c, 2)
p3_c = np.polyfit(years_c, y_c, 3)

y1_c = np.polyval(p1_c, years_c)
y2_c = np.polyval(p2_c, years_c)
y3_c = np.polyval(p3_c, years_c)

print("\n=========== AFTER CLEANING ===========")
print(f"Linear R² = {r2_score(y_c, y1_c):.4f}")
print(f"Quadratic R² = {r2_score(y_c, y2_c):.4f}")
print(f"Cubic R² = {r2_score(y_c, y3_c):.4f}")

print("\nFinal Models (After Cleaning):")
print("Linear:    ", print_equation(p1_c))
print("Quadratic: ", print_equation(p2_c))
print("Cubic:     ", print_equation(p3_c))

# ------------------------
# Individual plots AFTER
# ------------------------
models_c = [y1_c, y2_c, y3_c]

for i in range(3):
    plt.figure()
    plt.scatter(years_c, y_c, label="Clean Data")
    plt.plot(years_c, models_c[i], label=f"{titles[i]} Fit")
    plt.title(f"{titles[i]} Fit (After Cleaning)")
    plt.legend()
    plt.grid()
    plt.show()

# Combined AFTER
plt.figure()
plt.scatter(years_c, y_c, label="Clean Data")
plt.plot(years_c, y1_c, label="Linear")
plt.plot(years_c, y2_c, label="Quadratic")
plt.plot(years_c, y3_c, label="Cubic")
plt.title("All Models (After Cleaning)")
plt.legend()
plt.grid()
plt.show()

# ========================
# Prediction
# ========================
best_model = p2_c  # Quadratic
year_pred = 1997
y_pred = np.polyval(best_model, year_pred)

print(f"\nPredicted insured persons in 1997 with best Model : {y_pred:.2f}")

#problem 2: House Prices in a City