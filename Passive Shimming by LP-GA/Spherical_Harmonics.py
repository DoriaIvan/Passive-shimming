import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Read Excel file
data = pd.read_excel('../Passive Shimming by LP-GA/ConT-003 sample points field_20240620.xlsx', sheet_name='Coordination')
data_np = data.values

# Extract numerical data (assuming first few columns might be non-numeric)
data_numeric = data_np[:, :12].astype(float)  # Adjust based on your data structure

# Save data
np.save('data.npy', data_numeric)

# Extract fields
Bm_x = data_numeric[:, 7]  # unit to mT
np.save('Bm_x.npy', Bm_x)

B0mag = data_numeric[:, 10]

# Extract coordinates and convert to mm
R = data_numeric[:, 0] * 1e3  # unit to mm///扫球半径改为15mm
theta = data_numeric[:, 1]
phi = data_numeric[:, 2]
X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

# Plot 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X, Y, Z, s=16)
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.grid(True)
ax.set_box_aspect([1, 1, 1])  # This is equivalent to 'axis equal'
ax.view_init(elev=360, azim=270)
plt.show()

# Harmonic components and expression
Z0 = np.ones(len(data_numeric[:, 3]))

Z2 = Z**2 - 0.5 * (X**2 + Y**2)  # Z^2-0.5*(X^2+Y^2)
ZX = Z * X  # ZX
ZY = Z * Y  # ZY
X2 = X**2 - Y**2  # X^2-Y^2
XY = X * Y  # XY

Z3 = Z**3 - 1.5 * Z * (X**2 + Y**2)  # Z^3-3/2Z(X^2+Y^2)
Z2X = X * (4 * Z**2 - (X**2 + Y**2))  # X(4Z^2-(X^2+Y^2))
Z2Y = Y * (4 * Z**2 - (X**2 + Y**2))  # Y(4Z^2-(X^2+Y^2))
ZX2 = Z * (X**2 - Y**2)  # Z(X^2-Y^2)
XYZ = Z * X * Y  # ZXY
X3 = X**3 - 3 * Y**2 * X  # X^3-3Y^2X
Y3 = -Y**3 + 3 * X**2 * Y  # -Y^3+3X^2Y

Z4 = 8 * Z**4 - 24 * Z**2 * (X**2 + Y**2) + 3 * (X**4 + Y**4 + 2 * X**2 * Y**2)
Z3X = 4 * Z**3 * X - 3 * (X**3 * Z + X * Y**2 * Z)
Z3Y = 4 * Z**3 * Y - 3 * (Y**3 * Z + Y * X**2 * Z)
Z2X2 = 6 * Z**2 * (X**2 - Y**2) - X**4 + Y**4
Z2XY = 6 * Z**2 * X * Y - X**3 * Y - X * Y**3
ZX3 = Z * X**3 - 3 * Y**2 * X * Z
ZY3 = -Z * Y**3 + 3 * X**2 * Y * Z
X4 = X**4 - 6 * X**2 * Y**2 + Y**4
Y4 = Y * X**3 - X * Y**3

# Create F matrix with proper dimensions
F = np.column_stack((Z0, Z, X, Y, Z2, ZX, ZY, X2, XY))

Hx = (np.max(Bm_x) - np.min(Bm_x)) / (np.max(Bm_x) + np.min(Bm_x)) * 2 * 1e6

print(f'Hx = {Hx} ppm')
Bm_x_ave = (np.max(Bm_x) + np.min(Bm_x)) / 2

print(f'Bx_ave = {Bm_x_ave} ppm')
print(np.max(Bm_x))
print(np.min(Bm_x))
np.save('F.npy', F)

# Regression analysis - fixing the regression part
# In Python, we need to use a slightly different approach for multivariate regression
model = np.linalg.lstsq(F, Bm_x, rcond=None)
b_factor1 = model[0]  # This gives the coefficients

y_reg_X = F @ b_factor1  # Matrix multiplication for prediction

# Element-wise multiplication between each column of F and the corresponding coefficient
B_sh = np.zeros_like(F)
for i in range(F.shape[1]):
    B_sh[:, i] = F[:, i] * b_factor1[i]

H_sh = np.zeros(F.shape[1])
for i in range(F.shape[1]):
    column_data = B_sh[:, i]  # Get data from column i
    max_val = np.max(column_data)  # Max value
    min_val = np.min(column_data)  # Min value
    H_sh[i] = (max_val - min_val) / Bm_x_ave * 1e6

# Plot initial field
plt.figure(figsize=(10, 6))
plt.plot(Bm_x, '-r')
plt.legend(['Initial Field'])
plt.xlabel('Sample Points')
plt.ylabel('Bm / mT')
plt.show()

# Calculate coefficients using least squares - we've already done this above
coefficients = b_factor1  # reusing the same coefficients

# Each coefficient's corresponding magnetic field value
B_contributions = np.zeros_like(F)
for i in range(F.shape[1]):
    B_contributions[:, i] = F[:, i] * coefficients[i]

# Calculate contribution to uniformity
B_contributions_max = np.max(B_contributions, axis=0)
B_contributions_min = np.min(B_contributions, axis=0)
B_contributions_avg = np.mean(B_contributions, axis=0)
# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    uniformity_contributions = (B_contributions_max - B_contributions_min) / B_contributions_avg * 1e6
    # Replace inf/NaN values with 0
    uniformity_contributions = np.nan_to_num(uniformity_contributions, nan=0, posinf=0, neginf=0)

B_z = coefficients[1] * Z
B_x = coefficients[2] * X
B_y = coefficients[3] * Y
B_z2 = coefficients[4] * Z2
B_zx = coefficients[5] * ZX
B_zy = coefficients[6] * ZY
B_x2 = coefficients[7] * X2
B_xy = coefficients[8] * XY

# Calculate inhomogeneity with safe division
def safe_inhom(field):
    max_val = np.max(field)
    min_val = np.min(field)
    sum_val = max_val + min_val
    if sum_val != 0:
        return (max_val - min_val) / sum_val * 2 * 1e6
    return 0

inhomogeneity = [
    safe_inhom(B_z),
    safe_inhom(B_x),
    safe_inhom(B_y),
    safe_inhom(B_z2),
    safe_inhom(B_zx),
    safe_inhom(B_zy),
    safe_inhom(B_x2),
    safe_inhom(B_xy)
]

# Bar plot for b_factor
plt.figure(figsize=(12, 6))
b_factor_fig_X = b_factor1 * 1e3  # unit uT
plt.bar(range(len(b_factor_fig_X) - 1), b_factor_fig_X[1:])
plt.xticks(range(len(b_factor_fig_X) - 1), 
          ['Z', 'X', 'Y', 'Z^2', 'ZX', 'ZY', 'X^2', 'XY'])
plt.xlabel('Spatial Dependence of Bx')
plt.ylabel('Amplitude / uT')
plt.show()

# Bar plot for H_sh
plt.figure(figsize=(12, 6))
plt.bar(range(len(H_sh) - 1), H_sh[1:])
plt.xticks(range(len(H_sh) - 1), 
          ['Z', 'X', 'Y', 'Z^2', 'ZX', 'ZY', 'X^2', 'XY'])
plt.xlabel('SH components before PS')
plt.ylabel('inhomogeneity / ppm')
plt.show()

# Calculate H_error
H_error = np.zeros(F.shape[1] - 1)
for j in range(1, F.shape[1]):
    mean_val = np.mean(F[:, j])
    if mean_val != 0:
        H_temp = (np.max(F[:, j]) - np.min(F[:, j])) / mean_val * 1e6
    else:
        H_temp = 0
    H_error[j - 1] = H_temp