import numpy as np
import matplotlib.pyplot as plt

# Step 2: Define the function
def f(x, y):
    return np.sin(x) * np.cos(y)

# Step 3: Create the grid of values
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

# Step 4: Evaluate the function on the grid
Z = f(X, Y)

# Step 5: Create and display the heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, cmap='RdBu')
plt.colorbar(label='Function Value')
plt.title('Heatmap of $f(x, y) = \sin(x) \cdot \cos(y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()