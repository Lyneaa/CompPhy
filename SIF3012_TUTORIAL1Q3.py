import numpy as np
import matplotlib.pyplot as plt

# Parameters
max_iterations = 100  # Max Iterations
tolerance = 1e-6  # Tolerance

h = 0.1  # Step size
x_start, x_end = 1, 2  # Boundaries
u_start, u_end = 1/2, 1/3  # Boundary conditions


x_values = np.arange(x_start, x_end + h, h)
n = len(x_values)  # Number of grid points

# Initial guess for u (linear interpolation between boundary values)
u_guess = np.linspace(u_start, u_end, n)

# Initial guess for u (linear interpolation)
u_current = np.linspace(u_start, u_end, n)

# Residuals (F)
def residuals(u):
    res = np.zeros(n)
    res[0] = u[0] - u_start  # x=1
    res[-1] = u[-1] - u_end  # x=2
    for i in range(1, n-1):
        u_i, u_ip1, u_im1 = u[i], u[i+1], u[i-1]
        res[i] = (u_im1 - 2*u_i + u_ip1) / h**2 - u_i**3 + u_i * (u_ip1 - u_im1) / (2*h)
    return res

# Function to compute the Jacobian matrix (J)
def jacobian(u):
    J = np.zeros((n, n))
    J[0, 0] = 1  # x=1
    J[-1, -1] = 1  # x=2
    for i in range(1, n-1):
        u_i, u_ip1, u_im1 = u[i], u[i+1], u[i-1]
        J[i, i-1] = 1 / h**2 - u_i / (2*h)  # Partial derivative u[i-1]
        J[i, i] = -2 / h**2 - 3 * u_i**2 + (u_ip1 - u_im1) / (2*h)  # Partial derivative u[i]
        J[i, i+1] = 1 / h**2 + u_i / (2*h)  # Partial derivative u[i+1]
    return J

# Newton-Raphson
for iteration in range(max_iterations):
    F = residuals(u_current)  # Compute residuals
    if np.linalg.norm(F, ord=np.inf) < tolerance:
        break
    J = jacobian(u_current)  # Jacobian matrix
    delta_u = np.linalg.solve(J, -F)  
    u_current += delta_u  
    # Check for convergence
    if np.linalg.norm(delta_u, ord=np.inf) < tolerance:
        break

# Exact solution
u_exact = 1 / (x_values + 1)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, u_current, 'o-', label="Numerical Solution ")
plt.plot(x_values, u_exact, 'x--', label="Exact Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Comparison of Numerical and Exact Solutions (Newton-Raphson)")
plt.legend()
plt.grid(True)
plt.show()

print(x_values, u_current, u_exact)