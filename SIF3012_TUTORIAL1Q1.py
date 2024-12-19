import numpy as np
import matplotlib.pyplot as plt

# parameters
a = 0      # Left boundary
b = 1      # Right boundary
u_a = 0    # u(0) = 0
u_b = 2    # u(1) = 1

# Exact solution
def exact_solution(x):
    return np.exp(2) * (1/(np.exp(4)-1)) * (np.exp(2*x)-np.exp(-2*x)) + x

# Shooting method with RK4
def rk4_shooting(f1, f2, a, b, u_a, v_guess, h):
    n = int((b - a) / h) + 1
    x = np.linspace(a, b, n)
    u = np.zeros(n)
    v = np.zeros(n)
    
    u[0] = u_a
    v[0] = v_guess
    
    for i in range(1, n):
        xi = x[i - 1]
        
        k1u = h * f1(u[i - 1], v[i - 1], xi)
        k1v = h * f2(u[i - 1], v[i - 1], xi)
        
        k2u = h * f1(u[i - 1] + 0.5 * k1u, v[i - 1] + 0.5 * k1v, xi + 0.5 * h)
        k2v = h * f2(u[i - 1] + 0.5 * k1u, v[i - 1] + 0.5 * k1v, xi + 0.5 * h)
        
        k3u = h * f1(u[i - 1] + 0.5 * k2u, v[i - 1] + 0.5 * k2v, xi + 0.5 * h)
        k3v = h * f2(u[i - 1] + 0.5 * k2u, v[i - 1] + 0.5 * k2v, xi + 0.5 * h)
        
        k4u = h * f1(u[i - 1] + k3u, v[i - 1] + k3v, xi + h)
        k4v = h * f2(u[i - 1] + k3u, v[i - 1] + k3v, xi + h)
        
        u[i] = u[i - 1] + (k1u + 2 * k2u + 2 * k3u + k4u) / 6
        v[i] = v[i - 1] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    
    return x, u

# System of ODEs for shooting method
def f1(u, v, x):
    return v

def f2(u, v, x):
    return 4 * (u - x)

# initial guess for v(0) in the shooting method
def shooting_method(a, b, u_a, u_b, h, tol=1e-6):
    v_guess = 0.0  # Initial guess
    max_iter = 100
    for _ in range(max_iter):
        x, u = rk4_shooting(f1, f2, a, b, u_a, v_guess, h)
        error = u[-1] - u_b
        if abs(error) < tol:
            print(f"Shooting method converged with v'(0) = {v_guess}")
            return x, u
        v_guess -= error * 0.1  # Proportional adjustment
    print("Shooting method did not converge.")
    return x, u

# Finite difference method
def finite_difference(a, b, u_a, u_b, h):
    n = int((b - a) / h) + 1  # Number of grid points
    x = np.linspace(a, b, n)
    
    # Matrix and RHS construction
    A = np.zeros((n - 2, n - 2))
    b_vec = np.zeros(n - 2)
    
    for i in range(n - 2):
        xi = x[i + 1]
        A[i, i] = 2 + 4 * h**2  # Diagonal 
        if i > 0:
            A[i, i - 1] = -1    # Lower diagonal
        if i < n - 3:
            A[i, i + 1] = -1    # Upper diagonal
        b_vec[i] = 4 * xi * h**2  # RHS contribution
    
    # Boundary conditions
    b_vec[0] += u_a  # Left boundary 
    b_vec[-1] += u_b  # Right boundary c
    
    # Solve
    u_inner = np.linalg.solve(A, b_vec)
    
    # Combine with boundary values
    u_full = np.concatenate(([u_a], u_inner, [u_b]))
    return x, u_full

# Step sizes
h_values = [1/4, 1/8, 1/16]


plt.figure(figsize=(12, 8))

# Plot exact solution
x_exact = np.linspace(a, b, 1000)
u_exact = exact_solution(x_exact)
plt.plot(x_exact, u_exact, label="Exact Solution", linestyle="--", color="black")

# Solve and plot finite difference solutions for different step sizes
for h in h_values:
    x_fd, u_fd = finite_difference(a, b, u_a, u_b, h)
    plt.plot(x_fd, u_fd, label=f"Finite Diff. (h={h:.4f})", marker="o", markersize=4)

for h in h_values:
    x_sh, u_sh = shooting_method(a, b, u_a, u_b, h)
    plt.plot(x_sh, u_sh, label=f"Shooting (RK4) (h={h:.4f})", marker="x", markersize=4)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Comparison of Finite Difference, Shooting (RK4), and Exact Solutions")
plt.legend()
plt.grid(True)
plt.show()
