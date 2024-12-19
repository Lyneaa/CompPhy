import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 1  # p(x) = 1 
q = 2  # q(x) = 2 
g_0 = -0.3  # x=0
g_1 = -0.1  # x=pi/2

# Exact solution 
def exact_solution(x):
    return -(1/10) * (np.sin(x) + 3 * np.cos(x))

# RK4 method
def rk4_method(f, g, a, b, u_a, v_a, h, num_steps):
    x_vals = np.linspace(a, b, num_steps)
    u_vals = np.zeros(num_steps)
    v_vals = np.zeros(num_steps)
    u_vals[0] = u_a
    v_vals[0] = v_a
    
    for i in range(1, num_steps):
        k1u = h * v_vals[i - 1]
        k1v = h * (v_vals[i - 1] + 2 * u_vals[i - 1] + np.cos(x_vals[i - 1]))

        k2u = h * (v_vals[i - 1] + 0.5 * k1v)
        k2v = h * (v_vals[i - 1] + 0.5 * k1v + 2 * (u_vals[i - 1] + 0.5 * k1u) + np.cos(x_vals[i - 1] + 0.5 * h))

        k3u = h * (v_vals[i - 1] + 0.5 * k2v)
        k3v = h * (v_vals[i - 1] + 0.5 * k2v + 2 * (u_vals[i - 1] + 0.5 * k2u) + np.cos(x_vals[i - 1] + 0.5 * h))

        k4u = h * (v_vals[i - 1] + k3v)
        k4v = h * (v_vals[i - 1] + k3v + 2 * (u_vals[i - 1] + k3u) + np.cos(x_vals[i - 1] + h))

        u_vals[i] = u_vals[i - 1] + (1/6) * (k1u + 2*k2u + 2*k3u + k4u)
        v_vals[i] = v_vals[i - 1] + (1/6) * (k1v + 2*k2v + 2*k3v + k4v)

    return x_vals, u_vals, v_vals

# Shooting method
def shooting_method(g_0, g_1, a, b, h, tol=1e-5, max_iter=100):
    u_a, u_b = g_0, g_1
    v_guess = 0.0  # Initial guess for u'(0)
    num_steps = int((b - a) / h) + 1
    
    for _ in range(max_iter):
        x_vals, u_vals, v_vals = rk4_method(None, None, a, b, u_a, v_guess, h, num_steps)
        u_b_computed = u_vals[-1]
        if np.abs(u_b_computed - u_b) < tol:
            print(f"Converged after {_+1} iterations.")
            return x_vals, u_vals
        v_guess -= (u_b_computed - u_b) * 0.1  # Proportional adjustment
    print("Max iterations reached.")
    return x_vals, u_vals

# Interval [a, b]
a = 0
b = np.pi / 2

# Step size
h1 = np.pi / 4
h2 = np.pi / 8

# Solve BVP
x_vals_h1, u_vals_h1 = shooting_method(g_0, g_1, a, b, h1)
x_vals_h2, u_vals_h2 = shooting_method(g_0, g_1, a, b, h2)

# Exact solution
x_vals_exact = np.linspace(a, b, 1000)
u_exact = exact_solution(x_vals_exact)

# Finite difference
def f(x):
    return -np.cos(x)

h_values = [np.pi / 4, np.pi / 8, np.pi/32]
n_values = [int((np.pi/2) / h) + 1 for h in h_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals_exact, u_exact, label='Exact Solution', linestyle='--', color='black')

# Plot RK4
plt.plot(x_vals_h1, u_vals_h1, label=f'RK4 Solution (h={h1:.4f})', marker='o', markersize=6)
plt.plot(x_vals_h2, u_vals_h2, label=f'RK4 Solution (h={h2:.4f})', marker='o', markersize=6)

# Solve and plot the FDM
for h in h_values:
    n = int((np.pi/2) / h) + 1
    A = np.zeros((n - 2, n - 2))
    b = np.zeros(n - 2)

    for i in range(n - 2):
        xi = i * h
        A[i, i] = 2 + h**2 * q
        if i > 0:
            A[i, i - 1] = -(1 + (h/2) * p)
        if i < n - 3:
            A[i, i + 1] = -(1 - (h/2) * p)

    for i in range(n - 2):
        xi = (i + 1) * h
        b[i] = h**2 * f(h * (i+1))
    b[0] = (1 + 1/2 * p * h) * g_0 + h**2 * f(h) 
    b[-1] = (1 - 1/2 * p * h) * g_1 + h**2 * f(1 - h) 

    u_inner = np.linalg.solve(A, b)
    u_full = np.concatenate(([g_0], u_inner, [g_1]))
    x_grid = np.linspace(0, np.pi / 2, n)
    plt.plot(x_grid, u_full, label=f'Finite Diff. (h={h:.4f})', marker='x', markersize=6)


plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of Finite Difference, Shooting (RK4) and Exact Solutions')
plt.legend()
plt.grid(True)
plt.show()
