import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
x_start, x_end = 0, 1  # Boundaries
h = 0.05  # Step size
x_values = np.arange(x_start, x_end + h, h) 

# Exact solution
def exact_solution(x):
    return np.log((np.e - 1) * x + 1)

# u'(x) = v(x), v'(x) = -v(x)^2
def system_of_equations(u, v):
    du_dx = v
    dv_dx = -v**2
    return du_dx, dv_dx

# Forward integration
def forward_integration(v0):
    u, v = 0, v0  # Initial conditions
    u_values = [u]
    for x in x_values[:-1]:
        du_dx, dv_dx = system_of_equations(u, v)
        u += h * du_dx
        v += h * dv_dx
        u_values.append(u)
    return u_values

# Secant method
def secant_method(v0, v1, tolerance=1e-6, max_iterations=100):
    for _ in range(max_iterations):
        # Integrate forward for both guesses
        u_values_v0 = forward_integration(v0)
        u_values_v1 = forward_integration(v1)
        
        # Evaluate the boundary condition error at x=1
        f_v0 = u_values_v0[-1] - 1  # u(1) - 1
        f_v1 = u_values_v1[-1] - 1  # u(1) - 1
        
        # Secant update for v0 and v1
        v_new = v1 - f_v1 * (v1 - v0) / (f_v1 - f_v0)
        if abs(v_new - v1) < tolerance:
            return v_new, forward_integration(v_new)
        
        v0, v1 = v1, v_new
    raise RuntimeError("Secant method did not converge")

# Initial guesses for v(0)
v0_guess, v1_guess = 1.0, 1.5

# Solve using secant method
v0_final, u_numerical = secant_method(v0_guess, v1_guess)

u_exact = exact_solution(x_values)

# Tabulate results
results = np.column_stack((x_values, u_numerical, u_exact))

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(x_values, u_numerical, 'o-', label="Numerical Solution")
plt.plot(x_values, u_exact, 'x--', label="Exact Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Comparison of Numerical and Exact Solutions")
plt.legend()
plt.grid(True)
plt.show()

# Results in table
df_results = pd.DataFrame(results, columns=["x", "Numerical u(x)", "Exact u(x)"])
print(df_results)