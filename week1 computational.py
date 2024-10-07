import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 - 2 * x * np.exp(-x/2)

def df(x):
    return x * np.exp(-x/2) - 2 * np.exp(-x/2)

x = 0
tolerance = 1.0e-5
max_iterations = 100

for i in range (max_iterations):
    xi = x - f(x)/df(x) 
    delx = abs(xi-x)
    x = xi
    if delx < tolerance:
      break
    print(i,xi,f(x),delx)
    plt.plot(x,f(x),'bx--')


x_vals = np.linspace(0, 10, 100)
f_vals = f(x_vals)

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Newton-Raphson Method")
plt.plot(x_vals, f_vals, 'g-', label='f(x)')
plt.axhline(0, color='k', lw=0.5)  
plt.axvline(x=x, color='b', linestyle='--', label=f'Root: x={x:.4f}')
plt.legend()
plt.grid()
plt.show()