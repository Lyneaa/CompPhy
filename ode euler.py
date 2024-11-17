import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return y * (1 - np.exp(2 * x))
    #return x - 0.5 * np.exp(2 * x) + 0.5

x_0 = 0
y_0 = 1
h = 1 / 4
iterations = 1 / h
n = int(iterations)

x_val = np.zeros(n + 1) # or np.linspace(0, 1, 5)
y_val = np.zeros(n + 1)
x_val[0] = x_0
y_val[0] = y_0

for i in range(1, n+1):
    x_val[i] = x_val[i - 1] + h
    y_val[i] = y_val[i - 1] + h * f(x_val[i - 1], y_val[i - 1])
    print(f"Iteration {i} for x : {x_val[i]} , y : {y_val[i]}")
    
plt.plot(x_val, y_val, label = "Euler Method", marker = "o")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Euler Method")
plt.legend()
plt.grid()
plt.show()
