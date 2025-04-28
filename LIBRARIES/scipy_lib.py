from scipy.optimize import minimize

def func(x):
    return x**2 + 3*x + 2

result = minimize(func, x0=0)
print("Minimum value at x =", result.x)
