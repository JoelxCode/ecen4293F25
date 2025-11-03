import numpy as np
import matplotlib.pyplot as plt

# Parameters
V_s = 5.0    # Source voltage (V)
R = 1000.0   # Resistor (Ohms)
I_s = 1e-12  # Saturation current (A)
V_T = 0.025  # Thermal voltage (V)


def f(V):
    """ Diode equation: represents KVL around the loop """
    return V_s - R * I_s * (np.exp(V / V_T) - 1) - V


def f_prime(V):
    """ Derivative of f(V) wrt V """
    return -R * I_s * (np.exp(V / V_T) / V_T) - 1


# Newton-Raphson method
def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=100):
    V = x0
    for i in range(max_iter):
        f_val = f(V)
        fp_val = f_prime(V)
        V_new = V - f_val / fp_val
        if abs(V_new - V) < tol:
            print(f"Converged in {i+1} iterations.")
            return V_new
        V = V_new
    raise ValueError("Did not converge.")


# Initial guess
V0 = 0.7
V_solution = newton_raphson(f, f_prime, V0)
print(f"Diode voltage (Vd) = {V_solution:.6f} V")

# Plot f(V) and f'(V)
V_range = np.linspace(0, 1, 200)
f_values = f(V_range)
f_prime_values = f_prime(V_range)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(V_range, f_values, label="f(V)", color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("V (V)")
plt.ylabel("f(V)")
plt.title("Diode Equation f(V)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(V_range, f_prime_values, label="f'(V)", color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("V (V)")
plt.ylabel("f'(V)")
plt.title("Derivative of f(V)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()