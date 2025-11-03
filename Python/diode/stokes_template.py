import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0     # cavity length
k = 0.2     # nonlinearity constant
U = 1.0     # lid velocity

# Streamfunction definition
def psi(A, X, Y):
    return A * X * (L - X) * Y * (L - Y)

def velocity_field(A, X, Y):
    """Compute u and v from psi"""
    u = A * (L - X) * (L - 2 * Y)
    v = -A * (L - Y) * (L - 2 * X)
    return u, v

# Nonlinear function and derivative
def f(A):
    return A * (L**2 / 4) * (1 + k * A**2) - U / 2

def f_prime(A):
    return (L**2 / 4) * (1 + 3 * k * A**2)

# Newton-Raphson
def newton_raphson(f, f_prime, A0, tol=1e-8, max_iter=50):
    A = A0
    for i in range(max_iter):
        f_val = f(A)
        fp_val = f_prime(A)
        A_new = A - f_val / fp_val
        if abs(A_new - A) < tol:
            print(f"Converged in {i+1} iterations.")
            return A_new
        A = A_new
    raise ValueError("Did not converge.")

# Solve for A
A0 = 1.0
A_solution = newton_raphson(f, f_prime, A0)
print(f"A_solution = {A_solution:.6f}")

# Plot f(A) and f'(A)
A_range = np.linspace(-2, 2, 200)
f_values = f(A_range)
fp_values = f_prime(A_range)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(A_range, f_values, color='blue', label="f(A)")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("A"); plt.ylabel("f(A)")
plt.legend(); plt.grid(True); plt.title("Nonlinear Equation f(A)")

plt.subplot(1,2,2)
plt.plot(A_range, fp_values, color='red', label="f'(A)")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("A"); plt.ylabel("f'(A)")
plt.legend(); plt.grid(True); plt.title("Derivative f'(A)")
plt.tight_layout()
plt.show()

# Plot streamlines
Nx, Ny = 50, 50
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
Psi = psi(A_solution, X, Y)
u, v = velocity_field(A_solution, X, Y)

plt.figure(figsize=(6,6))
contours = plt.contourf(X, Y, Psi, levels=30, cmap='viridis')
plt.colorbar(contours, label="ψ(x,y)")
step = 3
plt.quiver(X[::step, ::step], Y[::step, ::step], u[::step, ::step], v[::step, ::step], color='white', scale=10)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Lid-driven Cavity Flow Field")
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
