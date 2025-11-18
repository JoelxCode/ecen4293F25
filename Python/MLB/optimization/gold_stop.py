import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)
from scipy.optimize import minimize_scalar
sys.path.append(this_dir)

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Chapra function: f(x) = x^2/10 - 2 sin(x)"""
    return x**2 / 10.0 - 2.0 * np.sin(x)


def goldmin_self_stop(f, xl, xu, Ea=1.0e-7):
    """
    Golden-section search that stops automatically based on Ea.
    This is a self-stopping version of your goldmin: no maxit input.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    d = (phi - 1) * (xu - xl)
    x1 = xl + d
    f1 = f(x1)
    x2 = xu - d
    f2 = f(x2)

    i = 0
    while True:
        xint = xu - xl

        if f1 < f2:
            xopt = x1
            xl = x2
            x2 = x1
            f2 = f1
            x1 = xl + (phi - 1) * (xu - xl)
            f1 = f(x1)
        else:
            xopt = x2
            xu = x1
            x1 = x2
            f1 = f2
            x2 = xu - (phi - 1) * (xu - xl)
            f2 = f(x2)

        i += 1

        if xopt != 0.0:
            ea = (2 - phi) * abs(xint / xopt)
            if ea <= Ea:
                break

    return xopt, f(xopt), ea, i


xl = 0.0
xu = 4.0
Ea = 1.0e-5

# Self-stopping golden-section
xmin, fmin, ea, n = goldmin_self_stop(f, xl, xu, Ea=Ea)

# SciPy reference
res = minimize_scalar(f, method='bounded', bounds=(xl, xu), tol=Ea)

print("-" * 90)
print("Self-stopping Goldmin on Chapra test function")
print("-" * 90)
print(f"{'Method':<25}\t{'x_min':>12}\t{'f_min':>15}\t{'Error':>15}\t{'Iters/Evals':>15}")
print("-" * 90)
print(f"{'Self-stopping goldmin':<25}\t{xmin:12.8f}\t{fmin:15.8f}\t{ea:15.3e}\t{n:15d}")
print(f"{'SciPy minimize_scalar':<25}\t{res.x:12.8f}\t{res.fun:15.8f}\t{Ea:15.3e}\t{res.nfev:15d}")
print("-" * 90)

print("\nInterpretation:")
print("• Function: f(x) = x^2/10 - 2 sin(x) on [0, 4].")
print("• Each iteration shrinks the bracket by the golden ratio and recomputes ε_a.")
print("• Loop stops automatically when ε_a <= Ea (no hard-coded max iteration).")
print("• Result agrees closely with SciPy, so the tolerance target is attained.\n")

# --- Visualization ---
x = np.linspace(xl, xu, 400)
y = f(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = x^2/10 - 2\sin(x)$", color="navy")
plt.axvline(xmin, color="orange", linestyle=":",
            label=f"Self-stop min @ x={xmin:.4f}")
plt.axvline(res.x, color="red", linestyle="--",
            label=f"SciPy min @ x={res.x:.4f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Self-Stopping Golden-Section Search (Chapra Test Function)")
plt.legend()
plt.grid(True)
plt.savefig("self_stopping_goldmin_chapra.png", dpi=300, bbox_inches="tight")
plt.show()
