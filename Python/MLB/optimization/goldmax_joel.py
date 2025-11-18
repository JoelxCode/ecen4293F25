import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)
from scipy.optimize import minimize_scalar
sys.path.append(this_dir)

import numpy as np
import matplotlib.pyplot as plt
from goldmin import goldmin   # reuse your function


def f(x):
    """Same quartic as before."""
    return 4*x - 1.8*x**2 + 1.2*x**3 - 0.3*x**4


def goldmax(f, xl, xu, Ea=1.0e-7, maxit=30):
    """
    Golden-section search for a maximum using your goldmin.
    We minimize -f(x) and then flip the sign back.
    """
    def g(x):
        return -f(x)  # turn max problem into min problem

    xopt, gmin, ea, n = goldmin(g, xl, xu, Ea=Ea, maxit=maxit)
    return xopt, -gmin, ea, n


xl = 1.0
xu = 4.0

# SciPy: maximize by minimizing -f
res = minimize_scalar(lambda x: -f(x), method='bounded',
                      bounds=(xl, xu), tol=1.0e-5)

xmax, fmax, ea, n = goldmax(f, xl, xu, Ea=1.0e-5)

print("-" * 90)
print("Goldmax on quartic from slide 3")
print("-" * 90)
print(f"{'Method':<22}\t{'x_max':>12}\t{'f_max':>15}\t{'Error':>15}\t{'Iters/Evals':>15}")
print("-" * 90)
print(f"{'Chapra goldmax':<22}\t{xmax:12.8f}\t{fmax:15.8f}\t{ea:15.3e}\t{n:15d}")
print(f"{'SciPy minimize_scalar':<22}\t{res.x:12.8f}\t{-res.fun:15.8f}\t{1.0e-5:15.3e}\t{res.nfev:15d}")
print("-" * 90)

print("\nInterpretation:")
print("• We maximize f(x) by minimizing -f(x).")
print("• goldmax just wraps your goldmin and flips the sign.")
print("• Again the SciPy and Chapra results are very close, validating the implementation.\n")

# --- Visualization ---
x = np.linspace(xl, xu, 400)
y = f(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = 4x - 1.8x^2 + 1.2x^3 - 0.3x^4$")
plt.axvline(xmax, color="green", linestyle="--",
            label=f"Chapra max @ x={xmax:.4f}")
plt.axvline(res.x, color="purple", linestyle=":",
            label=f"SciPy max @ x={res.x:.4f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Golden-Section Maximum (Quartic Function)")
plt.legend()
plt.grid(True)
plt.savefig("goldmax_quartic_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
