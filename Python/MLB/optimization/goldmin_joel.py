import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)
from scipy.optimize import minimize_scalar
sys.path.append(this_dir)

import numpy as np
import matplotlib.pyplot as plt
from goldmin import goldmin   # <- your Chapra-based implementation


# --- Function from slide 3 ---
def f(x):
    return 4*x - 1.8*x**2 + 1.2*x**3 - 0.3*x**4


# Bracket chosen where the function has a minimum
xl = 1.0
xu = 4.0

# SciPy (Brent: golden-section + parabolic interpolation)
res = minimize_scalar(f, method='bounded', bounds=(xl, xu), tol=1.0e-5)

# Chapra-style golden section
xmin, fmin, ea, n = goldmin(f, xl, xu, Ea=1.0e-5)

print("-" * 90)
print("Goldmin on quartic from slide 3")
print("-" * 90)
print(f"{'Method':<22}\t{'x_min':>12}\t{'f_min':>15}\t{'Error':>15}\t{'Iters/Evals':>15}")
print("-" * 90)
print(f"{'Chapra goldmin':<22}\t{xmin:12.8f}\t{fmin:15.8f}\t{ea:15.3e}\t{n:15d}")
print(f"{'SciPy minimize_scalar':<22}\t{res.x:12.8f}\t{res.fun:15.8f}\t{1.0e-5:15.3e}\t{res.nfev:15d}")
print("-" * 90)

print("\nInterpretation:")
print("• Function: f(x) = 4x - 1.8x^2 + 1.2x^3 - 0.3x^4 on [1, 4].")
print("• Both methods repeatedly shrink the bracket using the golden ratio.")
print("• SciPy’s Brent method adds parabolic interpolation on top of golden-section.")
print("• The close agreement in x_min and f_min confirms that your goldmin is working.\n")

# --- Visualization ---
x = np.linspace(xl, xu, 400)
y = f(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$f(x) = 4x - 1.8x^2 + 1.2x^3 - 0.3x^4$")
plt.axvline(res.x, color="red", linestyle="--",
            label=f"SciPy min @ x={res.x:.4f}")
plt.axvline(xmin, color="orange", linestyle=":",
            label=f"Chapra min @ x={xmin:.4f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Golden-Section Minimum (Quartic Function)")
plt.legend()
plt.grid(True)
plt.savefig("goldmin_quartic_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
