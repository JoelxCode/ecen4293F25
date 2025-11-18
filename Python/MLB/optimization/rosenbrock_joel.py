import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)
from scipy.optimize import minimize
sys.path.append(this_dir)

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Rosenbrock function and gradient
# ----------------------------------------------------------------------
def rosen(xy):
    """
    Rosenbrock 'banana' function:
        f(x, y) = 100 (y - x^2)^2 + (1 - x)^2
    xy: array-like of length 2
    """
    x, y = xy
    return 100.0 * (y - x**2)**2 + (1.0 - x)**2


def grad_rosen(xy):
    """
    Gradient of the Rosenbrock function.
    """
    x, y = xy
    dfdx = -400.0 * x * (y - x**2) - 2.0 * (1.0 - x)
    dfdy = 200.0 * (y - x**2)
    return np.array([dfdx, dfdy])


# ----------------------------------------------------------------------
# Steepest Descent with backtracking line search
# ----------------------------------------------------------------------
def steepest_descent(f, grad_f, x0, tol=1.0e-6, max_iter=10_000,
                     alpha0=1.0, rho=0.5, c1=1.0e-4):
    """
    Steepest descent (gradient descent) with simple Armijo backtracking.

    Parameters
    ----------
    f       : objective function
    grad_f  : gradient function
    x0      : initial guess (length-2 array-like)
    tol     : gradient norm tolerance
    max_iter: maximum iterations
    alpha0  : initial step size for each line search
    rho     : step size reduction factor (0<rho<1)
    c1      : Armijo condition constant

    Returns
    -------
    x_best  : final point
    f_best  : final objective value
    gnorm   : final gradient norm
    iters   : number of outer iterations
    path    : (N, 2) array of iterates for plotting
    """
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            break

        d = -g   # steepest descent direction
        alpha = alpha0

        # Backtracking line search (Armijo)
        f_x = f(x)
        while True:
            x_new = x + alpha * d
            if f(x_new) <= f_x + c1 * alpha * np.dot(g, d):
                break
            alpha *= rho
            if alpha < 1.0e-12:   # safeguard
                break

        x = x_new
        path.append(x.copy())

    x_best = x
    f_best = f(x_best)
    gnorm = np.linalg.norm(grad_f(x_best))
    return x_best, f_best, gnorm, k + 1, np.array(path)


# ----------------------------------------------------------------------
# Run comparison: Steepest Descent vs BFGS on Rosenbrock
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initial guess (you can change this to see different behavior)
    x0 = np.array([-1.5, 1.5])

    # --- Custom steepest descent ---
    x_sd, f_sd, gnorm_sd, n_sd, path_sd = steepest_descent(
        rosen, grad_rosen, x0,
        tol=1.0e-6, max_iter=20_000,
        alpha0=1.0, rho=0.5, c1=1.0e-4
    )

    # --- SciPy BFGS (with callback to record path) ---
    path_bfgs = []

    def callback_bfgs(xk):
        path_bfgs.append(xk.copy())

    res = minimize(
        rosen,
        x0,
        method="BFGS",
        jac=grad_rosen,
        callback=callback_bfgs,
        options={"gtol": 1.0e-6, "disp": False}
    )
    x_bfgs = res.x
    f_bfgs = res.fun
    gnorm_bfgs = np.linalg.norm(grad_rosen(x_bfgs))
    n_bfgs = res.nit
    path_bfgs = np.array(path_bfgs)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("-" * 90)
    print("Rosenbrock Optimization: Steepest Descent vs BFGS")
    print("-" * 90)
    print(f"{'Method':<20}\t{'x* (x1,x2)':>24}\t{'f(x*)':>15}\t{'||grad||':>15}\t{'Iterations':>10}")
    print("-" * 90)
    print(f"{'Steepest Descent':<20}\t"
          f"[{x_sd[0]:9.5f}, {x_sd[1]:9.5f}]\t"
          f"{f_sd:15.8f}\t{gnorm_sd:15.3e}\t{n_sd:10d}")
    print(f"{'BFGS (SciPy)':<20}\t"
          f"[{x_bfgs[0]:9.5f}, {x_bfgs[1]:9.5f}]\t"
          f"{f_bfgs:15.8f}\t{gnorm_bfgs:15.3e}\t{n_bfgs:10d}")
    print("-" * 90)

    print("\nInterpretation:")
    print("• Rosenbrock has a narrow curved valley; plain steepest descent makes many zig-zag steps.")
    print("• BFGS builds an approximate Hessian, so its steps align better with the valley and converge faster.")
    print("• Both should converge near the true minimum at (1, 1) with f(1,1) = 0.\n")

    # ------------------------------------------------------------------
    # Contour plot + paths
    # ------------------------------------------------------------------
    # Grid for contours
    x1 = np.linspace(-2.0, 2.0, 400)
    x2 = np.linspace(-1.0, 3.0, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 100.0 * (X2 - X1**2)**2 + (1.0 - X1)**2

    plt.figure(figsize=(7, 6))
    # Log-spaced levels show the valley better
    levels = np.logspace(-1, 3, 20)
    cs = plt.contour(X1, X2, Z, levels=levels, linewidths=0.8)
    plt.clabel(cs, inline=1, fontsize=8)

    # Plot paths
    plt.plot(path_sd[:, 0], path_sd[:, 1], "r.-", label="Steepest Descent")
    if path_bfgs.size > 0:
        plt.plot(path_bfgs[:, 0], path_bfgs[:, 1], "b.-", label="BFGS")

    # Mark start and final points
    plt.plot(x0[0], x0[1], "ko", label="Start")
    plt.plot(1.0, 1.0, "gx", markersize=10, label="True minimum (1,1)")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Rosenbrock Function: Steepest Descent vs BFGS Paths")
    plt.legend()
    plt.grid(True)
    plt.savefig("rosenbrock_sd_bfgs_paths.png", dpi=300, bbox_inches="tight")
    plt.show()

