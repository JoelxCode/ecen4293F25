import numpy as np
import matplotlib.pyplot as plt
#Notes: Nump is a packafr for vetor and matrix manipulation. 
#Scipy is  a packahe for scientific and technical computing.

#They make data manipulation run faster for a variety of reasons that I don't fully understand.

#This is the assignment that I am going to submit for ECEN. The task is to use this starter document to create 4 different plots using matplotlib that look like the one provided in the assignment pdf.

# Data
x = np.linspace(0, 2*np.pi, 400)
x_long = np.linspace(0, 20, 500)
t = np.linspace(0, 2*np.pi, 500)
X, Y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z = np.sin(X**2 + Y**2)

fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
fig.suptitle("ECEN4293 Matplotlib Oscillations & Patterns",
             fontsize=14, fontweight='bold')

# (A) Sin & Cos
# (A) Sin & Cos  (top-left)
ax = axs[0, 0]
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set_title("(A) sin & cosine")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
# TODO: plot sin and cos with legend, grid, labels

# (B) Damped Oscillation
ax = axs[0, 1]

y_damped = np.exp(-0.1 * x_long) * np.sin(5 * x_long)
ax.plot(x_long, y_damped)

ax.set_title("(B) Damped Oscillation")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)

# (C) 2D Wave Heatmap
ax = axs[1, 0]
im = ax.imshow(Z, origin="lower", extent=[-3, 3, -3, 3], aspect="equal")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

cbar.set_label("Value")
ax.set_title("(C) 2D Wave Pattern")

ax.set_xlabel("X")
ax.set_ylabel("Y")

# (D) Lissajous Curve  (bottom-right)
ax = axs[1, 1]
x_liss = np.sin(3 * t)
y_liss = np.sin(4 * t)

ax.plot(x_liss, y_liss)
ax.set_aspect("equal", adjustable="box")
ax.set_title("(D) Lissajous Curve")
ax.set_xlabel("x(t) = sin(3t)")
ax.set_ylabel("y(t) = sin(4t)")
ax.grid(True, alpha=0.2)

# Save a hi-res image (as required) and/or show
fig.savefig("ecen4293_patterns_hi-res.png", dpi=300)
plt.show()