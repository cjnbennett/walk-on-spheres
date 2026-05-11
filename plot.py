# Plot the u(x,y) solution obtained from WoS.
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("input", nargs="?", default="poisson_wos.h5")
p.add_argument("-o", "--output", default="u.png")
p.add_argument("--cmap", default="RdBu_r")
args = p.parse_args()

# read hdf5
with h5py.File(args.input, "r") as f:
    x: np.ndarray = f["x"][:]  # type: ignore[index]
    y: np.ndarray = f["y"][:]  # type: ignore[index]
    u: np.ndarray = f["u"][:]  # type: ignore[index]

# colour scale of plot
finite = u[np.isfinite(u)]  # type: ignore[index]
if finite.min() < 0 < finite.max():
    vmax = float(np.max(np.abs(finite)))
    vmin = -vmax
else:
    vmin = float(finite.min())
    vmax = float(finite.max())

# plot
fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
mesh = ax.pcolormesh(x, y, u, shading="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
fig.colorbar(mesh, ax=ax, label=r"$u$")
# domain boundary
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1)
# labels
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"$u(x,y)$")
# save
fig.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
