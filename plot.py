# Plot the u(x,y) solution obtained from WoS.
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("input", nargs="?", default="poisson_wos.h5")
p.add_argument("-o", "--output", default="u.png")
p.add_argument("--mesh", default="meshes/unit_circle.obj", help=".obj file for the domain boundary overlay")
p.add_argument("--cmap", default="RdBu_r")
args = p.parse_args()

# read hdf5
with h5py.File(args.input, "r") as f:
    x: np.ndarray = f["x"][:]  # type: ignore[index]
    y: np.ndarray = f["y"][:]  # type: ignore[index]
    u: np.ndarray = f["u"][:]  # type: ignore[index]

# read 2D .obj boundary: "v x y z" vertices, "l i j k ..." polylines (1-indexed)
verts: list[tuple[float, float]] = []
polylines: list[list[int]] = []
with open(args.mesh) as mf:
    for line in mf:
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            verts.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "l":
            polylines.append([int(i) - 1 for i in parts[1:]])
V = np.array(verts)

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
# domain boundary (from .obj)
for poly in polylines:
    ax.plot(V[poly, 0], V[poly, 1], "k-", lw=1)
# clip plot bounds to the mesh's bounding box (with 10% padding)
pad = 0.10 * max(np.ptp(V[:, 0]), np.ptp(V[:, 1]))
ax.set_xlim(V[:, 0].min() - pad, V[:, 0].max() + pad)
ax.set_ylim(V[:, 1].min() - pad, V[:, 1].max() + pad)
# labels
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"$u(x,y)$")
# save
fig.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
