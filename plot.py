# Plot the u(x,y[,z]) solution obtained from WoS.
# 2D output -> single pcolormesh with the polyline boundary overlay.
# 3D output -> XZ slice + YZ slice + 3D rendering with both slice planes overlaid.
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("input", nargs="?", default="poisson_wos.h5")
p.add_argument("-o", "--output", default="u.png")
p.add_argument("--mesh", default="meshes/annulus.obj", help=".obj file for the mesh overlay")
p.add_argument("--cmap", default="RdBu_r")
args = p.parse_args()

# read hdf5: /u has shape (Nx, Ny, Nz) with Nz=1 for 2D meshes
with h5py.File(args.input, "r") as f:
    x: np.ndarray = f["x"][:]  # type: ignore[index]
    y: np.ndarray = f["y"][:]  # type: ignore[index]
    z: np.ndarray = f["z"][:]  # type: ignore[index]
    u: np.ndarray = f["u"][:]  # type: ignore[index]
is_3D = u.shape[2] > 1

# read .obj boundary
verts_raw: list[list[float]] = []
segs_raw: list[list[int]] = []
tris_raw: list[list[int]] = []
with open(args.mesh) as mf:
    for line in mf:
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            verts_raw.append([float(parts[1]), float(parts[2]), float(parts[3]) if len(parts) > 3 else 0.0])
        elif parts[0] == "l":
            segs_raw.append([int(i) - 1 for i in parts[1:]])
        elif parts[0] == "f":
            tris_raw.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])

verts = np.array(verts_raw)
segs = np.array(segs_raw)
tris = np.array(tris_raw)

# diverging colour scale around 0 if data spans both signs
finite = u[np.isfinite(u)]  # type: ignore[index]
if finite.min() < 0 < finite.max():
    vmax = float(np.max(np.abs(finite)))
    vmin = -vmax
else:
    vmin = float(finite.min())
    vmax = float(finite.max())

if not is_3D:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, u[:, :, 0].T, shading="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=r"$u$")
    for seg in segs:
        ax.plot(verts[seg, 0], verts[seg, 1], "k-", lw=1)
    pad = 0.10 * max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1]))  # 10% padding around mesh bbox
    ax.set_xlim(verts[:, 0].min() - pad, verts[:, 0].max() + pad)
    ax.set_ylim(verts[:, 1].min() - pad, verts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$u(x,y)$")
else:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    y_mid = len(y) // 2
    x_mid = len(x) // 2
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(args.cmap)

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)

    # XZ slice at middle y
    ax_xz = fig.add_subplot(1, 3, 1)
    mesh_xz = ax_xz.pcolormesh(x, z, u[:, y_mid, :].T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title(f"XZ slice at y = {y[y_mid]:.2f}")

    # YZ slice at middle x
    ax_yz = fig.add_subplot(1, 3, 2)
    mesh_yz = ax_yz.pcolormesh(y, z, u[x_mid, :, :].T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_title(f"YZ slice at x = {x[x_mid]:.2f}")

    fig.colorbar(mesh_xz, ax=[ax_xz, ax_yz], label=r"$u$", shrink=0.9)

    # 3D mesh + slice planes overlaid
    ax_3D = fig.add_subplot(1, 3, 3, projection="3d")
    ax_3D.add_collection3d(Poly3DCollection([verts[t] for t in tris], alpha=0.10, edgecolor="k", facecolor="lightgray", linewidth=0.3))

    # XZ slice plane at y_mid, coloured by u[:, y_mid, :]
    XZ_x, XZ_z = np.meshgrid(x, z)
    ax_3D.plot_surface(XZ_x, np.full_like(XZ_x, y[y_mid]), XZ_z, facecolors=cmap(norm(u[:, y_mid, :].T)), shade=False, antialiased=False, rcount=len(z), ccount=len(x))

    # YZ slice plane at x_mid, coloured by u[x_mid, :, :]
    YZ_y, YZ_z = np.meshgrid(y, z)
    ax_3D.plot_surface(np.full_like(YZ_y, x[x_mid]), YZ_y, YZ_z, facecolors=cmap(norm(u[x_mid, :, :].T)), shade=False, antialiased=False, rcount=len(z), ccount=len(y))

    pad3D = 0.10 * np.ptp(verts, axis=0).max()  # 10% padding around bbox
    ax_3D.set_xlim(verts[:, 0].min() - pad3D, verts[:, 0].max() + pad3D)
    ax_3D.set_ylim(verts[:, 1].min() - pad3D, verts[:, 1].max() + pad3D)
    ax_3D.set_zlim(verts[:, 2].min() - pad3D, verts[:, 2].max() + pad3D)
    ax_3D.set_xlabel("x")
    ax_3D.set_ylabel("y")
    ax_3D.set_zlabel("z")
    ax_3D.set_title("Boundary mesh + slice planes")

fig.savefig(args.output, dpi=150)
print(f"Saved to {args.output}")
