import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



#Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)


#Load anomaly data
def load_anomaly(path="anomaly_data.mat"):
    data = loadmat(path)
    # Expect (ny, nx, nz) grids
    x = data["x"]
    y = data["y"]
    z = data["z"]
    rho = data["rho"]
    return x, y, z, rho


#Mass properties
def compute_cell_volume(x, y, z):
    # x varies along axis=1, y varies along axis=0, z varies along axis=2
    x_vals = np.unique(x[0, :, 0])
    y_vals = np.unique(y[:, 0, 0])
    z_vals = np.unique(z[0, 0, :])

    dx = float(np.median(np.diff(x_vals)))
    dy = float(np.median(np.diff(y_vals)))
    dz = float(np.median(np.diff(z_vals)))

    return dx * dy * dz, dx, dy


def total_mass_and_barycentre(x, y, z, rho, dV):
    m_cells = rho * dV
    m_total = float(np.sum(m_cells))

    xb = float(np.sum(m_cells * x) / m_total)
    yb = float(np.sum(m_cells * y) / m_total)
    zb = float(np.sum(m_cells * z) / m_total)

    return m_total, (xb, yb, zb)


#Mean density slices 
def mean_density_slices(x, y, z, rho):
    #rho shape is (ny, nx, nz)
    #xz = mean over y (axis=0) -> (nx, nz)
    #yz = mean over x (axis=1) -> (ny, nz)
    #xy = mean over z (axis=2) -> (ny, nx)
    rho_xz = np.mean(rho, axis=0)
    rho_yz = np.mean(rho, axis=1)
    rho_xy = np.mean(rho, axis=2)

    x_vals = x[0, :, 0]      #(nx,)
    y_vals = y[:, 0, 0]      #(ny,)
    z_vals = z[0, 0, :]      #(nz,)

    return x_vals, y_vals, z_vals, rho_xz, rho_yz, rho_xy


def plot_mean_density(x_vals, y_vals, z_vals, rho_xz, rho_yz, rho_xy, bary):
    xb, yb, zb = bary
    vmin = 0.0
    vmax = float(max(rho_xz.max(), rho_yz.max(), rho_xy.max()))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    #XZ (x vs z), rho_xz is (nx, nz)
    XZ_X, XZ_Z = np.meshgrid(x_vals, z_vals, indexing="ij")
    c1 = axes[0].contourf(XZ_X, XZ_Z, rho_xz, levels=30, vmin=vmin, vmax=vmax)
    axes[0].plot([xb], [zb], "xk", markersize=8)
    axes[0].set_title("Mean density (xz)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    fig.colorbar(c1, ax=axes[0])

    #YZ (y vs z), rho_yz is (ny, nz)
    YZ_Y, YZ_Z = np.meshgrid(y_vals, z_vals, indexing="ij")
    c2 = axes[1].contourf(YZ_Y, YZ_Z, rho_yz, levels=30, vmin=vmin, vmax=vmax)
    axes[1].plot([yb], [zb], "xk", markersize=8)
    axes[1].set_title("Mean density (yz)")
    axes[1].set_xlabel("y (m)")
    axes[1].set_ylabel("z (m)")
    fig.colorbar(c2, ax=axes[1])

    #XY (x vs y), rho_xy is (ny, nx)
    XY_X, XY_Y = np.meshgrid(x_vals, y_vals, indexing="xy")  # -> (ny, nx)
    c3 = axes[2].contourf(XY_X, XY_Y, rho_xy, levels=30, vmin=vmin, vmax=vmax)
    axes[2].plot([xb], [yb], "xk", markersize=8)
    axes[2].set_title("Mean density (xy)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    fig.colorbar(c3, ax=axes[2])

    fig.savefig("examples/mean_density_slices.png", dpi=200)
    fig.savefig("examples/anomaly_mean_density.png", dpi=200)
    plt.close(fig)


#Survey modelling
def build_survey_grid(dx=5.0, extent=100.0):
    xs = np.arange(-extent, extent + dx, dx)
    ys = np.arange(-extent, extent + dx, dx)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # X,Y shape (ny, nx)
    return X, Y, xs, ys


def compute_gz(X, Y, z_survey, x, y, z, rho, dV):
    xc = x.ravel()
    yc = y.ravel()
    zc = z.ravel()
    m = (rho.ravel() * dV)

    #Build survey points
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = np.full_like(Xf, float(z_survey))

    gz = np.zeros_like(Xf, dtype=float)

    #Accumulate contribution from all masses 
    for k in range(m.size):
        dx = Xf - xc[k]
        dy = Yf - yc[k]
        dz = Zf - zc[k]
        r2 = dx*dx + dy*dy + dz*dz
        r = np.sqrt(r2)
        r3 = np.where(r > 0, r2 * r, np.inf)
        gz += G * m[k] * dz / r3  #downward component

    return gz.reshape(X.shape)


def plot_field(X, Y, field, title, filename, nlevels=50):
    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.contourf(X, Y, field, levels=nlevels)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(c, ax=ax)
    fig.savefig(filename, dpi=200)
    plt.close(fig)


#Derivatives (NO np.roll artifacts)
def d2gz_dz2_from_laplace(gz, dx, dy):
    dGdy, dGdx = np.gradient(gz, dy, dx)        
    d2Gdyy, _ = np.gradient(dGdy, dy, dx)
    _, d2Gdxx = np.gradient(dGdx, dy, dx)
    return -(d2Gdxx + d2Gdyy)


def plot_derivatives_2x2(X, Y, dgz0, d2z0, dgz100, d2z100, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    c1 = axes[0, 0].contourf(X, Y, dgz0, levels=60)
    axes[0, 0].set_title(r"$\partial g_z/\partial z$ at z = 0 m")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(c1, ax=axes[0, 0])

    c2 = axes[0, 1].contourf(X, Y, d2z0, levels=60)
    axes[0, 1].set_title(r"$\partial^2 g_z/\partial z^2$ at z = 0 m")
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("y (m)")
    axes[0, 1].set_aspect("equal", adjustable="box")
    fig.colorbar(c2, ax=axes[0, 1])

    c3 = axes[1, 0].contourf(X, Y, dgz100, levels=60)
    axes[1, 0].set_title(r"$\partial g_z/\partial z$ at z = 100 m")
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 0].set_ylabel("y (m)")
    axes[1, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(c3, ax=axes[1, 0])

    c4 = axes[1, 1].contourf(X, Y, d2z100, levels=60)
    axes[1, 1].set_title(r"$\partial^2 g_z/\partial z^2$ at z = 100 m")
    axes[1, 1].set_xlabel("x (m)")
    axes[1, 1].set_ylabel("y (m)")
    axes[1, 1].set_aspect("equal", adjustable="box")
    fig.colorbar(c4, ax=axes[1, 1])

    fig.savefig(filename, dpi=200)
    plt.close(fig)


#Main
def main():
    x, y, z, rho = load_anomaly()
    dV, dx, dy = compute_cell_volume(x, y, z)

    m_total, bary = total_mass_and_barycentre(x, y, z, rho, dV)
    rho_max = float(np.max(rho))
    rho_mean = float(np.mean(rho))

    with open("section7_output.txt", "w") as f:
        f.write(f"Total mass (kg): {m_total}\n")
        f.write(f"Barycentre (x,y,z) [m]: {bary}\n")
        f.write(f"Maximum density (kg/m^3): {rho_max}\n")
        f.write(f"Mean density (kg/m^3): {rho_mean}\n")

    #Mean density figure
    x_vals, y_vals, z_vals, rho_xz, rho_yz, rho_xy = mean_density_slices(x, y, z, rho)
    plot_mean_density(x_vals, y_vals, z_vals, rho_xz, rho_yz, rho_xy, bary)

    #Survey maps
    X, Y, xs, ys = build_survey_grid(dx=5.0, extent=100.0)

    z_levels = [0.0, 1.0, 100.0, 110.0]
    gz_maps = {}

    for zz in z_levels:
        gz_maps[zz] = compute_gz(X, Y, zz, x, y, z, rho, dV)
        plot_field(X, Y, gz_maps[zz], f"gz (down) at z={int(zz)} m", f"examples/gz_z{int(zz)}.png")

    #2x2 gz grid 
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, zz in zip(axes.ravel(), z_levels):
        c = ax.contourf(X, Y, gz_maps[zz], levels=60)
        ax.set_title(f"gz at z={int(zz)} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(c, ax=ax)
    fig.savefig("examples/gz_2x2_grid.png", dpi=200)
    plt.close(fig)

    #First derivative ∂gz/∂z
    dgz_dz_z0 = (gz_maps[1.0] - gz_maps[0.0]) / 1.0
    dgz_dz_z100 = (gz_maps[110.0] - gz_maps[100.0]) / 10.0

    plot_field(X, Y, dgz_dz_z0, "dgz/dz at z=0 m", "examples/dgz_dz_z0.png")
    plot_field(X, Y, dgz_dz_z100, "dgz/dz at z=100 m", "examples/dgz_dz_z100.png")

    #Second derivative
    d2gz_z0 = d2gz_dz2_from_laplace(gz_maps[0.0], dx=5.0, dy=5.0)
    d2gz_z100 = d2gz_dz2_from_laplace(gz_maps[100.0], dx=5.0, dy=5.0)

    plot_field(X, Y, d2gz_z0, "d2gz/dz2 at z=0 m (Laplace)", "examples/d2gz_dz2_z0.png")
    plot_field(X, Y, d2gz_z100, "d2gz/dz2 at z=100 m (Laplace)", "examples/d2gz_dz2_z100.png")

    plot_derivatives_2x2(
        X, Y, dgz_dz_z0, d2gz_z0, dgz_dz_z100, d2gz_z100,
        "examples/anomaly_survey_derivatives.png"
    )


if __name__ == "__main__":
    main()
