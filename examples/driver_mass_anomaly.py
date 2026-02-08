import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def main():
    #Load the distributed density anomaly data
    data = loadmat("anomaly_data.mat")

    x = data["x"]
    y = data["y"]
    z = data["z"]
    rho = data["rho"]

    #Grid spacing
    dx = dy = dz = 2.0
    cell_volume = dx * dy * dz

    #Total mass and barycentre
    total_mass = np.sum(rho) * cell_volume

    x_bar = np.sum(x * rho) * cell_volume / total_mass
    y_bar = np.sum(y * rho) * cell_volume / total_mass
    z_bar = np.sum(z * rho) * cell_volume / total_mass

    print("Total mass (kg):", total_mass)
    print("Barycentre (x, y, z) [m]:", x_bar, y_bar, z_bar)
    print("Maximum cell density (kg/m^3):", np.max(rho))
    print("Mean density (kg/m^3):", np.mean(rho))

    #Mean density cross-sections
    rho_xz = np.mean(rho, axis=1)
    rho_yz = np.mean(rho, axis=0)
    rho_xy = np.mean(rho, axis=2)

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    im0 = axes[0].contourf(rho_xz, cmap="viridis")
    axes[0].plot(x_bar, z_bar, "xk", markersize=6)
    axes[0].set_title("Mean density (xz)")

    im1 = axes[1].contourf(rho_yz, cmap="viridis")
    axes[1].plot(y_bar, z_bar, "xk", markersize=6)
    axes[1].set_title("Mean density (yz)")

    im2 = axes[2].contourf(rho_xy, cmap="viridis")
    axes[2].plot(x_bar, y_bar, "xk", markersize=6)
    axes[2].set_title("Mean density (xy)")

    for ax in axes:
        plt.colorbar(im0, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
