import numpy as np
import matplotlib.pyplot as plt

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


#Make an (x, y) grid
def make_xy_grid(dx, x_min=-100, x_max=100, y_min=-100, y_max=100):
    """
    A meshgrid of X and Y coordinates.
    dx controls the spacing between grid points.
    """
    xs = np.arange(x_min, x_max + dx, dx)
    ys = np.arange(y_min, y_max + dx, dx)
    X, Y = np.meshgrid(xs, ys)  #X and Y are 2D arrays
    return X, Y


#Compute U and gz on grid for one z-level
def compute_U_and_gz_on_grid(X, Y, z, xm, m):
    """
    For every (x, y) point on the grid, compute:
      - U(x,y,z)  gravity potential
      - gz(x,y,z) vertical gravity effect (positive downward)
    """
    #empty arrays to store results
    U = np.zeros_like(X, dtype=float)
    gz = np.zeros_like(X, dtype=float)

    #Loop over every grid cell (row i, column j)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            #Build the 3D observation point [x, y, z]
            x = np.array([X[i, j], Y[i, j], z], dtype=float)

            #Compute potential and vertical effect at that point
            U[i, j] = gravity_potential_point(x, xm, m)
            gz[i, j] = gravity_effect_point(x, xm, m)

    return U, gz


#3 rows (z=0,10,100) and 2 columns (U, gz) plOT
def plot_for_dx(dx, xm, m):
    """
    a 3x2 figure:
      Rows = z levels: 0, 10, 100
      Col 1 = U
      Col 2 = gz
    Use SAME COLor scale for all U PLOts and all gz plots.
    """
    zs = [0, 10, 100]  
    X, Y = make_xy_grid(dx)

    #Compute U and gz for each z and store them
    U_maps = []
    gz_maps = []
    for z in zs:
        U, gz = compute_U_and_gz_on_grid(X, Y, z, xm, m)
        U_maps.append(U)
        gz_maps.append(gz)

    #Find global min/max so colorbars are consistent
    Umin = min(U.min() for U in U_maps)
    Umax = max(U.max() for U in U_maps)
    gzmin = min(g.min() for g in gz_maps)
    gzmax = max(g.max() for g in gz_maps)

    #a 3x2 subplot layout
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
    fig.suptitle(f"Point mass gravity (dx = {dx} m)", fontsize=14)

    #Loop through each row (each z)
    for row, z in enumerate(zs):

        #Left column: U
        axU = axes[row, 0]
        cU = axU.contourf(X, Y, U_maps[row], levels=30,
                          vmin=Umin, vmax=Umax, cmap="viridis")
        axU.plot(X, Y, "xk", markersize=2)  # show grid points
        axU.set_title(f"U at z={z} m")
        axU.set_xlabel("x (m)")
        axU.set_ylabel("y (m)")
        plt.colorbar(cU, ax=axU)

        #Right column: gz
        axg = axes[row, 1]
        cg = axg.contourf(X, Y, gz_maps[row], levels=30,
                          vmin=gzmin, vmax=gzmax, cmap="viridis")
        axg.plot(X, Y, "xk", markersize=2)
        axg.set_title(f"gz (down) at z={z} m")
        axg.set_xlabel("x (m)")
        axg.set_ylabel("y (m)")
        plt.colorbar(cg, ax=axg)

    plt.tight_layout()
    plt.show()


#Set anomaly ands run dx cases

def main():
    m = 1.0e7
    xm = np.array([0.0, 0.0, -10.0], dtype=float)

    plot_for_dx(5.0, xm, m)
    plot_for_dx(25.0, xm, m)


if __name__ == "__main__":
    main()
