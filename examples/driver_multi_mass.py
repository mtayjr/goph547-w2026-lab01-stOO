import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point



#The grids with the forward model
def make_xy_grid(dx, x_min=-100, x_max=100, y_min=-100, y_max=100):
    xs = np.arange(x_min, x_max + dx, dx)
    ys = np.arange(y_min, y_max + dx, dx)
    return np.meshgrid(xs, ys)


def total_U_gz_at_point(x_obs, masses, locations):
    """
    Sum contributions from multiple masses point which s the linear superpoositiojn.
    """
    U_total = 0.0
    gz_total = 0.0
    for mi, xmi in zip(masses, locations):
        U_total += gravity_potential_point(x_obs, xmi, mi)
        gz_total += gravity_effect_point(x_obs, xmi, mi)
    return U_total, gz_total


def compute_fields_on_grid(X, Y, z, masses, locations):
    U = np.zeros_like(X, dtype=float)
    gz = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_obs = np.array([X[i, j], Y[i, j], z], dtype=float)
            U[i, j], gz[i, j] = total_U_gz_at_point(x_obs, masses, locations)

    return U, gz



#Part B: Generate 5-mass set
def generate_mass_set(
    total_mass=1e7,
    com_target=np.array([0.0, 0.0, -10.0]),
    sigma_m=None,
    sigma_xy=20.0,
    sigma_z=2.0,
    mu_xy=0.0,
    mu_z=-10.0,
    min_depth=-1.0,
    max_tries=5000,
    seed=None
):
    """
    Generate 5 masses (m_i) at positions (x_i,y_i,z_i) by:
      sum m_i = total_mass
      center of mass = com_target
      and all z_i <= min_depth   below ground for at least 1 m

    Randomly create 4 masses and 4 locations, then calculate for for the 5th.
    """
    rng = np.random.default_rng(seed)

    if sigma_m is None:
        sigma_m = total_mass / 100.0

    mu_m = total_mass / 5.0

    for _ in range(max_tries):
        #Draw 4 masses
        m4 = rng.normal(loc=mu_m, scale=sigma_m, size=4)
        if np.any(m4 <= 0):
            continue

        #Draw 4 locations
        x4 = rng.normal(loc=mu_xy, scale=sigma_xy, size=4)
        y4 = rng.normal(loc=mu_xy, scale=sigma_xy, size=4)
        z4 = rng.normal(loc=mu_z, scale=sigma_z, size=4)

        #Connect below-ground requirement for the first 4
        if np.any(z4 > min_depth):
            continue

        #Calculate the 5th mass so total mass matches
        m5 = total_mass - np.sum(m4)
        if m5 <= 0:
            continue

        #Calculate for 5th location from COM constraints:
        # com = (sum m_i r_i) / (sum m_i) = target
        # => m5 * r5 = total_mass*target - sum_{i=1..4} m_i r_i
        sum_mr_x = np.sum(m4 * x4)
        sum_mr_y = np.sum(m4 * y4)
        sum_mr_z = np.sum(m4 * z4)

        x5 = (total_mass * com_target[0] - sum_mr_x) / m5
        y5 = (total_mass * com_target[1] - sum_mr_y) / m5
        z5 = (total_mass * com_target[2] - sum_mr_z) / m5

        #Clarify the  depth condition for 5th mass has been assigned
        if z5 > min_depth:
            continue

        #Build up final arrays
        masses = np.concatenate([m4, [m5]])
        locations = np.column_stack([np.concatenate([x4, [x5]]),
                                     np.concatenate([y4, [y5]]),
                                     np.concatenate([z4, [z5]])])

        #Last
        m_check = masses.sum()
        com_check = (masses[:, None] * locations).sum(axis=0) / m_check

        if not np.isclose(m_check, total_mass):
            continue
        if not np.allclose(com_check, com_target, rtol=0, atol=1e-6):
            continue

        return masses, locations

    raise RuntimeError("Failed to generate a valid mass set. Increase max_tries or relax constraints.")



#Plotting Block
def plot_U_gz_for_set(masses, locations, dx, set_id, out_dir="examples"):
    zs = [0, 10, 100]
    X, Y = make_xy_grid(dx)

    U_maps = []
    gz_maps = []

    for z in zs:
        U, gz = compute_fields_on_grid(X, Y, z, masses, locations)
        U_maps.append(U)
        gz_maps.append(gz)

    Umin = min(U.min() for U in U_maps)
    Umax = max(U.max() for U in U_maps)
    gmin = min(g.min() for g in gz_maps)
    gmax = max(g.max() for g in gz_maps)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"Multi-mass set {set_id} (dx = {dx} m)", fontsize=18)

    for row, z in enumerate(zs):
        axU = axes[row, 0]
        cU = axU.contourf(X, Y, U_maps[row], levels=30, vmin=Umin, vmax=Umax, cmap="viridis")
        axU.plot(X, Y, "xk", markersize=2)
        axU.set_title(f"U at z={z} m")
        axU.set_xlabel("x (m)")
        axU.set_ylabel("y (m)")
        plt.colorbar(cU, ax=axU)

        axg = axes[row, 1]
        cg = axg.contourf(X, Y, gz_maps[row], levels=30, vmin=gmin, vmax=gmax, cmap="viridis")
        axg.plot(X, Y, "xk", markersize=2)
        axg.set_title(f"gz (down) at z={z} m")
        axg.set_xlabel("x (m)")
        axg.set_ylabel("y (m)")
        plt.colorbar(cg, ax=axg)

    plt.tight_layout()

    #Save figure
    out_path = f"{out_dir}/multi_mass_set{set_id}_dx{int(dx)}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_path}")



#Main
def main():
    total_mass = 1e7
    com_target = np.array([0.0, 0.0, -10.0])

    #Generate 3 sets
    for set_id in [1, 2, 3]:
        masses, locations = generate_mass_set(
            total_mass=total_mass,
            com_target=com_target,
            sigma_xy=20.0,
            sigma_z=2.0,
            mu_xy=0.0,
            mu_z=-10.0,
            min_depth=-1.0,
            seed=set_id  # reproducible
        )

        #Check
        com_check = (masses[:, None] * locations).sum(axis=0) / masses.sum()
        print(f"\nSet {set_id}:")
        print("  Total mass:", masses.sum())
        print("  COM:", com_check)
        print("  min z:", locations[:, 2].min(), "max z:", locations[:, 2].max())

        #Save to .mat
        mat_name = f"mass_set_{set_id}.mat"
        savemat(mat_name, {"masses": masses, "locations": locations})
        print(f"Saved .mat file: {mat_name}")

        #Make plots for dx = 5 and 25
        plot_U_gz_for_set(masses, locations, dx=5.0, set_id=set_id)
        plot_U_gz_for_set(masses, locations, dx=25.0, set_id=set_id)


if __name__ == "__main__":
    main()
