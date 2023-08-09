import numpy as np
import matplotlib.pyplot as plt

STATE_UPPER = np.array([2, 1, 2, 1, 2, 1, 0.2, 0.2, 0.2, 1, 1, 1])
STATE_LOWER = np.array([-2, -1, -2, -1, 0, -1, -0.2, -0.2, -0.2, -1, -1, -1])

def load_unconstrained_trajectories():
    return np.load('examples/rl/models/cbf_ppo/safety_coef_0/obs_hist.npy')

def load_constrained_trajectories():
    return np.load('examples/rl/models/cbf_ppo/safety_coef_10/obs_hist.npy')

def load_pos_ref():
    return np.load('examples/rl/pos_ref.npy')

def plot_trajectory(ax, x, y, z, label: str):


    ax.plot(x, y, z, label=label, color='b', linewidth=2)
    ax.scatter(x[0], y[0], z[0], color='g', s=100, label='Start')  # Start point
    ax.scatter(x[-1], y[-1], z[-1], color='r', s=100, label='End')  # End point

if __name__ == "__main__":
    
    unconstrained = load_unconstrained_trajectories()
    constrained = load_constrained_trajectories()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Trajectory Visualization')
    ax.legend()

    for i in range(3):
        x, y, z = unconstrained[i, :, 0], unconstrained[i, :, 2], unconstrained[i, :, 4]
        plot_trajectory(ax, x, y, z, f'Unconstrained {i}')

    for i in range(3):
        x, y, z = constrained[i, :, 0], constrained[i, :, 2], constrained[i, :, 4]
        plot_trajectory(ax, x, y, z, f'Constrained {i + 3}')
    pos_ref = load_pos_ref()
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], label='Reference', color='r', linewidth=2)

    plt.show()
