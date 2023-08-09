import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

STATE_UPPER = np.array([2, 1, 2, 1, 2, 1, 0.2, 0.2, 0.2, 1, 1, 1])
STATE_LOWER = np.array([-2, -1, -2, -1, 0, -1, -0.2, -0.2, -0.2, -1, -1, -1])

def parse_args():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=str)
    return parser.parse_args()

def load_trajectories(save_dir):
    return np.load(f'{save_dir}/obs_hist.npy')

def load_pos_ref():
    return np.load('examples/rl/pos_ref.npy')

def plot_trajectory(ax, x, y, z, label: str):
    ax.plot(x, y, z, label=label, color='b', linewidth=2)
    ax.scatter(x[0], y[0], z[0], color='g', s=100, label='Start')  # Start point
    ax.scatter(x[-1], y[-1], z[-1], color='r', s=100, label='End')  # End point

def draw_box(ax, x_limit, y_limit, z_limit):
    # Draw each face. The vertices are taken in a clockwise order
    # Bottom
    x_box, y_box, z_box = x_limit, y_limit, z_limit
    vertices = [[x_box[0], y_box[0], z_box[0]], [x_box[0], y_box[1], z_box[0]], 
                [x_box[1], y_box[1], z_box[0]], [x_box[1], y_box[0], z_box[0]]]
    ax.add_collection3d(art3d.Poly3DCollection([vertices], color='cyan', alpha=0.5))

    # Top
    vertices = [[x_box[0], y_box[0], z_box[1]], [x_box[0], y_box[1], z_box[1]], 
                [x_box[1], y_box[1], z_box[1]], [x_box[1], y_box[0], z_box[1]]]
    ax.add_collection3d(art3d.Poly3DCollection([vertices], color='cyan', alpha=0.5))

    # Front
    vertices = [[x_box[0], y_box[0], z_box[0]], [x_box[1], y_box[0], z_box[0]], 
                [x_box[1], y_box[0], z_box[1]], [x_box[0], y_box[0], z_box[1]]]
    ax.add_collection3d(art3d.Poly3DCollection([vertices], color='cyan', alpha=0.5))

    # Back
    vertices = [[x_box[0], y_box[1], z_box[0]], [x_box[1], y_box[1], z_box[0]], 
                [x_box[1], y_box[1], z_box[1]], [x_box[0], y_box[1], z_box[1]]]
    ax.add_collection3d(art3d.Poly3DCollection([vertices], color='cyan', alpha=0.5))

    # Left
    vertices = [[x_box[0], y_box[0], z_box[0]], [x_box[0], y_box[1], z_box[0]], 
                [x_box[0], y_box[1], z_box[1]], [x_box[0], y_box[0], z_box[1]]]
    ax.add_collection3d(art3d.Poly3DCollection([vertices], color='cyan', alpha=0.5))

    # Right
    vertices = [[x_box[1], y_box[0], z_box[0]], [x_box[1], y_box[1], z_box[0]], 
                [x_box[1], y_box[1], z_box[1]], [x_box[1], y_box[0], z_box[1]]]
    ax.add_collection3d(art3d.Poly3DCollection([vertices], color='cyan', alpha=0.5))

if __name__ == "__main__":
    
    args = parse_args()
    data = load_trajectories(args.restore)

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

    # Define the vertices that compose the box
    x_box = [-2, -0.8]
    y_box = [-2, 2]
    z_box = [0, 2]
     
    draw_box(ax, x_box, y_box, z_box)

    for i in range(10):
        x, y, z = data[i, :, 0], data[i, :, 2], data[i, :, 4]
        plot_trajectory(ax, x, y, z, f'Trajectory {i}')

    pos_ref = load_pos_ref()
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], label='Reference', color='r', linewidth=2)

    plt.show()
