import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
def load_images_txt(images_txt_path):
    camera_centers = []
    camera_dirs = []
    with open(images_txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            data = line.split()
            qw, qx, qy, qz = map(float, data[1:5])
            t = np.array(list(map(float, data[5:8])))
            R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
            C = -R_wc.T @ t
            camera_centers.append(C)
            view_dir = R_wc.T @ np.array([0, 0, 1])
            camera_dirs.append(view_dir)
    return np.array(camera_centers), np.array(camera_dirs)
def visualize_poses(camera_centers, camera_dirs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        c="red",
        marker="^",
        s=100,
        label="Camera Centers",
    )
    ax.quiver(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        camera_dirs[:, 0],
        camera_dirs[:, 1],
        camera_dirs[:, 2],
        length=0.1,
        color="blue",
        label="Viewing Direction",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Pose Visualization (COLMAP)")

    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()
def main(input_dir):
    images_txt = os.path.join(input_dir, "images.txt")

    if not os.path.exists(images_txt):
        raise FileNotFoundError(f"images.txt not found in {input_dir}")

    camera_centers, camera_dirs = load_images_txt(images_txt)

    if len(camera_centers) == 0:
        print("No camera poses found.")
        return

    visualize_poses(camera_centers, camera_dirs)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP camera poses")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing images.txt (COLMAP TXT model)",
    )
    args = parser.parse_args()

    main(args.input_dir)
