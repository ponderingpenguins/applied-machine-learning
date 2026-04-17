import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE = "../../Gait-Datasets-TIFS20/Dataset #1/train/Inertial Signals"


def load(name):
    return pd.read_csv(f"{BASE}/{name}.txt", sep=r"\s+", header=None).to_numpy()


ax_ = load("train_acc_x")
ay_ = load("train_acc_y")
az_ = load("train_acc_z")
gx_ = load("train_gyr_x")
gy_ = load("train_gyr_y")
gz_ = load("train_gyr_z")

sample_index = 1
fs = 50.0  # sampling rate
dt = 1.0 / fs
gyro_in_degrees = True  # set False if gyro is already in rad/s
subtract_gravity = False  # set False if accel is already body (gravity-removed)

acc = np.stack(
    [ax_[sample_index], ay_[sample_index], az_[sample_index]], axis=1
)  # (N, 3)
gyr = np.stack([gx_[sample_index], gy_[sample_index], gz_[sample_index]], axis=1)
if gyro_in_degrees:
    gyr = np.deg2rad(gyr)

N = acc.shape[0]


# --- 1. Integrate gyro → orientation as quaternions ---
def quat_mul(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_rotate(q, v):
    qv = np.array([0.0, *v])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_mul(quat_mul(q, qv), qc)[1:]


q = np.array([1.0, 0.0, 0.0, 0.0])
orientations = np.zeros((N, 4))
orientations[0] = q
for i in range(1, N):
    w = gyr[i]
    theta = np.linalg.norm(w) * dt
    if theta > 1e-12:
        axis = w / np.linalg.norm(w)
        dq = np.array([np.cos(theta / 2), *(axis * np.sin(theta / 2))])
    else:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    q = quat_mul(q, dq)
    q /= np.linalg.norm(q)
    orientations[i] = q

# --- 2. Rotate body accel → world frame ---
acc_world = np.array([quat_rotate(orientations[i], acc[i]) for i in range(N)])

# --- 3. Remove gravity (estimated as the mean of world-frame accel) ---
if subtract_gravity:
    acc_world = acc_world - acc_world.mean(axis=0)

# --- 4. Integrate → velocity → position (trapezoidal) ---
vel = np.zeros_like(acc_world)
pos = np.zeros_like(acc_world)
for i in range(1, N):
    vel[i] = vel[i - 1] + 0.5 * (acc_world[i] + acc_world[i - 1]) * dt
    pos[i] = pos[i - 1] + 0.5 * (vel[i] + vel[i - 1]) * dt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=1.5)
ax.scatter(*pos[0], c="green", s=50, label="start")
ax.scatter(*pos[-1], c="red", s=50, label="end")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"Reconstructed 3D trajectory — sample {sample_index}")
ax.legend()
plt.tight_layout()
plt.savefig(f"../figures/trajectory_3d_sample_{sample_index}.png", dpi=120)
plt.show()
