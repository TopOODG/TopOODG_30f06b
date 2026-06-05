import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import torch
torch.set_num_threads(1)


def extract_bifurcation_data(
    trajectories,
    control_params,
    transient_steps,
    relevant_dim=0,
    tail_steps=100,
    fixed_point_atol=1e-2,
):
    def pad_ragged(values_per_cp):
        max_len = max((values.shape[0] for values in values_per_cp), default=0)
        padded = np.full((len(values_per_cp), max_len), np.nan, dtype=np.float64)
        for idx, values in enumerate(values_per_cp):
            if values.size:
                padded[idx, : values.shape[0]] = values
        return padded

    minima_per_cp = []
    maxima_per_cp = []
    extrema_per_cp = []
    fixed_points_per_cp = []

    for cp_idx in range(trajectories.shape[0]):
        minima = []
        maxima = []
        fixed_points = []
        for ic_idx in range(trajectories.shape[1]):
            signal = trajectories[cp_idx, ic_idx, transient_steps:, relevant_dim]
            if signal.shape[0] < 3:
                continue

            center = signal[1:-1]
            maxima_mask = (center > signal[:-2]) & (center > signal[2:])
            minima_mask = (center < signal[:-2]) & (center < signal[2:])

            minima.append(center[minima_mask])
            maxima.append(center[maxima_mask])

            if not np.any(maxima_mask) and not np.any(minima_mask):
                tail = signal[-tail_steps:]
                if tail.size and np.all(np.isclose(tail, tail[0], atol=fixed_point_atol)):
                    fixed_points.append(np.array([tail[0]], dtype=np.float64))

        minima_arr = (
            np.concatenate(minima).astype(np.float64, copy=False)
            if minima
            else np.empty(0, dtype=np.float64)
        )
        maxima_arr = (
            np.concatenate(maxima).astype(np.float64, copy=False)
            if maxima
            else np.empty(0, dtype=np.float64)
        )
        fixed_points_arr = (
            np.concatenate(fixed_points).astype(np.float64, copy=False)
            if fixed_points
            else np.empty(0, dtype=np.float64)
        )

        minima_per_cp.append(minima_arr)
        maxima_per_cp.append(maxima_arr)
        extrema_per_cp.append(np.concatenate((minima_arr, maxima_arr)))
        fixed_points_per_cp.append(fixed_points_arr)

    return {
        "control_params": control_params,
        "minima": pad_ragged(minima_per_cp),
        "minima_counts": np.array([values.shape[0] for values in minima_per_cp], dtype=np.int64),
        "maxima": pad_ragged(maxima_per_cp),
        "maxima_counts": np.array([values.shape[0] for values in maxima_per_cp], dtype=np.int64),
        "extrema": pad_ragged(extrema_per_cp),
        "extrema_counts": np.array([values.shape[0] for values in extrema_per_cp], dtype=np.int64),
        "fixed_points": pad_ragged(fixed_points_per_cp),
        "fixed_points_counts": np.array([values.shape[0] for values in fixed_points_per_cp], dtype=np.int64),
    }


def lorenz63_system(state, t, rho):
    """
    Lorenz63 system:
    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = xy - beta z
    """
    sigma = 10.0
    beta = 8.0 / 3.0
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


num_subjects = 16
num_ICs = 8
delta_t = 0.01
T_train = 10
num_train_steps = int(T_train / delta_t)
num_test_steps = 20_000
bif_relevant_dim = 0
bif_transient_steps = num_test_steps // 5
bif_tail_steps = 100
bif_fixed_point_atol = 1e-2

rho_train_min = 5.0
rho_train_max = 22.5
rho_extrap_max = 35.0

cps_id = np.linspace(rho_train_min, rho_train_max, num_subjects)
# Keep extrapolation strictly outside the training rho domain.
cps_ood = np.linspace(rho_train_max, rho_extrap_max, num_subjects + 1)[1:]
cps = np.concatenate([cps_id, cps_ood])

ICs_train = np.random.uniform(low=-15.0, high=15.0, size=(num_subjects, num_ICs, 3))
ICs_test_id = np.random.uniform(low=-15.0, high=15.0, size=(num_subjects, num_ICs, 3))
ICs_test_ood = np.random.uniform(low=-15.0, high=15.0, size=(num_subjects, num_ICs, 3))
t_train = np.arange(num_train_steps) * delta_t
t_test = np.arange(num_test_steps) * delta_t

print(f"in-domain control params:     {cps_id}")
print(f"out-of-domain control params: {cps_ood}")


# integrating training data
X_train = np.zeros((cps_id.shape[0], num_ICs, t_train.shape[0], 3))  # (num_subjects, num_ts_per_subject, T_train, N)
for i, cp in enumerate(cps_id):
    for j in range(num_ICs):
        IC = ICs_train[i, j, :]
        X_train[i, j, :, :] = odeint(lorenz63_system, IC, t_train, args=(cp,))
print(f"X_train shape: {X_train.shape}")

# integrating in-domain test data
X_id = np.zeros((cps_id.shape[0], num_ICs, t_test.shape[0], 3))  # (num_subjects, num_ts_per_subject, T_test, N)
for i, cp in enumerate(cps_id):
    for j in range(num_ICs):
        IC = ICs_test_id[i, j, :]
        X_id[i, j, :, :] = odeint(lorenz63_system, IC, t_test, args=(cp,))
print(f"X_id shape: {X_id.shape}")

# integrating out-of-domain test data
X_ood = np.zeros((cps_ood.shape[0], num_ICs, t_test.shape[0], 3))  # (num_subjects, num_ts_per_subject, T_test, N)
for i, cp in enumerate(cps_ood):
    for j in range(num_ICs):
        IC = ICs_test_ood[i, j, :]
        X_ood[i, j, :, :] = odeint(lorenz63_system, IC, t_test, args=(cp,))
print(f"X_ood shape: {X_ood.shape}")


# standardizing training data
mean = X_train.mean(axis=(0, 1, 2))
std = X_train.std(axis=(0, 1, 2))
X_train_standardized = (X_train - mean) / std
standardization = np.concatenate([mean, std])
np.save('standardization.npy', standardization)

# applying standardization to out-of-domain test data
X_ood_standardized = (X_ood - mean) / std
# applying standardization to in-domain test data
X_id_standardized = (X_id - mean) / std


bifurcation_id = extract_bifurcation_data(
    X_id_standardized,
    cps_id,
    transient_steps=bif_transient_steps,
    relevant_dim=bif_relevant_dim,
    tail_steps=bif_tail_steps,
    fixed_point_atol=bif_fixed_point_atol,
)
bifurcation_ood = extract_bifurcation_data(
    X_ood_standardized,
    cps_ood,
    transient_steps=bif_transient_steps,
    relevant_dim=bif_relevant_dim,
    tail_steps=bif_tail_steps,
    fixed_point_atol=bif_fixed_point_atol,
)


# saving data
np.savez(
    'lorenz63_dataset_standardized.npz',
    X_train_standardized=X_train_standardized,
    X_test_id_standardized=X_id_standardized,
    X_test_ood_standardized=X_ood_standardized,
    cps_id=cps_id,
    cps_ood=cps_ood,
    standardization=standardization
)
np.savez(
    'bifurcation_data.npz',
    id_control_params=bifurcation_id['control_params'],
    id_minima=bifurcation_id['minima'],
    id_minima_counts=bifurcation_id['minima_counts'],
    id_maxima=bifurcation_id['maxima'],
    id_maxima_counts=bifurcation_id['maxima_counts'],
    id_extrema=bifurcation_id['extrema'],
    id_extrema_counts=bifurcation_id['extrema_counts'],
    id_fixed_points=bifurcation_id['fixed_points'],
    id_fixed_points_counts=bifurcation_id['fixed_points_counts'],
    ood_control_params=bifurcation_ood['control_params'],
    ood_minima=bifurcation_ood['minima'],
    ood_minima_counts=bifurcation_ood['minima_counts'],
    ood_maxima=bifurcation_ood['maxima'],
    ood_maxima_counts=bifurcation_ood['maxima_counts'],
    ood_extrema=bifurcation_ood['extrema'],
    ood_extrema_counts=bifurcation_ood['extrema_counts'],
    ood_fixed_points=bifurcation_ood['fixed_points'],
    ood_fixed_points_counts=bifurcation_ood['fixed_points_counts'],
    relevant_dim=bif_relevant_dim,
    transient_steps=bif_transient_steps,
    tail_steps=bif_tail_steps,
    fixed_point_atol=bif_fixed_point_atol,
)

# plotting trajectories for all rho values
fig, axes = plt.subplots(4, 8, figsize=(24, 12))
axes = axes.flatten()

for idx, cp in enumerate(cps):
    ax = axes[idx]
    if idx < cps_id.shape[0]:  # ID
        id_idx = idx
        for j in range(num_ICs):
            traj = X_id_standardized[id_idx, j, :, :]
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.5, linewidth=0.5)
        for j in range(num_ICs):
            traj = X_train_standardized[id_idx, j, :, :]
            ax.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.6, linewidth=0.8)
        ax.set_title(f'rho={cp:.2f} (ID)', fontsize=9)
    else:  # OOD
        ood_idx = idx - cps_id.shape[0]
        for j in range(num_ICs):
            traj = X_ood_standardized[ood_idx, j, :, :]
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.5, linewidth=0.5)
        ax.set_title(f'rho={cp:.2f} (OOD)', fontsize=9)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.tick_params(labelsize=7)

plt.suptitle('Lorenz63 Trajectories (blue=train, gray=test)', fontsize=12)
plt.tight_layout()
plt.savefig('lorenz63_trajectories.png', dpi=100)
plt.show()

