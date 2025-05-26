import numpy as np
import torch
from torch_sdf import NeRFMLP, sdf_param

def linear_interpolate_axis0(data):
    """
    Linearly interpolates along axis 0 for a n x 3 array.
    Assumes NaNs are present and should be interpolated.
    """
    if data.shape[1] != 3:
        raise ValueError("Input array must have shape (n, 3)")

    interpolated = np.copy(data)
    n, _ = data.shape

    for col in range(3):
        col_data = data[:, col]
        valid = ~np.isnan(col_data)
        if np.sum(valid) < 2:
            continue  # Not enough points to interpolate

        interpolated[:, col] = np.interp(
            np.arange(n),
            np.flatnonzero(valid),
            col_data[valid]
        )
    
    return interpolated

geom_model = NeRFMLP(sdf_param())
geom_model.load_state_dict(torch.load("../Data/uav_revise/sdf_weights.pt"))
geom_model.to("cuda")

for param in geom_model.out_MLP.parameters():
    print(param)
curve_points = np.load("../Data/uav_revise/curve_full.npy")
interp_curve = linear_interpolate_axis0(curve_points)
# print(curve_points)
# exit()
# init_r = np.array([0.0,0.0,5.0]).reshape(1,-1)
# target_r = np.array([15.0,0.0,10.0]).reshape(1,-1)
# curve_points = np.concatenate((init_r,curve_points,target_r))

curve_tensor = torch.tensor(curve_points,dtype=torch.float32,device = "cuda")
_, dists = geom_model(curve_tensor/15)
# print(dists)
# print((dists>0).sum())