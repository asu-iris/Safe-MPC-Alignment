import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from skimage import measure
import trimesh
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt


class baseModel(torch.nn.Module):
    def __init__(self, params):
        super(baseModel, self).__init__()
        self.p_fns = ["sin","cos"]
        for k, v in vars(params).items():
            print(k,v)
            setattr(self, k, v)
        self.freqs = 2. ** torch.linspace(0., self.max_freq, steps=self.max_freq + 1) #10-20
        self.in_ch = 3 * (len(self.p_fns) * (self.max_freq + 1) + 1)
    def embed(self, coords):
        return torch.cat([coords, *[getattr(torch, p_fn)(coords * freq) for freq in self.freqs for p_fn in self.p_fns]], -1)


class NeRFMLP(baseModel):
    def __init__(self, params):
        super(NeRFMLP, self).__init__(params)
        self.coords_MLP = nn.ModuleList(
            [nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD - 1)]]
        )
        self.feature_MLP = nn.Linear(self.netW, self.feat_dim)
        self.out_MLP = nn.Linear(self.feat_dim, self.out_ch)
    def forward(self, x):
        x = self.embed(x)
        h = x
        for idx, mlp in enumerate(self.coords_MLP):
            h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h))
        feat = self.feature_MLP(h)
        out = self.out_MLP(feat)
        return feat,out
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

class sdf_param:
    def __init__(self) -> None:
        self.max_freq = 15
        self.netW = 128
        self.feat_dim = 16
        self.netD = 6
        self.skips = [3]
        self.out_ch = 1

def pairwise_distances(reference, query):
    # Expand dimensions for broadcasting
    ref_expand = reference.unsqueeze(0)  # (1, n_ref, d)
    query_expand = query.unsqueeze(1)    # (batch_size, 1, d)
    
    # Compute pairwise differences and distances
    diff = query_expand - ref_expand   # (batch_size, n_ref, d)
    dist = torch.norm(diff, dim=-1) - 2.5   # (batch_size, n_ref)
    min_idx, min_dist = torch.min(dist,dim=1).indices, torch.min(dist,dim=1).values
    min_idx = min_idx.view(query.shape[0], 1, 1).expand(-1, 1, 3)
    normals =  torch.gather(diff, dim=1, index=min_idx).squeeze(1)
    normals = normals/torch.norm(normals,p=2,dim=-1).unsqueeze(-1)
    print(normals[5000])
    return min_dist, normals

import numpy as np

def sample_near_surface_sdf(pointcloud, normals=None, n_samples_per_point=10, epsilon=0.01):
    """
    Sample points near a point cloud surface for SDF training.

    Args:
        pointcloud (Nx3 np.ndarray): surface points
        normals (Nx3 np.ndarray or None): normals to assign SDF sign (optional)
        n_samples_per_point (int): number of samples around each surface point
        epsilon (float): stddev of Gaussian noise added to surface points

    Returns:
        query_points (Mx3 np.ndarray): sampled 3D points
        sdf_values (Mx1 np.ndarray): signed distances (approximate)
    """
    N = len(pointcloud)
    query_points = []
    sdf_values = []

    for i in range(N):
        p = pointcloud[i]
        for _ in range(n_samples_per_point):
            offset = np.random.normal(scale=epsilon, size=3)
            q = p + offset
            d = np.linalg.norm(offset)
            sign = 1.0

            if normals is not None:
                n = normals[i]
                sign = np.sign(np.dot(offset, n))
                if sign == 0:
                    sign = 1.0  # edge case
            else:
                sign = np.random.choice([-1, 1])  # random sign if normals are unavailable

            query_points.append(q)
            sdf_values.append(sign * d)

    return np.array(query_points)

if __name__=="__main__":
    import open3d as o3d
    curve_points = np.load("../Data/uav_revise/curve_full.npy")
    surface_points = np.load("../Data/uav_revise/circle_full.npy")
    normals = np.load("../Data/uav_revise/normal_full.npy")

    curve_tensor = torch.tensor(curve_points,dtype=torch.float32)

    space_points = sample_near_surface_sdf(surface_points, n_samples_per_point=20, epsilon = 1.0)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(space_points[:,0], space_points[:,1], space_points[:,2], s=1)
    ax.set_title("Half-sphere (x < 0)")
    ax.set_box_aspect([1,1,1])
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(space_points)

    space_points  = torch.tensor(space_points ,dtype=torch.float32)

    dists, normals = pairwise_distances(curve_tensor,space_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.normalize_normals()
    pcd_down = pcd.uniform_down_sample(20)

    # o3d.visualization.draw_geometries([pcd_down],
    #                               point_show_normal=True)

    points_scale = space_points/15
    dists_scale = dists/15

    

    normals_tensor = torch.tensor(normals,dtype=torch.float32)

    # sdf_values = dists_scale.reshape(64, 64, 64).cpu().numpy()
    # verts, faces, normals, values = measure.marching_cubes(sdf_values, level=0.0, spacing=(sample_x.cpu()[1]-sample_x.cpu()[0],
    #                                                                                         sample_y.cpu()[1]-sample_y.cpu()[0], sample_z.cpu()[1]-sample_z.cpu()[0]))
    # # Create mesh
    # mesh_gt = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    # Visualize

    #dataset
    dataset = TensorDataset(points_scale, dists_scale, normals_tensor)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = NeRFMLP(params=sdf_param())
    model.to("cuda")
    EPOCHES = 250
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr = 0.001)
    for i in range(EPOCHES):
        epoch_loss = 0
        for p,d,n in dataloader:
            p = p.cuda().requires_grad_()
            d = d.cuda()
            n = n.cuda()

            optim.zero_grad()
            _, out = model(p)
            dist_loss = loss_fn(out.flatten(),d.flatten())
            p_grad = torch.autograd.grad(out.sum(), p, create_graph=True)[0]
            norm_loss = (1 - torch.cosine_similarity(p_grad, n, dim=-1)).mean()

            eikonal_loss = ((p_grad.norm(dim=-1) - 1)**2).mean()

            # p_withgrad = p.clone().detach()
            # p_withgrad.requires_grad_()
            # _, out_g = model(p_withgrad)
            # p_grad = torch.autograd.grad(out_g.sum(),p_withgrad,create_graph=True, retain_graph=True,only_inputs=True)[0]
            # norm_loss = (1 - torch.cosine_similarity(p_grad,n)).mean()

            loss = dist_loss + 0.1 * norm_loss + 0.01 * eikonal_loss
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        if (i+1) % 10 == 0:
            print("loss at epoch {}".format(i),epoch_loss)

    torch.save(model.state_dict(), "../Data/uav_revise/sdf_weights_2.pt")
    grid_size = 100  # can go higher if you have GPU memory
    x = torch.linspace(-0.5, 1.5, grid_size,device="cuda",dtype=torch.float32)
    y = torch.linspace(-0.5, 0.5, grid_size,device="cuda",dtype=torch.float32)
    z = torch.linspace(0.0, 1.0, grid_size,device="cuda",dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)

    with torch.no_grad():
        sdf_values = model(points)[1].reshape(grid_size, grid_size, grid_size).cpu().numpy()

    verts, faces, normals, values = measure.marching_cubes(sdf_values, level=0.0, spacing=(x.cpu()[1]-x.cpu()[0], y.cpu()[1]-y.cpu()[0], z.cpu()[1]-z.cpu()[0]))

    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Visualize
    mesh.show()
    mesh_gt.show()

    

