import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

# Load point clouds
#= o3d.io.read_point_cloud("/home/bilawal/pcd_monitoring/data/1691030283.707680940.pcd")  # base
# = o3d.io.read_point_cloud("/home/bilawal/pcd_monitoring/data/1695907280.927630819.pcd")  # latest
pcd2= o3d.io.read_point_cloud("data/1691030283.707680940.pcd")
pcd1= o3d.io.read_point_cloud("data/1691030372082529.pcd")
# Downsample for performance
voxel_size = 0.05
pcd1_down = pcd1.voxel_down_sample(voxel_size)
pcd2_down = pcd2.voxel_down_sample(voxel_size)

# Align with ICP (basic)
threshold = 0.5
trans_init = np.identity(4)
reg = o3d.pipelines.registration.registration_icp(
    pcd1_down, pcd2_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
pcd1_aligned = copy.deepcopy(pcd1_down).transform(reg.transformation)

# KDTree for nearest neighbor in pcd2
pcd2_tree = o3d.geometry.KDTreeFlann(pcd2_down)
displacements = []

for point in pcd1_aligned.points:
    [_, idx, _] = pcd2_tree.search_knn_vector_3d(point, 1)
    nearest = pcd2_down.points[idx[0]]
    dist = np.linalg.norm(np.asarray(point) - np.asarray(nearest))
    displacements.append(dist)

# Stats
displacements = np.array(displacements)
print(f"Mean displacement: {displacements.mean():.4f}")
print(f"Max displacement: {displacements.max():.4f}")

# Color coding by displacement (for visualization)
colors = plt.cm.jet((displacements - displacements.min()) / (displacements.max() - displacements.min()))[:, :3]
pcd1_aligned.colors = o3d.utility.Vector3dVector(colors)

# Save or visualize
o3d.visualization.draw_geometries([pcd1_aligned])

# Export displaced points
#o3d.io.write_point_cloud("displacement_output.pcd", pcd1_aligned)
