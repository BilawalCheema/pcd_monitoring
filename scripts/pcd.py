import open3d as o3d

# Load point clouds
pcd1 = o3d.io.read_point_cloud("data/1691030283.707680940.pcd")
pcd2 = o3d.io.read_point_cloud("data/1691030372082529.pcd")
#pcd2 = o3d.io.read_point_cloud("data/sim3/1696601885430859.pcd")

# Optional: downsample to improve performance
voxel_size = 0.05
pcd1_down = pcd1.voxel_down_sample(voxel_size)
pcd2_down = pcd2.voxel_down_sample(voxel_size)

# Assign colors to visually differentiate them
pcd1_down.paint_uniform_color([1, 0, 0])  # Red
pcd2_down.paint_uniform_color([0, 0, 1])  # Blue

# Show both in original coordinates
o3d.visualization.draw_geometries([pcd1_down, pcd2_down])
