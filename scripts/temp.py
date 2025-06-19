#!/usr/bin/env python3
# visualize_displacement.py
# Enhanced Mountain Slope Displacement Monitoring

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.spatial import KDTree
from scipy import stats

def load_and_preprocess(base_path, latest_path, voxel_size=0.05):
    """Load and preprocess point clouds"""
    # Load point clouds
    pcd_base = o3d.io.read_point_cloud("data/sim1/1691030283.707680940.pcd")
    pcd_latest = o3d.io.read_point_cloud("data/sim1/1691030372082529.pcd")
    
    # Downsample
    pcd_base_down = pcd_base.voxel_down_sample(voxel_size)
    pcd_latest_down = pcd_latest.voxel_down_sample(voxel_size)
    
    # Remove outliers
    pcd_base_down, _ = pcd_base_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd_latest_down, _ = pcd_latest_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    
    return pcd_base_down, pcd_latest_down
def align_point_clouds(base, latest, threshold=0.5):
    """Align point clouds using ICP"""
    trans_init = np.identity(4)
    reg = o3d.pipelines.registration.registration_icp(
        latest, base, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return latest.transform(reg.transformation)

def calculate_displacements(base, aligned, alert_threshold=5.0):
    """Calculate displacements and identify critical points"""
    base_points = np.asarray(base.points)
    aligned_points = np.asarray(aligned.points)
    
    # Build KDTree for base cloud
    base_tree = KDTree(base_points)
    
    displacements = []
    alert_locations = []
    
    for point in aligned_points:
        # Find nearest neighbor in base cloud
        dist, idx = base_tree.query(point)
        base_point = base_points[idx]
        
        # Calculate vertical displacement (most important for slopes)
        displacement = abs(point[2] - base_point[2])
        displacements.append(displacement)
        
        # Record critical displacements
        if displacement > alert_threshold:
            alert_locations.append((point, displacement))
    
    return np.array(displacements), alert_locations

# def create_displacement_colors(displacements, max_disp=10.0):
#     """Create color map where:
#        - Blue (low displacement)
#        - Green (medium displacement) 
#        - Red (high displacement)"""
#     colors = np.zeros((len(displacements), 3))
    
#     for i, d in enumerate(displacements):
#         # Normalize displacement (0-1)
#         norm_d = min(d / max_disp, 1.0)
        
#         # Smooth gradient from blue to red
#         if norm_d < 0.5:  # Blue to Green transition
#             colors[i] = [0, 2*norm_d, 1 - 2*norm_d]

#         else:  # Green to Red transition
#             colors[i] = [2*(norm_d-0.5), 1 - 2*(norm_d-0.5), 0]            
            
#     return colors

def create_displacement_colors(displacements, max_disp=0.5):
    """Create color map with smooth transition:
       - Blue: Low displacement
       - Green: Medium displacement
       - Red: High displacement"""
    colors = np.zeros((len(displacements), 3))
    
    for i, d in enumerate(displacements):
        # Normalize displacement (0-1)
        norm_d = min(d / max_disp, 1.0)
        
        if norm_d < 0.5:  # Blue to Green transition
            # At 0: blue [0,0,1], at 0.5: green [0,1,0]
            r = 0
            g = 2 * norm_d
            b = 1 - 2 * norm_d
            colors[i] = [r, g, b]
        else:  # Green to Red transition
            # At 0.5: green [0,1,0], at 1.0: red [1,0,0]
            r = 2 * (norm_d - 0.5)
            g = 1 - 2 * (norm_d - 0.5)
            b = 0
            colors[i] = [r, g, b]   
            
    return colors

def visualize_results(pcd, displacements, alert_locations):
    """Visualize the displacement results"""
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Terrain Displacement Analysis', width=1400, height=900)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Add coordinate frame
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    
    # Configure rendering
    render_opt = vis.get_render_option()
    render_opt.point_size = 4.0
    render_opt.background_color = np.array([0.1, 0.1, 0.2])
    
    # Print statistics and alerts
    print("\n=== DISPLACEMENT ANALYSIS ===")
    print(f"Mean displacement: {np.mean(displacements):.2f}m")
    print(f"Max displacement: {np.max(displacements):.2f}m")
    print(f"Points >5m displacement: {len(alert_locations)}")
    
    if alert_locations:
        print("\n⚠️ CRITICAL AREAS DETECTED:")
        for loc, dist in alert_locations[:5]:
            print(f"  {dist:.2f}m at X:{loc[0]:.1f} Y:{loc[1]:.1f} Z:{loc[2]:.1f}")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    # Configuration
    BASE_PATH = "/home/bilawal/pcd_monitoring/data/sim1/1691030283.707680940.pcd"
    LATEST_PATH = "/home/bilawal/pcd_monitoring/data/sim1/1691030372082529.pcd"
    VOXEL_SIZE = 0.01
    ALERT_THRESHOLD = 0.5
    #pcd_base = o3d.io.read_point_cloud("data/1691030283.707680940.pcd")
    #pcd_latest = o3d.io.read_point_cloud("data/1691030372082529.pcd")
    try:
        # 1. Load and preprocess
        pcd_base, pcd_latest = load_and_preprocess(BASE_PATH, LATEST_PATH, VOXEL_SIZE)
        
        # 2. Align point clouds
        pcd_aligned = align_point_clouds(pcd_base, pcd_latest)
        
        # 3. Calculate displacements
        displacements, alert_locations = calculate_displacements(pcd_base, pcd_aligned, ALERT_THRESHOLD)
        
        # 4. Create color mapping
        colors = create_displacement_colors(displacements)
        pcd_aligned.colors = o3d.utility.Vector3dVector(colors)
        
        # 5. Visualize results
        visualize_results(pcd_aligned, displacements, alert_locations)
        
        # 6. Save output
        o3d.io.write_point_cloud("/home/bilawal/pcd_monitoring/output/displacement_analysis.pcd", pcd_aligned)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()