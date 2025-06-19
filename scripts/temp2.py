#!/usr/bin/env python3
# visualize_displacement.py
# Enhanced Mountain Slope Displacement Monitoring with Smooth Heatmap

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
    pcd_base = o3d.io.read_point_cloud(base_path)
    pcd_latest = o3d.io.read_point_cloud(latest_path)
    
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

def create_displacement_colors(displacements, max_disp=0.5):
    """Create smooth color transition:
       - Blue: Low displacement
       - Cyan: Medium-low displacement
       - Green: Medium displacement
       - Yellow: Medium-high displacement
       - Red: High displacement"""
    colors = np.zeros((len(displacements), 3))
    
    for i, d in enumerate(displacements):
        # Normalize displacement (0-1)
        norm_d = min(d / max_disp, 1.0)
        
        # Continuous color progression with smooth transitions
        if norm_d < 0.2:  # Dark blue to light blue
            ratio = norm_d / 0.2
            r = 0
            g = 0.3 * ratio
            b = 1 - 0.5 * ratio
            
        elif norm_d < 0.4:  # Blue to cyan
            ratio = (norm_d - 0.2) / 0.2
            r = 0
            g = 0.3 + 0.7 * ratio
            b = 0.5 + 0.5 * ratio
            
        elif norm_d < 0.6:  # Cyan to green
            ratio = (norm_d - 0.4) / 0.2
            r = 0
            g = 1
            b = 1 - ratio
            
        elif norm_d < 0.8:  # Green to orange
            ratio = (norm_d - 0.6) / 0.2
            r = ratio
            g = 1 - 0.5 * ratio
            b = 0
            
        else:  # Orange to red
            ratio = (norm_d - 0.8) / 0.2
            r = 1
            g = 0.5 - 0.5 * ratio
            b = 0
            
        colors[i] = [r, g, b]   
            
    return colors

def visualize_results(pcd, displacements, alert_locations, max_disp=0.5):
    """Visualize the displacement results with smooth heatmap"""
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
    
    # Create a color legend in the 3D view
    create_3d_legend(vis, max_disp)
    
    # Print statistics and alerts
    print("\n=== DISPLACEMENT ANALYSIS ===")
    print(f"Mean displacement: {np.mean(displacements):.4f}m")
    print(f"Max displacement: {np.max(displacements):.4f}m")
    print(f"Points >{max_disp}m displacement: {len(alert_locations)}")
    
    # Print color legend in console
    print("\nCOLOR LEGEND:")
    print("🔵 Dark Blue: 0-10% displacement")
    print("🔵 Light Blue: 10-20% displacement")
    print("🟢 Cyan: 20-40% displacement")
    print("🟢 Green: 40-60% displacement")
    print("🟡 Yellow-Green: 60-80% displacement")
    print("🟠 Orange: 80-90% displacement")
    print("🔴 Red: 90-100% displacement")
    
    if alert_locations:
        print("\n⚠️ CRITICAL AREAS DETECTED:")
        for loc, dist in alert_locations[:5]:
            print(f"  {dist:.4f}m at X:{loc[0]:.2f} Y:{loc[1]:.2f} Z:{loc[2]:.2f}")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def create_3d_legend(vis, max_disp=0.5):
    """Create 3D color legend in the visualization"""
    # Create a vertical bar for the legend
    bar_height = max_disp * 1.2
    bar_width = 0.1
    bar_depth = 0.1
    bar_position = [-3, -3, 0]  # Position in 3D space
    
    # Create color gradient bar
    num_segments = 100
    for i in range(num_segments):
        height_frac = i / num_segments
        displacement_val = height_frac * max_disp
        
        # Create a small box for this segment
        box = o3d.geometry.TriangleMesh.create_box(width=bar_width, 
                                                  height=bar_height/num_segments, 
                                                  depth=bar_depth)
        
        # Position the box
        box.translate([bar_position[0], 
                      bar_position[1], 
                      bar_position[2] + height_frac * bar_height])
        
        # Calculate color for this displacement value
        norm_d = height_frac
        if norm_d < 0.2:
            ratio = norm_d / 0.2
            color = [0, 0.3 * ratio, 1 - 0.5 * ratio]
        elif norm_d < 0.4:
            ratio = (norm_d - 0.2) / 0.2
            color = [0, 0.3 + 0.7 * ratio, 0.5 + 0.5 * ratio]
        elif norm_d < 0.6:
            ratio = (norm_d - 0.4) / 0.2
            color = [0, 1, 1 - ratio]
        elif norm_d < 0.8:
            ratio = (norm_d - 0.6) / 0.2
            color = [ratio, 1 - 0.5 * ratio, 0]
        else:
            ratio = (norm_d - 0.8) / 0.2
            color = [1, 0.5 - 0.5 * ratio, 0]
        
        box.paint_uniform_color(color)
        vis.add_geometry(box)
    
    # Add labels
    for height in [0, max_disp/4, max_disp/2, 3*max_disp/4, max_disp]:
        text = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        text.translate([bar_position[0] + bar_width + 0.1, 
                       bar_position[1], 
                       bar_position[2] + (height/max_disp) * bar_height])
        text.paint_uniform_color([1, 1, 1])
        vis.add_geometry(text)

def main():
    # Configuration
    BASE_PATH = "/home/bilawal/pcd_monitoring/data/sim1/1691030283.707680940.pcd"
    LATEST_PATH = "/home/bilawal/pcd_monitoring/data/sim1/1691030372082529.pcd"
    VOXEL_SIZE = 0.05
    ALERT_THRESHOLD = 0.5
    MAX_DISP_FOR_COLORS = ALERT_THRESHOLD  # Use same for consistent scaling
    
    try:
        # 1. Load and preprocess
        pcd_base, pcd_latest = load_and_preprocess(BASE_PATH, LATEST_PATH, VOXEL_SIZE)
        
        # 2. Align point clouds
        pcd_aligned = align_point_clouds(pcd_base, pcd_latest)
        
        # 3. Calculate displacements
        displacements, alert_locations = calculate_displacements(pcd_base, pcd_aligned, ALERT_THRESHOLD)
        
        # 4. Create color mapping
        colors = create_displacement_colors(displacements, MAX_DISP_FOR_COLORS)
        pcd_aligned.colors = o3d.utility.Vector3dVector(colors)
        
        # 5. Visualize results
        visualize_results(pcd_aligned, displacements, alert_locations, MAX_DISP_FOR_COLORS)
        
        # 6. Save output
        output_path = "/home/bilawal/pcd_monitoring/output/displacement_analysis2.pcd"
        o3d.io.write_point_cloud(output_path, pcd_aligned)
        print(f"\nSaved displacement visualization to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting mountain slope displacement monitoring...")
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f} seconds")