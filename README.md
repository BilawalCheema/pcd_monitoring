
# 🏔️ Terrain Displacement Monitoring using Point Clouds

This project detects and visualizes terrain/slope displacement using 3D point clouds captured from LiDAR sensors. It uses [Open3D](http://www.open3d.org/) and Python for aligning scans, computing displacements, and visualizing results with a heatmap.

---

## 🧠 Core Features

- Point Cloud Alignment using ICP
- Vertical Displacement Detection
- Heatmap with Smooth Color Gradients (Blue → Red)
- 3D Color Legend in Visualizer
- Automatic Alert for Displacements over Threshold

---

## 📁 Project Structure

```
.
├── temp2.py     # Main code
├── data/                          # Input point clouds
├── output/                        # Output with displacement heatmap
├── README.md                      # Project report and explanation
```

## 🚀 How It Works

### Step 1: Load and Preprocess Point Clouds

```python
pcd_base = o3d.io.read_point_cloud(base_path)
pcd_latest = o3d.io.read_point_cloud(latest_path)
...
pcd_base_down = pcd_base.voxel_down_sample(voxel_size)
```

- Downsampling improves performance
- Outlier removal cleans the scan

---

### Step 2: Align Point Clouds using ICP

```python
reg = o3d.pipelines.registration.registration_icp(
    latest, base, threshold, np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
```

This brings both clouds into the same frame for comparison.

---

### Step 3: Displacement Calculation

```python
displacement = abs(point[2] - base_point[2])
```

- Only vertical (Z-axis) displacement is used
- Displacement > `threshold` is flagged as an alert

---

### Step 4: Color Mapping (Smooth Heatmap)

```python
# Green → Red scale based on displacement magnitude
colors[i] = [r, g, b]
```


---

### Step 5: 3D Visualization

```python
vis.add_geometry(pcd)
vis.run()
```

Includes a 3D color bar and on-console stats.

---

## ⚙️ Configuration

You can change the following in the `main()` function:

```python
VOXEL_SIZE = 0.05
ALERT_THRESHOLD = 0.5  
```

---

## 📦 Dependencies

```bash
pip install open3d numpy matplotlib scipy
```

---

## 📤 Output Example


The final output is saved as:

```bash
output/displacement_analysis2.pcd
```

You can open this in MeshLab or Open3D viewer.
