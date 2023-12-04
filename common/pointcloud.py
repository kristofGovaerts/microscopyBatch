import pandas as pd
import open3d as o3d
import numpy as np


def read_pointcloud_from_csv(csv_loc):
    df = pd.read_csv(csv_loc)
    df.columns = ['x','y','z','c']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(df[['x', 'y', 'z']]))
    return pcd
