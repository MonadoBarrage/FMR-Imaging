"""
Transform point clouds 
"""
import sys
import json
import copy
import torch
import numpy as np
import open3d
from se_math import transforms


def transform_point_cloud(pcd_path=None, pcd_file=None):
    """
    Transforms point clouds
    """


    if not pcd_path and not pcd_file:
        print("Nothing. quitting")
        sys.exit()

    with open('config.json',"r", encoding='utf-8') as config_file:
        config_data = json.load(config_file)
    print("Successfully read config.json")
    pcd = None


    if pcd_path:
        pcd = open3d.io.read_point_cloud(pcd_path)
    elif pcd_file:
        pcd = copy.deepcopy(pcd_file)
    

    R = pcd.get_rotation_matrix_from_xyz((config_data["rotation"]["x-axis"], config_data["rotation"]["y-axis"], config_data["rotation"]["z-axis"]))
    t = np.array([config_data["translation"]["x-cord"], config_data["translation"]["y-cord"], config_data["translation"]["z-cord"]])
    
    
    # pcd.translate(t,True)
    # center = pcd.get_center()
    # pcd = pcd.rotate(R,center)

    g_gt = np.eye(4)
    g_gt[:3, :3] = R
    g_gt[:3, 3] = pcd.get_center()

    pcd.transform(g_gt)

    return pcd, torch.tensor(g_gt)

    # trans = transforms.RandomTransformSE3(0.8, True)

    # p0_src_tensor = torch.tensor((np.asarray(pcd.points)),dtype=torch.float)
    # p0_tensor_transformed = trans(p0_src_tensor)
    
    # p1_src = p0_tensor_transformed.cpu().numpy()
    
    
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(p1_src)

    
    # return pcd
