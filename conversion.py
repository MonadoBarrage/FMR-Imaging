"""
Converts RGBD files to a single point cloud
"""

import open3d

def convert_rgbd_to_pcd(color_raw, depth_raw):
    """Converts RGBD images to point clouds"""
    color_image = open3d.io.read_image(color_raw)
    depth_image = open3d.io.read_image(depth_raw)

    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    return pcd
    