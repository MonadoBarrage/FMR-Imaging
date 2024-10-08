"""
Demo the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2021-04-13
"""
import os
import sys
import copy
import open3d
import torch
import torch.utils.data
import logging
import numpy as np
from model import PointNet, Decoder, SolveRegistration
import se_math.transforms as transforms
from datetime import datetime

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

output_dir = "./data/output/"

# visualize the point clouds
def draw_registration_result(source, target, transformation, time_str):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    open3d.io.write_point_cloud(f"{output_dir}/source_pre.ply", source_temp)

    print("Visualizing point clouds BEFORE FMR.....")
    open3d.visualization.draw_geometries([source_temp, target_temp])
    source_temp.transform(transformation)
    open3d.io.write_point_cloud(f"{output_dir}/source.ply", source_temp)
    open3d.io.write_point_cloud(f"{output_dir}/target.ply", target_temp)

    
    first_cloud = time_str + "_source.ply"
    second_cloud = time_str + "_target.ply"

    if (os.path.exists(first_cloud) or os.path.exists(second_cloud)):
        i = 1
        while (true):
            first_cloud = time_str + "_source_" + i + ".ply"
            second_cloud = time_str + "_target_" + i + ".ply"

            if (not os.path.exists(first_cloud)) and (not os.path.exists(second_cloud)):
                break
        
            i = i + 1

    open3d.io.write_point_cloud(f"{output_dir}/source.ply", source_temp)
    open3d.io.write_point_cloud(f"{output_dir}/target.ply", target_temp)

    open3d.io.write_point_cloud(first_cloud, source_temp)
    open3d.io.write_point_cloud(second_cloud, target_temp)

    print("Visualizing point clouds AFTER FMR.....")
    open3d.visualization.draw_geometries([source_temp, target_temp])
    

    # image = open3d.visualization.Visualizer.capture_screen_float_buffer(source_temp, do_render=False)
    # open3d.io.write_image("/userdata/end.png", image)

class Demo:
    def __init__(self):
        self.dim_k = 1024
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points, not used during the evaluation
        decoder = Decoder()
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, p0, p1, device):
        solver.eval()
        with torch.no_grad():
            p0 = torch.tensor(p0,dtype=torch.float).to(device)  # template (1, N, 3)
            p1 = torch.tensor(p1,dtype=torch.float).to(device)  # source (1, M, 3)
            solver.estimate_t(p0, p1, self.max_iter)

            est_g = solver.g  # (1, 4, 4)
            g_hat = est_g.cpu().contiguous().view(4, 4)  # --> [1, 4, 4]

            return g_hat

def main(p0, p1, p0_pcd, p1_pcd, time_str):
    fmr = Demo()
    model = fmr.create_model()
    pretrained_path = "./result/fmr_model_modelnet40.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = "cpu"
    model.to(device)
    T_est = fmr.evaluate(model, p0, p1, device)

    draw_registration_result(p1_pcd, p0_pcd, T_est, time_str)

if __name__ == '__main__':

    if ( len(sys.argv) == 2 and sys.argv[1] == "-h"):
        print("demo.py")
        print("FMR algorithm is used to match 3D point clouds together and \n")
        print("1. demo.py -> Uses sample point cloud files from data/sample")
        print("2. demo.py {-p} {}")
        print("3. demo.py {-p} {} {}")
        print("4. demo.py {-i} {} {}")
        print("5. demo.py {-i} {} {} {} {}")
        print("6. demo.py {-r}")
        exit()

    if not os.path.exists(f"{output_dir}/"): 
        os.makedirs(f"{output_dir}/") 

    now_time_str = f"{output_dir}/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



    match len(sys.argv):
        
        case 1:
            #Uses default files 
            print("Using default source files.....") 
            path0 = "./data/sample/src0.ply"
            path1 = "./data/sample/src1.ply"
            if (not os.path.exists(path0)):
                print("ERROR! src0.ply does not exist in data/sample folder. Exiting program.....")
                exit()
            elif (not os.path.exists(path1)):
                print("ERROR! src1.ply does not exist in data/sample folder. Exiting program.....")
                exit()
        case 2:
            #Uses ply files made in the last run of this program
            if(sys.argv[1] == "-r"):
                print("Using previously generated point cloud files.....")
                path0 = f"{output_dir}/source.ply"
                path1 = f"{output_dir}/target.ply"
                if (not os.path.exists(path0)):
                    print("ERROR! source.ply does not exist in data/output folder. Exiting program.....")
                    exit()
                elif (not os.path.exists(path1)):
                    print("ERROR! target.ply does not exist in data/output folder. Exiting program.....")
                    exit()
            
            else:
                print("ERROR! Invalid arguments given. Exiting program.....")
                exit()

        case 3:

            #Takes in one point cloud specified by user. Creates another point cloud by transforming the previous cloud.
            if sys.argv[1] == "-p":
                print("Using provided point cloud and creating another point cloud.....")
                path0 = sys.argv[2]
                if (not os.path.exists(path0)):
                    print(f"ERROR! {path0} does not exist.. Exiting program.....")
                    exit()
                
                path1 = now_time_str + "_point_cloud.ply"
                if (os.path.exists(path1)): 
                    i = 1
                    while (not os.path.exists(path1)):
                        path1 = now_time_str + "_point_cloud_" + i + ".ply"
                        i = i + 1

                pcd2 = open3d.io.read_point_cloud(path0)
                pcd2.translate((1.2,0,0),True)

                open3d.io.write_point_cloud(path1, pcd2)
                print(f"Created {path1} .")
                
            else:
                print("ERROR! Invalid arguments given. Exiting program.....")
                exit()
        case 4: 
            
            #Uses two point cloud files specified by user
            if sys.argv[1] == "-p":
                print("Using two provided point clouds.....")
                path0 = sys.argv[2]
                path1 = sys.argv[3]
                if (not os.path.exists(path0)):
                    print(f"ERROR! {path0} does not exist.. Exiting program.....")
                    exit()
                elif (not os.path.exists(path1)):
                    print(f"ERROR! {path1} does not exist. Exiting program.....")
                    exit()

            #Uses rgbd data given by user and creates one point cloud with it. Creates another point cloud by transforming the first cloud
            elif sys.argv[1] == "-i":
                print("Using provided image data to make one point cloud and transforming created point cloud to get another.....")
                if (not os.path.exists(sys.argv[2])):
                    print(f"ERROR! {sys.argv[2]} does not exist.. Exiting program.....")
                    exit()
                elif (not os.path.exists(sys.argv[3])):
                    print(f"ERROR! {sys.argv[3]} does not exist. Exiting program.....")
                    exit()

                path0 = now_time_str + "_rgbd1.ply"
                path1 = now_time_str + "_rgbd2.ply"
                if (os.path.exists(path0) or os.path.exists(path1)):
                    i = 1
                    while (not os.path.exists(path0)) and (not os.path.exists(path1)):
                        path0 = now_time_str + "_rgbd1_" + i + ".ply"
                        path1 = now_time_str + "_rgbd2_" + i + ".ply"
                        i = i + 1


                color_raw1 = open3d.io.read_image(sys.argv[2])
                depth_raw1 = open3d.io.read_image(sys.argv[3])
                rgbd_image1 = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw1, depth_raw1)
                pcd1 = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image1,open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
                

                # R = pcd2.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
                # pcd2.rotate(R,pcd2.get_center())
                
                open3d.io.write_point_cloud(path0, pcd1)
                print(f"Created {path0} .")


                pcd2 = open3d.io.read_point_cloud(path0)            
                pcd2.translate((1.2,5,5),True)

                open3d.io.write_point_cloud(path1, pcd2)
                print(f"Created {path1} .")
                # open3d.visualization.draw_geometries([pcd2])
            

            else:
                print("ERROR! Invalid arguments given. Exiting program.....")
                exit()

        case 6:

            if sys.argv[1] == "-i":
                print("Using provided image data to make two point clouds.....")

                if (not os.path.exists(sys.argv[2])):
                    print(f"ERROR! {sys.argv[2]} does not exist.. Exiting program.....")
                    exit()
                elif (not os.path.exists(sys.argv[3])):
                    print(f"ERROR! {sys.argv[3]} does not exist.. Exiting program.....")
                    exit()
                elif (not os.path.exists(sys.argv[4])):
                    print(f"ERROR! {sys.argv[4]} does not exist.. Exiting program.....")
                    exit()
                elif (not os.path.exists(sys.argv[5])):
                    print(f"ERROR! {sys.argv[5]} does not exist. Exiting program.....")
                    exit()

                path0 = now_time_str + "_rgbd1.ply"
                path1 = now_time_str + "_rgbd2.ply"
                if (os.path.exists(path0) or os.path.exists(path1)):
                    i = 1
                    while (not os.path.exists(path0)) and (not os.path.exists(path1)):
                        path0 = now_time_str + "_rgbd1_" + i + ".ply"
                        path1 = now_time_str + "_rgbd2_" + i + ".ply"
                        i = i + 1


                color_raw1 = open3d.io.read_image(sys.argv[2])
                depth_raw1 = open3d.io.read_image(sys.argv[3])
                rgbd_image1 = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw1, depth_raw1)
                pcd1 = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image1,open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
                

                pcd2 = pcd1              
                pcd2.translate((1.2,0,0),True)
                # R = pcd2.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
                # pcd2.rotate(R,pcd2.get_center())
                
                open3d.io.write_point_cloud(path0, pcd1)
                print(f"Created {path0} .")

                open3d.io.write_point_cloud(path1, pcd2)
                print(f"Created {path1} .")
                
            else:
                print("ERROR! Invalid arguments given. Exiting program.....")
                exit()


        case _:
            print("ERROR! Invalid number of arguments. Exiting program.....")
            exit()

        

    p0_src = open3d.io.read_point_cloud(path0)
    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.05)
    p0 = np.asarray(downpcd0.points)
    p0 = np.expand_dims(p0,0)

   
    # generate random rotation sample
    # trans = transforms.RandomTransformSE3(0.8, True)
    # p0_src_tensor = torch.tensor((np.asarray(p0_src.points)),dtype=torch.float)
    # p0_tensor_transformed = trans(p0_src_tensor)
    # p1_src = p0_tensor_transformed.cpu().numpy()
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(p1_src)
    # open3d.io.write_point_cloud(path1, pcd)
    

    p1 = open3d.io.read_point_cloud(path1)
    downpcd1 = p1.voxel_down_sample(voxel_size=0.05)
    p1 = np.asarray(downpcd1.points)
    p1 = np.expand_dims(p1, 0)
    main(p0, p1, downpcd0, downpcd1, now_time_str)
