"""
Demo the feature-metric registration algorithm

Author: MonadoBarrage
Original Creator: Xiaoshui Huang

Date: 2024-05-10
"""
import os
import sys
import copy
import logging
import argparse
import json
from datetime import datetime
from typing import Optional, TextIO
from sklearn.metrics import root_mean_squared_error
import open3d
import torch
import torch.utils.data
import numpy as np
from model import PointNet, Decoder, SolveRegistration
from conversion import convert_rgbd_to_pcd
from transform import transform_point_cloud
# from se_math import transforms


g_gt = np.eye(4)

# Pathing and files
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULT_PATH = "demo-results/"
metric_output: Optional[TextIO] = None

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Parser arguments
parser = argparse.ArgumentParser()
main_group = parser.add_mutually_exclusive_group()

first_group = main_group.add_argument_group()
first_group.add_argument('--p1', nargs='+', help="")
subgroup = first_group.add_mutually_exclusive_group()
subgroup.add_argument('--p2', nargs='+',help="")
subgroup.add_argument('--copy', action='store_true',help="")

main_group.add_argument("--r", action='store_true', help="")
parser.add_argument("--no-display", action='store_true', help="")

args = parser.parse_args()

# Functions
def initialize():
    """
    Initializes necessary folders and files
    if needed
    """

    global metric_output
    if not os.path.exists(RESULT_PATH):
        print(f"{RESULT_PATH} folder missing. Creating {RESULT_PATH} folder.....")
        os.makedirs(RESULT_PATH)
        print(f"Created {RESULT_PATH} folder in root directory.\n")

    if not os.path.exists("config.json"):
        print("config.json missing. Creating config.json with default values.....")
        with open('config.json', 'w', encoding='utf-8') as config_file:
            config_settings = {
                "rotation":
                {
                    "x-axis": 40,
                    "y-axis": 40,
                    "z-axis": 40
                },

                "translation":
                {
                    "x-cord": 50, 
                    "y-cord": 50, 
                    "z-cord": 50
                }
            }
            json.dump(config_settings, config_file, ensure_ascii=False, indent=4)
            print("Created config.json in root directory.\n")
    os.makedirs(f'{RESULT_PATH}/{time_str}')
    metric_output = open(f'{RESULT_PATH}/{time_str}/metrics.lock',mode="w",encoding="utf-8")

def handle_args():
    """ Interprets the arguments given by the user"""
    global g_gt

    if not args.p1 and not args.r:
        print("need either --p1 or --r")
        sys.exit()

    if not args.p2 and not args.copy and not args.r:
        print("need either --p2 or --copy argument")
        sys.exit()

    pcd1: open3d.cpu.pybind.geometry.PointCloud = None
    pcd2: open3d.cpu.pybind.geometry.PointCloud = None

    isDisplay = not args.no_display

    if args.p1:
        match len(args.p1):
            case 2:
                if not os.path.exists(args.p1[0]) or not os.path.exists(args.p1[1]):
                    print("Missing args")
                    sys.exit()
                pcd1 = convert_rgbd_to_pcd(color_raw=args.p1[0],depth_raw=args.p1[1])

            case 1:
                if not os.path.exists(args.p1[0]):
                    print("Missing args")
                    sys.exit()
                pcd1 = open3d.io.read_point_cloud(args.p1[0])
            case _:
                print("Invalid amount")
                sys.exit()
        open3d.io.write_point_cloud(f'{RESULT_PATH}/{time_str}/presource.ply', pcd1)
        if args.p2:
            match len(args.p2):
                case 2:
                    if not os.path.exists(args.p2[0]) or not os.path.exists(args.p2[1]):
                        print("Missing args")
                        sys.exit()
                    pcd2 = convert_rgbd_to_pcd(color_raw=args.p2[0],depth_raw=args.p2[1])
                case 1:
                    if not os.path.exists(args.p2[0]):
                        print("Missing args")
                        sys.exit()
                    pcd2 = open3d.io.read_point_cloud(args.p2[0])
                case _:
                    print("Invalid amount")
                    sys.exit()
        elif args.copy:
            pcd2_temp = copy.deepcopy(pcd1)
            pcd2, g_gt = transform_point_cloud(pcd_file=pcd2_temp)
            open3d.io.write_point_cloud(f'{RESULT_PATH}/{time_str}/pretarget.ply', pcd2)
    elif args.r:
        pcd1_path = f"{RESULT_PATH}/source.ply"
        pcd2_path = f"{RESULT_PATH}/target.ply"
        if not os.path.exists(pcd1_path) or not os.path.exists(pcd2_path):
            print(f"source.ply or target.ply does not exist in {RESULT_PATH} folder.")
            sys.exit()
        pcd1 = open3d.io.read_point_cloud(pcd1_path)
        pcd2 = open3d.io.read_point_cloud(pcd2_path)

    return pcd1, pcd2, isDisplay

def draw_registration_result(source, target, transformation, isDisplay):
    """ Visualize the point clouds """

    # pylint: disable=no-member
    if isDisplay:
        vis = open3d.visualization.Visualizer()


    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    # open3d.io.write_point_cloud("source_pre.ply", source_temp)
    # open3d.visualization.draw([source_temp, target_temp])
    if isDisplay:
        vis.create_window()
        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
        vis.run()
        vis.destroy_window()
    
    
    source_temp.transform(transformation)
    open3d.io.write_point_cloud(f'{RESULT_PATH}/{time_str}/source.ply', source_temp)
    open3d.io.write_point_cloud(f'{RESULT_PATH}/{time_str}/target.ply', target_temp)
    open3d.io.write_point_cloud(f'{RESULT_PATH}/source.ply', source_temp)
    open3d.io.write_point_cloud(f'{RESULT_PATH}/target.ply', target_temp)
    # open3d.visualization.draw([source_temp, target_temp])
    if isDisplay:
        vis.create_window()
        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
        vis.run()
        vis.destroy_window()


    g_est = copy.deepcopy(transformation)
    g_est[:3,3] = torch.tensor(target_temp.get_center())

    print(f'Transformation Ground truth:\n{g_gt}\n',file=metric_output)
    print(f'Transformation Estimated:\n{g_est}\n',file=metric_output)

    trace1 = g_gt[:3,:3].trace()
    np_val1 = (trace1 -1) / 2
    cos_theta1 = torch.tensor(np_val1, dtype=torch.float32)
    cos_theta1 = cos_theta1.detach().clone().clamp(-1.0, 1.0)
    angle1 = torch.acos(cos_theta1)

    trace2 = g_est[:3,:3].trace()
    np_val2 = (trace2 -1) / 2
    cos_theta2 = torch.tensor(np_val2, dtype=torch.float32)
    cos_theta2 = cos_theta2.detach().clone().clamp(-1.0, 1.0)
    angle2 = torch.acos(cos_theta2)

    print(f'Rotational Error:\n{torch.norm(angle1-angle2)}\n',file=metric_output)
    print(f'Translational Error:\n{torch.norm(g_est[:3,3] - g_gt[:3,3])}\n',file=metric_output)
    print(f'RMSE:\n{root_mean_squared_error(g_est,g_gt)}\n', file=metric_output)

class Demo:
    """ An object to initiate the demonstration """

    def __init__(self):
        self.dim_k = 1024
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        """ Makes a neural model to extract features and do registration """

        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points, not used during the evaluation
        decoder = Decoder()
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, p0, p1, device):
        """ Estimates registration and calculates for minimal error """      
    
        solver.eval()
        with torch.no_grad():
            p0 = torch.tensor(p0,dtype=torch.float).to(device)  # template (1, N, 3)
            p1 = torch.tensor(p1,dtype=torch.float).to(device)  # source (1, M, 3)
            solver.estimate_t(p0, p1, self.max_iter)

            est_g = solver.g  # (1, 4, 4)
            g_hat = est_g.cpu().contiguous().view(4, 4)  # --> [1, 4, 4]
                     
            return g_hat

def main(p0, p1, p0_pcd, p1_pcd, isDisplay):
    """ Runs the main program """

    fmr = Demo()
    model = fmr.create_model()
    pretrained_path = "./result/fmr_model_7scene.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = "cpu"
    model.to(device)

    time_start = datetime.now()
    t_est = fmr.evaluate(model, p0, p1, device)
    time_end = datetime.now()
    print(f"Time:\n{((time_end-time_start).microseconds)} microseconds\n", file=metric_output)


    draw_registration_result(p1_pcd, p0_pcd, t_est, isDisplay)

if __name__ == '__main__':

    initialize()
    p0_src, p1_src, isDisplay = handle_args()

    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.05)
    p0_main = np.asarray(downpcd0.points)
    p0_main = np.expand_dims(p0_main,0)

    '''
    # generate random rotation sample
    trans = transforms.RandomTransformSE3(0.8, True)
    p0_src_tensor = torch.tensor((np.asarray(p0_src.points)),dtype=torch.float)
    p0_tensor_transformed = trans(p0_src_tensor)
    p1_src = p0_tensor_transformed.cpu().numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(p1_src)
    open3d.io.write_point_cloud(path1, pcd)
    '''

    downpcd1 = p1_src.voxel_down_sample(voxel_size=0.05)
    p1_main = np.asarray(downpcd1.points)
    p1_main = np.expand_dims(p1_main, 0)
    main(p0_main, p1_main, downpcd0, downpcd1,isDisplay)

    os.replace(metric_output.name, f'{RESULT_PATH}/{time_str}/metrics.txt')
    metric_output.close()
    sys.exit()
