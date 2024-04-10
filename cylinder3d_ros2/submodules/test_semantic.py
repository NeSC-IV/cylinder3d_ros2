import os
import time
import argparse
import sys
import numpy as np
import torch
import yaml
# import torch.multiprocessing as mp
# import torch_tensorrt
from builder import model_builder
from config.config import load_config_data
from dataloader.dataloader_test import SemKITTI_demo
from utils.load_save_util import load_checkpoint
import open3d as o3d
# from torch2trt import torch2trt

import warnings

warnings.filterwarnings("ignore")

class points_semantic():
    def __init__(self, homepath, args):
        self.pytorch_device = torch.device('cuda:0')
        config_path = args.config_path
        configs = load_config_data(config_path)
        self.dataset_config = configs['dataset_params']

        self.demo_batch_size = 2
        model_config = configs['model_params']
        train_hypers = configs['train_params']

        self.grid_size = model_config['output_shape']

        model_load_path = train_hypers['model_load_path']

        self.my_model = model_builder.build(model_config)
        model_load_path = homepath + "submodules/" + model_load_path
        if os.path.exists(model_load_path):
            self.my_model = load_checkpoint(model_load_path, self.my_model)
            # print("load_model from: ", model_load_path)
        # self.my_model = torch.compile(self.my_model)
        # torch._dynamo.config.suppress_errors = True
        self.my_model.to(self.pytorch_device)
        config_label = homepath + "submodules/" +self.dataset_config["label_mapping"]
        # print(config_label)
        with open(config_label, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.inv_learning_map = semkittiyaml['learning_map_inv']

        self.my_model.eval()

        self.data_preprocesser = SemKITTI_demo(grid_size=self.grid_size,
            ignore_label=self.dataset_config["ignore_label"],
            fixed_volume_space=self.dataset_config['fixed_volume_space'],
            max_volume_space=self.dataset_config['max_volume_space'],
            min_volume_space=self.dataset_config['min_volume_space'],
            )

    def sem_points(self,demo_grid, demo_pt_fea, scan):
        with torch.no_grad():
            # time_dateset = time.time()       
            # [demo_grid, demo_pt_fea] = self.data_preprocesser.preprocess(scan)
            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                demo_pt_fea]     # (n*9)，点对于体素中心极坐标（r,\theta,z）,体素中心极坐标（r,\theta,z）,点笛卡尔坐标（x,y）,intensity
            demo_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in demo_grid] # (n*3)，体素索引
            predict_labels = self.my_model(demo_pt_fea_ten, demo_grid_ten, self.demo_batch_size, val = True)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            count = 0 # 单回波
            inv_labels0 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            count = 1
            inv_labels1 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            sem_points1 = np.hstack ((scan[0],inv_labels0.reshape(-1,1)))
            sem_points2 = np.hstack ((scan[1],inv_labels1.reshape(-1,1)))
            # 删除动态物体
            # points = scan[:,:, 0:4]    # get [x,y,z,remission]print 2*n*4
            #sem_points = np.concatenate((scan,inv_labels),axis=2)
            sem_points = np.array([sem_points1,sem_points2])
            # time_end = time.time()
            # print('time total cost',time_end-time_dateset,'s')
        return sem_points
    
    def sem_points2(self,demo_grid, demo_pt_fea, scan):
        with torch.no_grad():
            time_dateset = time.time()       
            # [demo_grid, demo_pt_fea] = self.data_preprocesser.preprocess(scan)
            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                demo_pt_fea]     # (n*9)，点对于体素中心极坐标（r,\theta,z）,体素中心极坐标（r,\theta,z）,点笛卡尔坐标（x,y）,intensity
            demo_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in demo_grid] # (n*3)，体素索引
            torch.cuda.synchronize()
            time_start_predit = time.time()
            predict_labels = self.my_model(demo_pt_fea_ten, demo_grid_ten, self.demo_batch_size, val = True)
            torch.cuda.synchronize()
            time_end_predict = time.time()
            predict_labels = torch.argmax(predict_labels, dim=1)

            predict_labels = predict_labels.cpu().detach().numpy()
            
            count = 0 # 单回波
            inv_labels0 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            count = 1
            inv_labels1 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            sem_points1 = np.hstack ((scan[0],inv_labels0.reshape(-1,1)))
            sem_points2 = np.hstack ((scan[1],inv_labels1.reshape(-1,1)))
            # 删除动态物体
            # points = scan[:,:, 0:4]    # get [x,y,z,remission]print 2*n*4
            #sem_points = np.concatenate((scan,inv_labels),axis=2)
            sem_points = np.array([sem_points1,sem_points2])
            time_end = time.time()
            # finding the difference between
            # neighboring elements
            print('time total cost',time_end-time_dateset,'s')
            print('time predit cost',time_end_predict-time_start_predit,'s')
        return sem_points

    
    def sem_points1(self,demo_grid, demo_pt_fea, scan):
        with torch.no_grad():
            time_dateset = time.time()       
            # [demo_grid, demo_pt_fea] = self.data_preprocesser.preprocess(scan)
            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                demo_pt_fea]     # (n*4),点对于体素中心极坐标（r,\theta,z）,intensity
            demo_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in demo_grid] # (n*8)，点对于体素中心极坐标（r,\theta,z）,极坐标（r,\theta,z）,笛卡尔坐标（x,y）
            torch.cuda.synchronize()
            time_start_predit = time.time()
            predict_labels = self.my_model(demo_pt_fea_ten, demo_grid_ten, self.demo_batch_size, val = True)
            torch.cuda.synchronize()
            time_end_predict = time.time()
            predict_labels = torch.argmax(predict_labels, dim=1)
            torch.cuda.synchronize()
            time1 = time.time()
            predict_labels = predict_labels.detach().cpu().numpy()
            torch.cuda.synchronize()
            time2 = time.time()
            count = 0 # 单回波
            inv_labels0 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            
            # count = 1
            # inv_labels1 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            sem_points1 = np.hstack ((scan[0],inv_labels0.reshape(-1,1)))
            time3 = time.time()
            # sem_points2 = np.hstack ((scan[1],inv_labels1.reshape(-1,1)))
            # 删除动态物体
            # points = scan[:,:, 0:4]    # get [x,y,z,remission]print 2*n*4
            #sem_points = np.concatenate((scan,inv_labels),axis=2)
            sem_points = np.array([sem_points1])
            time_end = time.time()
            # finding the difference between
            # neighboring elements
            print('time total cost',time_end-time_dateset,'s')
            print('time predit cost',time_end_predict-time_start_predit,'s')
            # print('time predit cost pre',time_start_predit-time_dateset,'s')
            # print('time predit cost post',time_end-time_end_predict,'s')
            # print('time cost 0',time1-time_end_predict,'s')
            # print('time predit cost 1',time2-time1,'s')
            # print('time predit cost 2',time3-time2,'s')
        return sem_points
    
    
if __name__ == '__main__':
    # Training settings
    homepath = "/home/oliver/catkin_ros2/src/cylinder3d_ros2/cylinder3d_ros2/"
    datapath = "/media/oliver/Elements SE/dataset/KITTI/sequences/00/velodyne/"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default=homepath + 'submodules/config/semantickitti8_run.yaml')
    # parser.add_argument('--demo_folder', type=str, default='/home/oliver/sequences/00/velodyne', help='path to the folder containing demo lidar scans')
    # parser.add_argument('--save-folder', type=str, default='/home/oliver/sequences/00/labels', help='path to save your result')
    # parser.add_argument('--demo-label-folder', type=str, default='/home/oliver/sequences/00/labels', help='path to the folder containing demo labels')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    a = points_semantic(homepath,args)
    filename = datapath + '000000.bin'
    scan = np.fromfile(filename, dtype=np.float32)
    scan1 = scan.reshape((-1, 4))
    [demo_grid0, demo_pt_fea0] = a.data_preprocesser.preprocess(scan1)
    # 如果scan1的第一列元素小于0，则删除该元素
    # scan1 = scan1[scan1[:,0] > 0]

    # [demo_grid0, demo_pt_fea0] = a.data_preprocesser.preprocess(scan1)
    filename = datapath + '000001.bin'
    scan = np.fromfile(filename, dtype=np.float32)
    scan2 = scan.reshape((-1, 4))
    [demo_grid1, demo_pt_fea1] = a.data_preprocesser.preprocess(scan2)
    demo_grid = [demo_grid0 , demo_grid0]
    demo_pt_fea = [demo_pt_fea0, demo_pt_fea0]
    # # scan1和scan2 open3d实现NDT匹配
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(scan1[:,0:3])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(scan2[:,0:3])
    # # pcd1.paint_uniform_color([1, 0.706, 0])
    # # pcd2.paint_uniform_color([0, 0.651, 0.929])
    # startime = time.time()
    # # o3d.visualization.draw_geometries([pcd1, pcd2])
    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pcd1, pcd2, 0.2, np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(time.time() - startime)
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")

    print("sem1*****************")
    for i in range(1,10):
        afsem_points = a.sem_points1([demo_grid0], [demo_pt_fea0], np.array([scan1]))
        afsem_points = a.sem_points1([demo_grid0], [demo_pt_fea0], np.array([scan1]))
        # afsem_points = a.sem_points1([demo_grid0[0:30000]], [demo_pt_fea0[0:30000]], np.array([scan1[0:30000]]))
        # afsem_points = a.sem_points1([demo_grid0[0:60000]], [demo_pt_fea0[0:60000]], np.array([scan1[0:60000]]))
    print("sem3*****************")
    for i in range(1,5):
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))


