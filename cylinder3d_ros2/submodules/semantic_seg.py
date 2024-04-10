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
# from torch2trt import torch2trt

import warnings

warnings.filterwarnings("ignore")

class points_semantic():
    def __init__(self, configs):
        self.pytorch_device = torch.device(configs['device'])
        self.dataset_config = configs['dataset_params']

        # self.demo_batch_size = 2
        model_config = configs['model_params']
        train_hypers = configs['train_params']

        self.grid_size = model_config['output_shape']

        model_load_path = train_hypers['model_load_path']

        self.my_model = model_builder.build(model_config)
        model_load_path = configs['homepath'] + "/" + model_load_path
        if os.path.exists(model_load_path):
            self.my_model = load_checkpoint(model_load_path, self.my_model)
            # print("load_model from: ", model_load_path)
        # self.my_model = torch.compile(self.my_model)
        # torch._dynamo.config.suppress_errors = True
        self.my_model.to(self.pytorch_device)
        config_label = configs['homepath'] + "/" +self.dataset_config["label_mapping"]
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
            demo_pt_fea_ten = [torch.from_numpy(demo_pt_fea).type(torch.FloatTensor).to(self.pytorch_device)]     # (n*9)，点对于体素中心极坐标（r,\theta,z）,体素中心极坐标（r,\theta,z）,点笛卡尔坐标（x,y）,intensity
            demo_grid_ten = [torch.from_numpy(demo_grid).to(self.pytorch_device)] # (n*3)，体素索引
            predict_labels = self.my_model(demo_pt_fea_ten, demo_grid_ten, 1, val = True)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()[0]                   

            inv_labels0 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[demo_grid[:, 0], demo_grid[:, 1], demo_grid[:, 2]]).astype('uint32')
            sem_points1 = np.hstack ((scan,inv_labels0.reshape(-1,1)))
        return sem_points1
    
    def sem_points2(self,demo_grid, demo_pt_fea, scan):
        with torch.no_grad():     
            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                demo_pt_fea]     # (n*9)，点对于体素中心极坐标（r,\theta,z）,体素中心极坐标（r,\theta,z）,点笛卡尔坐标（x,y）,intensity
            demo_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in demo_grid] # (n*3)，体素索引
            predict_labels = self.my_model(demo_pt_fea_ten, demo_grid_ten, 2, val = True)
            predict_labels = torch.argmax(predict_labels, dim=1)

            predict_labels = predict_labels.cpu().detach().numpy()
            
            count = 0 # 单回波
            inv_labels0 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            count = 1
            inv_labels1 = np.vectorize(self.inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]).astype('uint32')
            sem_points1 = np.hstack ((scan[0],inv_labels0.reshape(-1,1)))
            sem_points2 = np.hstack ((scan[1],inv_labels1.reshape(-1,1)))
            sem_points = np.array([sem_points1,sem_points2])
            # finding the difference between
            # neighboring elements
        return sem_points

    
    
if __name__ == '__main__':
    # Training settings
    homepath = "/home/oliver/catkin_ros2/src/cylinder3d_ros2/cylinder3d_ros2/"
    datapath = "/media/oliver/Elements SE/dataset/KITTI/sequences/00/velodyne/"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default=homepath + 'submodules/config/semantickitti8_run.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    a = points_semantic(homepath,args)
    filename = datapath + '000000.bin'
    scan = np.fromfile(filename, dtype=np.float32)
    scan1 = scan.reshape((-1, 4))
    [demo_grid0, demo_pt_fea0] = a.data_preprocesser.preprocess(scan1)
    filename = datapath + '000001.bin'
    scan = np.fromfile(filename, dtype=np.float32)
    scan2 = scan.reshape((-1, 4))
    [demo_grid1, demo_pt_fea1] = a.data_preprocesser.preprocess(scan2)
    demo_grid = [demo_grid0 , demo_grid0]
    demo_pt_fea = [demo_pt_fea0, demo_pt_fea0]

    print("sem1*****************")
    for i in range(1,10):
        afsem_points = a.sem_points([demo_grid0], [demo_pt_fea0], np.array([scan1]))
        afsem_points = a.sem_points([demo_grid0], [demo_pt_fea0], np.array([scan1]))
    print("sem3*****************")
    for i in range(1,5):
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))
        afsem_points = a.sem_points2(demo_grid, demo_pt_fea, np.array([scan1, scan1]))


