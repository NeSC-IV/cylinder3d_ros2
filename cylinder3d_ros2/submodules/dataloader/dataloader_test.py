# -*- coding:utf-8 -*-
# author: shashenyiguang
# @file: dataloader_test.py 

import numpy as np


class SemKITTI_demo():
    def __init__(self, grid_size, ignore_label=255,
                    fixed_volume_space=False, max_volume_space=[50, np.pi, 2], 
                    min_volume_space=[0, -np.pi, -4]):
            self.grid_size = np.asarray(grid_size)
            self.ignore_label = ignore_label
            self.fixed_volume_space = fixed_volume_space
            self.max_volume_space = max_volume_space
            self.min_volume_space = min_volume_space
    def preprocess(self,raw_data):
        annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1) # 生成一个全0的列向量n*1
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8)) # ([x,y,z], [0])
        data_tuple += (raw_data[:, 3],) # ([x,y,z], [0], intensity)
        data = data_tuple
        xyz, labels, sig = data
        if len(sig.shape) == 2: sig = np.squeeze(sig) # intensity(1*n)
        xyz_pol = cart2polar(xyz) # (n*3)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0) # 100%的点的半径
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0) # 0%的点的半径
        max_bound = np.max(xyz_pol[:, 1:], axis=0) # 最大的极坐标[theta, h]
        min_bound = np.min(xyz_pol[:, 1:], axis=0) # 最小的极坐标[theta, h]
        max_bound = np.concatenate(([max_bound_r], max_bound)) # [max_bound_r, max_bound_theta, max_bound_h]
        min_bound = np.concatenate(([min_bound_r], min_bound)) # [min_bound_r, min_bound_theta, min_bound_h]

        if self.fixed_volume_space: # 固定的体素空间
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound # 体素范围
        cur_grid_size = self.grid_size # 体素的分辨率
        intervals = crop_range / (cur_grid_size - 1) # 体素的间隔

        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int) # 体素的索引，超出范围的点会被截断

        # voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        # dim_array = np.ones(len(self.grid_size) + 1, int)
        # dim_array[0] = -1
        # voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)

        # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        # label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        # label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        # data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound # 体素中心点的极坐标
        return_xyz = xyz_pol - voxel_centers # 体素中心点的极坐标减去点的极坐标，得到点相对于体素中心的极坐标
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1) # (n*8)，点对于体素中心极坐标（r,\theta,z）,极坐标（r,\theta,z）,笛卡尔坐标（x,y）

        return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1) # (n*9), + intensity
        # data_tuple += (grid_ind, labels, return_fea)

        # return collate_fn_BEV(data_tuple)
        return grid_ind, return_fea

def collate_fn_BEV(data):
    # data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    # label2stack = np.stack([d[1] for d in data]).astype(np.int)
    # grid_ind_stack = [d[2] for d in data]
    grid_ind_stack = [data[2]]
    # point_label = [d[3] for d in data]
    # xyz = [d[4] for d in data]
    xyz = [data[4]]
    return grid_ind_stack, xyz

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2) #半径1*n
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1) #极坐标（r,\theta,z）

def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label