# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_scatter
import time

class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim), # 标准化，使得每个维度的均值为0，方差为1，这可以提高网络的稳定性和性能

            nn.Linear(fea_dim, 64),  # n*9 -> n*64, y=x(n*9)A^T(9*64)+b(1*64)
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind, val = False):
        cur_dev = pt_fea[0].get_device() # get device id
        # forward1 = time.time()
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch)) # add batch index
        # forward2 = time.time()
        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]
        # forward3 = time.time()
        # shuffle the data
        if(not val):
            shuffled_ind = torch.randperm(pt_num, device=cur_dev)
            cat_pt_fea = cat_pt_fea[shuffled_ind, :]
            cat_pt_ind = cat_pt_ind[shuffled_ind, :]
        # forward4 = time.time()
        # unique xy grid index
        # return_inverse 是指定是否返回一个张量，表示输入张量中的元素在返回张量中的位置，默认为 False，return_counts 是指定是否返回一个张量，表示返回张量中的每个元素在输入张量中出现的次数，默认为 False，dim 是指定在哪个维度上进行唯一操作，默认为 None，表示对整个输入张量进行唯一操作。
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0) 
        unq = unq.type(torch.int64)
        # forward5 = time.time()
        # process feature
        # processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        # pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] # 每个体素只保留一个点特征作为体素特征
        cat_pt_fea = torch_scatter.scatter_max(cat_pt_fea, unq_inv, dim=0)[0] # 每个体素只保留一个点特征作为体素特征
        pooled_data = self.PPmodel(cat_pt_fea)
        # forward6 = time.time()
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data
        # forward7 = time.time()
        # print("total forward time:",forward7 - forward1)
        # print(forward2-forward1,forward3-forward2,forward4-forward3,forward5-forward4,forward6-forward5,forward7-forward6)
        return unq, processed_pooled_data
