import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = '/mnt/hdd/my/NYUDepth' #os.path.abspath(os.path.join(os.getcwd(), './'))

C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDepthv2' #'evscape4rgbx'
C.dataset_path = osp.join(C.root_dir, C.dataset_name)
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')  #rgb
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')  #label nyudepthv2:dont use colored label
C.gt_format = '.png'
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?), dsec, ddd17, deliver, sunrgbd
C.x_root_folder = osp.join(C.dataset_path, 'HHA') #event:polL_dolp_img
C.x_format = '.jpg'
C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input, if HHA, False
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 795
C.num_eval_imgs = 654 

# DDD17
# C.num_classes = 6
# C.class_names = ['flat', 'background', 'object', 'vegetation', 'human', 'vehicle']
# EventScape
# C.num_classes = 12
# C.class_names = ['Vehicle', 'Building', 'Wall', 'Vegetation', 'Road', 'Pole','RoadLines', 'Fences', 'Pedestrian', 'TrafficSign','Sidewalk','TrafficLight']

# DSEC11
# C.num_classes = 11
# C.class_names = ['background', 'building', 'fence', 'person', 'pole', 'road', 'sidewalk', 'vegetation', 'car', 'wall', 'trafficsign']
# DSEC19
# C.num_classes = 19
# C.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# DELIVER
# C.num_classes = 25
# C.class_names = ['Building', 'Fence', 'Other', 'Pedestrian', 'Pole', 'Roadline', 'Road', 'Sidewalk', 'Vegetation', 'Cars', 'Wall', 'TrafficSign', 'Sky', 'Ground', 'Bridge', 'RailTrack', 'GroundRail', 'TrafficLight', 'Static', 'Dynamic', 'Water', 'Terrain', 'TwelveWheel', 'Bus', 'Truck']

# NYUDepthv2
C.num_classes = 40
C.class_names = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

# MFNet
# C.num_classes = 9
# C.class_names = ['unlabelled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']

# SUNRGBD
# C.num_classes = 37
# C.class_names = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair',
#                             'sofa', 'table', 'door', 'window', 'bookshelf',
#                            'picture', 'counter', 'blinds', 'desk', 'shelves',
#                            'curtain', 'dresser', 'pillow', 'mirror',
#                            'floor mat', 'clothes', 'ceiling', 'books',
#                            'fridge', 'tv', 'paper', 'towel', 'shower curtain',
#                            'box', 'whiteboard', 'person', 'night stand',
#                            'toilet', 'sink', 'lamp', 'bathtub', 'bag']

"""Image Config"""
C.background = 255
C.image_height = int(0.9*480)
C.image_width = int(0.9*640)
C.downsample = False # True for Deliver, MFNet, NYUdepth, False for others
C.downsample_scale = 0.9
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b4' # Remember change the path below.
C.model_path = 'pretrained'
C.pretrained_model = C.model_path + '/mit_b4.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5 #6e-5 for others, 2e-4 for ddd17
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 2 #8
C.nepochs = 60 #60 for others, 40 for ddd17, 500 for nyudepth
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers =2 #16
C.train_scale_array =[0.5, 0.75, 1, 1.25, 1.5, 1.75] # [0.75, 1, 1.25] for other datasets except SUNRGBD
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  #[0.75, 1, 1.25]#[0.5, 0.75, 1, 1.25, 1.5, 1.75] 
C.eval_flip = False
C.eval_crop_size = [480, 640] # [height, width]

"""Store Config"""
C.checkpoint_start_epoch = 0 #250
C.checkpoint_step = 2

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_' + C.dataset_name + '_hha_' + C.backbone + '_CAFR')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
