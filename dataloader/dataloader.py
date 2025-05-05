import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config_deliver import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, rgb, gt, modal_x):
        #print('rgb0:', rgb.shape)
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)

        if config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, config.train_scale_array)
        #print('rgb1:', rgb.shape)
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        #print('rgb2:', rgb.shape)

        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        if config.resize == True: #if we just resize the image for training
            # Compute the resized size
            re_size = (int(config.image_height * config.resize_scale), int(config.image_width * config.resize_scale))  # (width, height)

            # Downsample each image
            p_rgb = cv2.resize(rgb, re_size, interpolation=cv2.INTER_LINEAR)
            p_gt = cv2.resize(gt, re_size, interpolation=cv2.INTER_NEAREST)  # Nearest neighbor for labels
            p_modal_x = cv2.resize(modal_x, re_size, interpolation=cv2.INTER_LINEAR)

            # Transpose dimensions for `p_rgb` and `p_modal_x`
            p_rgb = p_rgb.transpose(2, 0, 1)  # (C, H, W)
            p_modal_x = p_modal_x.transpose(2, 0, 1)  # (C, H, W)
        else:
            crop_size = (config.image_height, config.image_width)
            crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

            p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
            p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

            p_rgb = p_rgb.transpose(2, 0, 1)
            p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x

class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x

def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    #'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler