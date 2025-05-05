import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
from PIL import Image

from config_deliver import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img, get_class_colors
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

import torchprofile
import time

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']

        # for deliver dataset
        # img = cv2.resize(img, config.eval_crop_size, interpolation=cv2.INTER_LINEAR)
        # label = cv2.resize(label, config.eval_crop_size, interpolation=cv2.INTER_NEAREST)  # Nearest neighbor for labels
        # modal_x = cv2.resize(modal_x, config.eval_crop_size, interpolation=cv2.INTER_LINEAR)
        # end for deliver dataset
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')
            fn = name + '.png'
            # the following code is only for SUNRGBD dataset
            # pred_name = name.split(' ')
            # seg_name = pred_name[2].split('/')[-1]
            # fn = seg_name

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors(config.class_names, config.num_classes)
            #class_colors = get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            #logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)
        return result_line

def measure_inference_speed(network, device, warmup=10, test_runs=100):
    """
    测算网络的推理速度（FPS）

    :param network: 已经移动到 GPU 或 CPU 上的模型
    :param device: 计算设备（cuda 或 cpu）
    :param warmup: 预热次数，避免初次计算的额外开销对速度统计带来误差
    :param test_runs: 正式推理计数
    :return: 平均推理时间(秒/张)、FPS(帧每秒)
    """
    # 构造和 FLOPs 测试时相同大小的输入
    dummy_img = torch.randn(1, 3, 480, 640).to(device)
    dummy_modal_x = torch.randn(1, 3, 480, 640).to(device)

    # 1. 预热
    network.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = network(dummy_img, dummy_modal_x)
            # 如果是 GPU，需要做一次同步，保证预热阶段的计算全部完成
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # 2. 正式测算推理时间
    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.synchronize()  # 再次同步以准确记录起始时间
    
    with torch.no_grad():
        for _ in range(test_runs):
            _ = network(dummy_img, dummy_modal_x)
            # GPU 同步，确保每次推理都完成后再进行下一次
            if device.type == 'cuda':
                torch.cuda.synchronize()

    end_time = time.time()
    # 3. 统计结果
    total_time = end_time - start_time
    avg_time = total_time / test_runs  # 每张推理平均时间
    fps = 1.0 / avg_time if avg_time > 0 else float('inf')

    return avg_time, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default='output')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)

    # Load the checkpoint(new added)
    # print("eval_dir:", config.log_dir)
    checkpoint_dir = config.checkpoint_dir
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{args.epochs}.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0') #use gpu to load the model
        if 'state_dict' in checkpoint:
            network.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            network.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("The checkpoint does not contain 'state_dict' or 'model' key.")
        logger.info(f"Loaded checkpoint from '{checkpoint_path}'")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

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
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
    print("the eval size is:",config.eval_crop_size)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
        
    # Calculate and log FLOPs and total parameters after evaluation using real images
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # network.to(device)

    # dataset_name = config.dataset_name
    # # Create dummy inputs for img and modal_x with size 480x640
    # dummy_img = torch.randn(1, 3, 480, 640).to(device)  # (batch_size, channels, height, width)
    # dummy_modal_x = torch.randn(1, 3, 480, 640).to(device)  # Adjust shape as per your model requirements

    # dummy_img_2 = torch.randn(1, 3, 512, 512).to(device)  # (batch_size, channels, height, width)
    # dummy_modal_x_2 = torch.randn(1, 3, 512, 512).to(device)  # Adjust shape as per your model requirements

    # dummy_img_3 = torch.randn(1, 3, 200, 346).to(device)  # (batch_size, channels, height, width)
    # dummy_modal_x_3 = torch.randn(1, 3, 200, 346).to(device)  # Adjust shape as per your model requirements

    # # Calculate FLOPs considering both inputs
    # flops = torchprofile.profile_macs(network, (dummy_img, dummy_modal_x)) * 2  # Each MAC is 2 FLOPs
    # total_params = sum(p.numel() for p in network.parameters())
    # # Calculate FLOPs considering both inputs
    # flops_2 = torchprofile.profile_macs(network, (dummy_img_2, dummy_modal_x_2)) * 2  # Each MAC is 2 FLOPs
    # total_params_2 = sum(p.numel() for p in network.parameters())
    # # Calculate FLOPs considering both inputs
    # flops_3 = torchprofile.profile_macs(network, (dummy_img_3, dummy_modal_x_3)) * 2  # Each MAC is 2 FLOPs
    # total_params_3 = sum(p.numel() for p in network.parameters())


    # logger.info(f"Total FLOPs (input size 480x640): {flops}") #480x640
    # logger.info(f"Total Parameters: {total_params}")

    # # ========= 新增：测算模型推理速度(FPS) =========
    # avg_time, fps = measure_inference_speed(network, device, warmup=10, test_runs=100)
    # logger.info(f"Average inference time per frame: {avg_time} seconds")
    # logger.info(f"FPS: {fps}")

    # logger.info(f"Total FLOPs (input size 512x512): {flops_2}") #480x640
    # logger.info(f"Total Parameters: {total_params_2}")
    # # Write FLOPs and Params to val_log_file
    # with open(config.val_log_file, 'a') as log_file:
    #     log_file.write(f"\nDataset: '{dataset_name}'\n")
    #     log_file.write(f"\nInput size '480x640':\n") #480x640
    #     log_file.write(f"\nEvaluation on checkpoint '{args.epochs}':\n")
    #     log_file.write(f"Total FLOPs: {flops}\n")
    #     log_file.write(f"Total Parameters: {total_params}\n")
    #     log_file.write(f"Average inference time per frame: {avg_time} seconds\n")
    #     log_file.write(f"FPS: {fps}\n")

    #     log_file.write(f"\nInput size '512x512':\n") #512x512
    #     log_file.write(f"\nEvaluation on checkpoint '{args.epochs}':\n")
    #     log_file.write(f"Total FLOPs: {flops_2}\n")
    #     log_file.write(f"Total Parameters: {total_params_2}\n")

    #     log_file.write(f"\nInput size '200x346':\n") #200x346
    #     log_file.write(f"\nEvaluation on checkpoint '{args.epochs}':\n")
    #     log_file.write(f"Total FLOPs: {flops_3}\n")
    #     log_file.write(f"Total Parameters: {total_params_3}\n")
