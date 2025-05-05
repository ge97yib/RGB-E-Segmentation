18.05
edit directory: new_ddd17 from ddd17_seg
rgb_images belongs to folder 'images', named 'others' before
label_folder belongs to folder 'labels', named 'others' before
in order to match the pictures in events_images folder

22.05
train:
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 _train_load_from_last.py   #--devices 0
instead of 
#CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py

edit image reading function: _open_image in RGBXDataset.py line 44
bgr_img = cv2.imread(rgb_path)
rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB) read img as grayscale img
it causes error img-mean when we do the normalization

val: 
CUDA_VISIBLE_DEVICES=0 python eval.py --epochs 60

23.05
To do: edit num_class, class_names in config.py

30.05
crossing line can not be well seperated.
use DSEC/allignment.py to process the rgb image left

05,06
dsec events=8074, rgb=8082, labels=8082
Deleting zurich_city_01_a_000680.png
Deleting zurich_city_00_a_000938.png
Deleting zurich_city_02_a_000234.png
Deleting zurich_city_07_a_001462.png
Deleting zurich_city_06_a_001522.png
Deleting zurich_city_05_a_001752.png
Deleting zurich_city_04_a_000700.png
Deleting zurich_city_08_a_000786.png
Deleting zurich_city_01_a_000680.png
Deleting zurich_city_00_a_000938.png
Deleting zurich_city_02_a_000234.png
Deleting zurich_city_07_a_001462.png
Deleting zurich_city_06_a_001522.png
Deleting zurich_city_05_a_001752.png
Deleting zurich_city_04_a_000700.png
Deleting zurich_city_08_a_000786.png

08.06
change the class EventReader in /home/tang/code/others/RGBX_Semantic_Segmentation/DSEC/scripts/visualization/eventreader.py: make sure the last part will be generated as an image even through less than time threshold
To handle the scenario where the remaining data in the .h5 file is not perfectly divisible by delta_time_ms (50 milliseconds)

convert h5 to png: 
python DSEC/scripts/convert_h5_to_png_rec.py /mnt/hdd/my/dsec/train_event/zurich_city_08_a/events/left/events.h5 /mnt/hdd/my/dsec/train_event/zurich_city_08_a/events/left/rectify_map.h5 /mnt/hdd/my/dsec/train_event/zurich_city_08_a/events_image


08.07
TODO: use mit_b2.pth if possible
modify the cmx architecture with Adain module(skip CM-FRM part only replace FFM part with Adain module)
run the cmnext baseline

10.07
dual_segformer line320-330, 376,393, 410，427 comment out the code(with self.FRMs)
net_utils line152,153 comment out the code:
        # B, N, _C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous

13.07
/eigine/evaluator.py line78: replace "epoch-s%.pth.ter" with "checkpoint_%s.pth"

16.07
EventScape:
There are 49591 PNG files in the directory /mnt/hdd/my/EventScape/train/Town01.
There are 56080 PNG files in the directory /mnt/hdd/my/EventScape/train/Town03.
There are 16658 PNG files in the directory /mnt/hdd/my/EventScape/train/Town02.
sum: 122329
There are 22493 PNG files in the directory /mnt/hdd/my/EventScape/val/Town05.
There are 26183 PNG files in the directory /mnt/hdd/my/EventScape/test/Town05.

17.07
check train.txt/test.txt finished
start trainning EventScape with CAFR: 5h/epoch

18.08
DELIER/depth
There are 1574 PNG files in the directory /mnt/hdd/my/deliver/DELIVER/depth/fog.
There are 1571 PNG files in the directory /mnt/hdd/my/deliver/DELIVER/depth/cloud.
There are 1586 PNG files in the directory /mnt/hdd/my/deliver/DELIVER/depth/night.
There are 1577 PNG files in the directory /mnt/hdd/my/deliver/DELIVER/depth/sun.
There are 1577 PNG files in the directory /mnt/hdd/my/deliver/DELIVER/depth/rain.

delete the val picture for event and semantic by mistake....now only 3380 pictures left (test+train)    

22.07
generate config.py for NYUdepthV2
modify the train.txt and test.txt

31.07
check the function: get_class_colors(config.class_names, config.num_classes)

11.08
using _train instead of train
check the class number

20.08
added new script: _train_load_from_last.py
line_32: added checkpoint_path for resuming training at a certain checkpoint

22.08
--resume argument in _train_load_from_last.py for resuming training at a certain checkpoint

01,09
builder.py encoder_decoder: line 106-116, assert rgb.shape == modal_x.shape

07,09
data processing with SUNRGBD dataset
input_size: 474,477 or 0.8
RGBXDataset.py line 36, line 38: add .strip()
line 39: add .split()
print rgb and x size: /home/tang/code/others/RGBX_Semantic_Segmentation/models/builder.py line 106-108
eval.py line 36-38 only for SUNRGBD

23.09
add the EISNet modules under the folder: /home/tang/code/others/RGBX_Semantic_Segmentation/models, named eisnet.py
TODO: modify trainer.py to train the EISNet

06.10
add AEIM modules under the folder: /home/tang/code/others/RGBX_Semantic_Segmentation/models/net_utils.py, starting from line 226
modify dual_segformer.py to use AEIM: line 235--add the aeim
line 379--use AEIM
line 560-- add aet=True
use EISNet to process the event datac

20.10
modify the ddd17 dataset to our rgbx data format
start training ddd17 with CAFR

27.10
add the linear attention
train: DSEC, MFNet
!! last class always be nan, C.gt_transform should be False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)

01.11
TODO: 
train dsec with CAFR but C.gt_transform = False
train ddd17seg with CAFR but C.gt_transform = False 346*200
train ddd17seg with CAFR+linear attention but C.gt_transform = False
try to record the training time

# DDD17 346*200
C.num_train_imgs = 15950 
C.num_eval_imgs = 3890 

# MFNet 640*480  scale=0.9
C.num_train_imgs = 784 
C.num_eval_imgs = 392

# DSEC 640*440
C.num_train_imgs = 8082 
C.num_eval_imgs = 2809 

# DELIVER 1042*1042  scale=0.5
C.num_train_imgs = 3893
C.num_eval_imgs = 1897

# NYUDepth 640*480  scale=0.9
C.num_train_imgs = 795
C.num_eval_imgs = 654

# SUN_RGBD 730*530  scale=0.8
C.num_train_imgs = 5285
C.num_eval_imgs = 5050

03.11
add the FLOPs and params calculation module

TODO:
train NYUDepth with CAFR+linear attention but C.gt_transform = True 640*480 
train SUNRGBD with CAFR+linear attention but C.gt_transform = True 730*530
train DELIVER with CAFR+linear attention but try C.gt_transform = False 1042*1042

05.11
modify the _train_load_from_last.py line14 # from config_nyudepth import config
modify the eval.py line10 # from config_nyudepth import config
modify the dataloader.py line6 #from config_nyudepth import config
use the different config file for different dataset

14.11
add mi loss
add downsampling part for loading images

20.11
dsec from 40epoch:training error, mi_loss set to 0, forgot to change 
deliver classes error: training with half iamge size but eval with original image size: using resize instead of crop
solved!

23.11
for Deliver dataset
use the different gt read method
modify RGBXDataset.py line 59-62
