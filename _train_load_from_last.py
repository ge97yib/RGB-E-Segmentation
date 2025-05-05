import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config_deliver import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

import wandb  # Import wandb for tracking

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add argument for resuming from a specific checkpoint
parser.add_argument('--resume', type=str, default=None, help="Path to a specific checkpoint to resume training from")

##### Load Checkpoint Logic #####
def load_checkpoint(checkpoint_path, model, optimizer):
    if osp.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint keys: {checkpoint.keys()}")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("The checkpoint does not contain 'state_dict' or 'model' key.")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            logger.warning("The checkpoint does not contain 'optimizer' key.")
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return start_epoch
    else:
        logger.info(f"No checkpoint found at '{checkpoint_path}'")
        return 0

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(config.log_dir)   

    # Initialize wandb before training begins
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        wandb.init(project="segmentation-project", config=config)  # Initialize wandb project with config

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    print(f"Train Datset: {config.dataset_name}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        # engine.link_tb(tb_dir, generate_tb_dir)
        if not osp.exists(generate_tb_dir):
            engine.link_tb(tb_dir, generate_tb_dir)
        else:
            logger.warning(f"Symbolic link '{generate_tb_dir}' already exists.")

    # Config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # Group weight and config optimizer
    base_lr = config.lr
    params_list = group_weight([], model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # Load checkpoint if resume flag is set
    if args.resume:
        checkpoint_path = args.resume
    else:
        checkpoint_path = None

    if checkpoint_path:
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        start_epoch = 0

    # Config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin training:')

    # Track the last 10 checkpoints
    checkpoint_list = []

    for epoch in range(start_epoch + 1, config.nepochs + 1):  # Start from the loaded epoch
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            # Load minibatch
            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # Reduce loss for multi-GPU
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = f'Epoch {epoch}/{config.nepochs} Iter {idx + 1}/{config.niters_per_epoch}: lr={lr:.4e} loss={reduce_loss.item():.4f} total_loss={sum_loss / (idx + 1):.4f}'
            else:
                sum_loss += loss.item()
                print_str = f'Epoch {epoch}/{config.nepochs} Iter {idx + 1}/{config.niters_per_epoch}: lr={lr:.4e} loss={loss.item():.4f} total_loss={sum_loss / (idx + 1):.4f}'

            # Log the loss and learning rate to wandb
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                wandb.log({"train_loss": sum_loss / (idx + 1), "learning_rate": lr})

            del loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        # Save the checkpoint
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            checkpoint_filename = f"checkpoint_{epoch}.pth"
            if engine.distributed and (engine.local_rank == 0):
                logger.info(f"Saving checkpoint: {checkpoint_filename}")
                engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link, filename=checkpoint_filename)
            elif not engine.distributed:
                logger.info(f"Saving checkpoint: {checkpoint_filename}")
                engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link, filename=checkpoint_filename)

            # Track the checkpoint
            checkpoint_list.append(checkpoint_filename)

            # Remove older checkpoints to keep only the last 10
            if len(checkpoint_list) > 5:
                old_checkpoint = checkpoint_list.pop(0)
                old_checkpoint_path = osp.join(config.checkpoint_dir, old_checkpoint)
                if osp.isfile(old_checkpoint_path):
                    os.remove(old_checkpoint_path)
                    logger.info(f"Removed old checkpoint: {old_checkpoint_path}")

    # Finalize the wandb run after training
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        wandb.finish()