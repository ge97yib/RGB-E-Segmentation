import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist

from .logger import get_logger
from utils.pyt_utils import load_model, parse_devices, extant_file, link_file, ensure_dir

logger = get_logger()

class State(object):
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['epoch', 'iteration', 'dataloader', 'model',
                         'optimizer']
            setattr(self, k, v)
            

class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
        
        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            os.environ['MASTER_PORT'] = self.args.port
            dist.init_process_group(backend="nccl", world_size=self.world_size, init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            self.devices = parse_devices(self.args.devices)


    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='0',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', default=0, type=int,
                       help='process rank on node')
        p.add_argument('-p', '--port', type=str,
                       default='16005',
                       dest="port",
                       help='port for init_process_group')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    # def save_checkpoint(self, path):
    #     logger.info("Saving checkpoint to file {}".format(path))
    #     t_start = time.time()
    #
    #     state_dict = {}
    #
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in self.state.model.state_dict().items():
    #         key = k
    #         if k.split('.')[0] == 'module':
    #             key = k[7:]
    #         new_state_dict[key] = v
    #     state_dict['model'] = new_state_dict
    #     state_dict['optimizer'] = self.state.optimizer.state_dict()
    #     state_dict['epoch'] = self.state.epoch
    #     state_dict['iteration'] = self.state.iteration
    #
    #     t_iobegin = time.time()
    #     torch.save(state_dict, path)
    #     del state_dict
    #     del new_state_dict
    #     t_end = time.time()
    #     logger.info(
    #         "Save checkpoint to file {}, "
    #         "Time usage:\n\tprepare checkpoint: {}, IO: {}".format(
    #             path, t_iobegin - t_start, t_end - t_iobegin))
    # def cleanup_state_dict(self, state_dict):
    #     """
    #     Clean up the keys of the state dictionary from a model, particularly
    #     necessary if the model is wrapped in nn.DataParallel or nn.DistributedDataParallel.
    #
    #     Args:
    #         state_dict (dict): The state dictionary of the model to clean.
    #
    #     Returns:
    #         OrderedDict: The cleaned state dictionary.
    #     """
    #     from collections import OrderedDict
    #     cleaned_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # Remove the 'module.' prefix if present (common in models wrapped in DataParallel)
    #         key = k[7:] if k.startswith('module.') else k
    #         cleaned_state_dict[key] = v
    #     return cleaned_state_dict

    def save_checkpoint(self, path):
        """
        Saves the current state of the training including model parameters,
        optimizer state, and current epoch into a checkpoint file.

        Args:
            path (str): The file path to save the checkpoint.
        """
        logger.info(f"Saving checkpoint to file {path}")
        t_start = time.time()

        # Prepare the model state dictionary for saving.
        state_dict = {
            'epoch': self.state.epoch,
            'iteration': self.state.iteration,
            'optimizer': self.state.optimizer.state_dict(),
            'model': self.cleanup_state_dict(self.state.model.state_dict())
        }

        # Save the checkpoint to the filesystem.
        try:
            torch.save(state_dict, path)
            logger.info("Checkpoint saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

        t_end = time.time()
        logger.info(f"Checkpoint IO time: {t_end - t_start:.2f} seconds")

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    # def save_and_link_checkpoint(self, checkpoint_dir, log_dir, log_dir_link):
    #     ensure_dir(checkpoint_dir)
    #     if not osp.exists(log_dir_link):
    #         link_file(log_dir, log_dir_link)
    #     current_epoch_checkpoint = osp.join(checkpoint_dir, 'epoch-{}.pth'.format(
    #         self.state.epoch))
    #     self.save_checkpoint(current_epoch_checkpoint)
    #     last_epoch_checkpoint = osp.join(checkpoint_dir, 'epoch-last.pth')
    #     link_file(current_epoch_checkpoint, last_epoch_checkpoint)
    def save_and_link_checkpoint(self, checkpoint_dir, log_dir, log_dir_link, filename='checkpoint.pth'):
        path = os.path.join(checkpoint_dir, filename)
        logger.info("Saving checkpoint to file {}".format(path))

        # Ensure the checkpoint and log directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir_link, exist_ok=True)

        t_start = time.time()

        state_dict = {
            'model': {k.replace('module.', ''): v for k, v in self.state.model.state_dict().items()},
            'optimizer': self.state.optimizer.state_dict(),
            'epoch': self.state.epoch,
            'iteration': self.state.iteration
        }

        torch.save(state_dict, path)
        t_iobegin = time.time()
        # Creating a symbolic link to the latest checkpoint for easy access
        link_path = os.path.join(log_dir_link, 'latest_checkpoint.pth.tar')
        if os.path.islink(link_path):
            os.unlink(link_path)
        os.symlink(path, link_path)

        t_end = time.time()
        logger.info(
            "Checkpoint saved and linked. Time usage: prepare checkpoint: {}, IO: {}".format(t_iobegin - t_start,
                                                                                             t_end - t_iobegin))

    # def restore_checkpoint(self):
    #     t_start = time.time()
    #     if self.distributed:
    #         # load the model on cpu first to avoid GPU RAM surge
    #         # when loading a model checkpoint
    #         # tmp = torch.load(self.continue_state_object,
    #         #                  map_location=lambda storage, loc: storage.cuda(
    #         #                      self.local_rank))
    #         tmp = torch.load(self.continue_state_object, map_location=torch.device('cpu'))
    #     else:
    #         tmp = torch.load(self.continue_state_object)
    #     t_ioend = time.time()
    #     self.state.model = load_model(self.state.model, tmp['model'], is_restore=True)
    #     self.state.optimizer.load_state_dict(tmp['optimizer'])
    #     self.state.epoch = tmp['epoch'] + 1
    #     self.state.iteration = tmp['iteration']
    #     del tmp
    #     t_end = time.time()
    #     logger.info(
    #         "Load checkpoint from file {}, "
    #         "Time usage:\n\tIO: {}, restore checkpoint: {}".format(
    #             self.continue_state_object, t_ioend - t_start, t_end - t_ioend))
    def restore_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.state.checkpoint_dir, 'latest_checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            logger.info(f"Restoring checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.state.model.load_state_dict(checkpoint['model'])
            self.state.optimizer.load_state_dict(checkpoint['optimizer'])
            self.state.epoch = checkpoint['epoch']
            self.state.iteration = checkpoint['iteration']
            logger.info(f"Checkpoint restored successfully from epoch {self.state.epoch}.")
        else:
            logger.info("No checkpoint found. Starting training from scratch.")

    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
