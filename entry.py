# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import torch
import logging
import wandb
import numpy as np
import random

from utils.arguments import load_opt_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args=None):
    '''
    [Main function for the entry point]
    1. Set environment variables for distributed training.
    2. Load the config file and set up the trainer.
    '''
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)
    
    
    opt, cmdline_args = load_opt_command(args)
    command = cmdline_args.command

    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    # update_opt(opt, command)
    world_size = 1
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    if opt['TRAINER'] == 'xdecoder':
        from trainer import XDecoder_Trainer as Trainer
    else:
        assert False, "The trainer type: {} is not defined!".format(opt['TRAINER'])
    
    trainer = Trainer(opt)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    if command == "train":
        if opt['rank'] == 0 and opt['WANDB']:
            wandb.login(key=os.environ['WANDB_KEY'])
            init_wandb(opt, trainer.save_folder, job_name=trainer.save_folder)
        trainer.train()
    elif command == "evaluate":
        trainer.eval()
    elif command == "histpt": #add for histpt
        trainer.train(method="histpt")
    else:
        raise ValueError(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
    sys.exit(0)
