import numpy as np
import os
import random
import wandb

import torch
import argparse
import logging

from train import fit
from datasets import create_dataset, create_dataloader
from models import *
from log import setup_default_logging

from accelerate import Accelerator

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(args):

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = args.grad_accum_steps,
        mixed_precision             = args.mixed_precision
    )

    setup_default_logging()
    torch_seed(args.seed)

    # set device
    '''
    # Setting the right device

    The Accelerator class knows the right device to move any PyTorch object to at any time, 
    so you should change the definition of device to come from Accelerator:

    # change device setting
    - device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    + device = accelerator.device # this is not neccesary
    '''
    _logger.info('Device: {}'.format(accelerator.device))

    # build Model
    model = ResNet50()
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset, testset = create_dataset(datadir=args.datadir, aug_name=args.aug_name)
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=256, shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, t_mult=1, eta_min=0.00001)
    
    # prepraring accelerator
    '''
    # Preparing your objects
    
    Next you need to pass all of the important objects related to training into prepare(). 
    🤗 Accelerate will make sure everything is setup in the current environment for you to start training:
    
    '''
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )

    # make save directory
    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    # load checkpoints
    if args.ckpdir:
        accelerator.load_state(args.ckpdir)

    # initialize wandb
    if args.use_wandb:
        wandb.init(name=args.exp_name, project='Accelerator Test', config=args)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        accelerator  = accelerator,
        epochs       = args.epochs, 
        savedir      = savedir,
        use_wandb    = args.use_wandb,
        log_interval = args.log_interval)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Accelerator Test")
    # exp setting
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')

    # optimizer
    parser.add_argument('--opt-name',type=str,choices=['SGD','Adam'],help='optimizer name')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')

    # augmentation
    parser.add_argument('--aug-name',type=str,choices=['default','weak','strong'],help='augmentation type')

    # train
    parser.add_argument('--epochs',type=int,default=50,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--grad-accum-steps',type=int,default=1,help='gradients accumulation steps')
    parser.add_argument('--mixed-precision',type=str,default=None,choices=[None,'fp16','bf16'],help='choice mixed precision')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')
    parser.add_argument('--ckpdir',type=str,default=None,help='checkpoint directory')

    # seed
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    # wandb
    parser.add_argument('--use-wandb',action='store_true',help='use wandb')

    args = parser.parse_args()

    run(args)