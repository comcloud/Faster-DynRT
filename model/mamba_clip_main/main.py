# Based on SLIP code bases
# https://github.com/facebookresearch/SLIP
# --------------------------------------------------------'
import argparse
import os

try:
    import wandb
except ImportError:
    wandb = None

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import model.mamba_clip_main.models as models
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='A-CLIP pre-training and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='yfcc15m', type=str, choices=['yfcc15m', 'cc3m', 'cc12m', 'coco', 'redcaps'])
    parser.add_argument('--metadata', default='yfcc15m.pkl', type=str,
                    help='path to metadata file (see README for details)')
    parser.add_argument('--root', default='', type=str,
                        help='path to dataset root')
    parser.add_argument('--output-dir', default='./', type=str, help='path where to save, empty for no saving')
    # Model
    parser.add_argument('--model', default='ACLIP_VITB16', type=str)
    parser.add_argument('--mask-ratio', default=0., type=float)
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # Training
    parser.add_argument('--momentum-ema', default=0.996, type=float, help="""Base EMA
    parameter. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=2, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--base-lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--descriptions', default='training', type=str)

    # For cluster
    parser.add_argument('--imagenet-val-zip', default='', type=str)
    parser.add_argument('--imagenet-val-txt', default='', type=str)
    return parser

def get_model(args):
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(mask_ratio=args.mask_ratio)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200,find_unused_parameters=False)

    return model



def test_sarcasm(args):
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    model = get_model(args)






if __name__ == '__main__':
    parser = argparse.ArgumentParser('A-CLIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    test_sarcasm(args)



