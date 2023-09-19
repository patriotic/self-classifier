import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models

import copy
import vit as vits
import torch.nn as nn

from model import Model
import utils
from apex.parallel.LARC import LARC

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))
model_names = ['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'] + torchvision_archs

parser = argparse.ArgumentParser(description='PyTorch ImageNet Self-Supervised Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='linear warmup epochs (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=4.8, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--start-warmup', default=0.3, type=float,
                    help='initial warmup learning rate')
parser.add_argument('--final-lr', default=None, type=float,
                    help='final learning rate (None for constant learning rate)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--sgd', action='store_true',
                    help='use SGD optimizer')
parser.add_argument('--lars', action='store_true',
                    help='use LARS optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=16, type=int,
                    metavar='N', help='print frequency (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cls-size', type=int, default=[1000], nargs='+',
                    help='size of classification layer. can be a list if cls-size > 1')
parser.add_argument('--new-cls-size', type=int, default=[10], nargs='+',
                    help='size of new classification layer. can be a list if cls-size > 1')
parser.add_argument('--num-cls', default=1, type=int, metavar='NCLS',
                    help='number of classification layers')
parser.add_argument('--save-path', default='../saved/', type=str,
                    help='save path for checkpoints')
parser.add_argument('--pretrained', default=None, type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--rm-pretrained-cls', action='store_true',
                    help='ignore classifier when loading pretrained model (used for initializing imagenet subset)')
parser.add_argument('--queue-len', default=262144, type=int,
                    help='length of nearest neighbor queue')
parser.add_argument('--dim', default=128, type=int, metavar='DIM',
                    help='size of MLP embedding layer')
parser.add_argument('--hidden-dim', default=4096, type=int, metavar='HDIM',
                    help='size of MLP hidden layer')
parser.add_argument('--num-hidden', default=3, type=int,
                    help='number of MLP hidden layers')
parser.add_argument('--row-tau', default=0.1, type=float,
                    help='row softmax temperature (default: 0.1)')
parser.add_argument('--col-tau', default=0.05, type=float,
                    help='column softmax temperature (default: 0.05)')
parser.add_argument('--use-amp', action='store_true',
                    help='use automatic mixed precision')
parser.add_argument("--syncbn_process_group_size", default=0, type=int,
                    help='process group size for syncBN layer')
parser.add_argument('--use-lsf-env', action='store_true',
                    help='use LSF env variables')
parser.add_argument('--use-bn', action='store_true',
                    help='use batch normalization layers in MLP')
parser.add_argument('--fixed-cls', action='store_true',
                    help='use a fixed classifier')
parser.add_argument('--global-crops-scale', type=float, nargs='+', default=(0.4, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
                    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we 
                    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local-crops-number', type=int, default=6,
                    help="""Number of small local views to generate. 
                    Set this parameter to 0 to disable multi-crop training. 
                    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local-crops-scale', type=float, nargs='+', default=(0.05, 0.4),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image. 
                    Used for small local view cropping of multi-crop.""")
parser.add_argument('--patch-size', default=16, type=int,
                    help="""Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller 
                    values leads to better performance but requires more memory. 
                    Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling 
                    mixed precision training to avoid unstabilities.""")
parser.add_argument('--clip-grad', type=float, default=0.0,
                    help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can 
                    help optimization for larger ViT architectures. 0 for disabling.""")
parser.add_argument('--no-nn-aug', action='store_true',
                    help='do not use nearest neighbor augmentation')
parser.add_argument('--no-bias-wd', action='store_true',
                    help='do not regularize biases nor Norm parameters')
parser.add_argument('--bbone-wd', type=float, default=None,
                    help='backbone weight decay. if set to None weight_decay is used for backbone as well.')
parser.add_argument('--eps', type=float, default=1e-12,
                    help='small value to avoid division by zero and log(0)')
parser.add_argument('--subset-file', default=None, type=str,
                    help='path to imagenet subset txt file')
parser.add_argument('--no-leaky', action='store_true',
                    help='use regular relu layers instead of leaky relu in MLP')


def main():
    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch in vits.__dict__.keys():
        base_model = vits.__dict__[args.arch](patch_size=args.patch_size)
        backbone_dim = base_model.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        base_model = torchvision_models.__dict__[args.arch]()
        backbone_dim = base_model.fc.weight.shape[1]
    else:
        raise Exception("Unknown architecture: {}".format(args.arch))
    model = Model(base_model=base_model,
                  dim=args.dim,
                  hidden_dim=args.hidden_dim,
                  cls_size=args.cls_size,
                  num_cls=args.num_cls,
                  num_hidden=args.num_hidden,
                  use_bn=args.use_bn,
                  backbone_dim=backbone_dim,
                  fixed_cls=args.fixed_cls,
                  no_leaky=args.no_leaky)

    # print(model)
    mlp_head = copy.deepcopy(model.mlp_head)
    print(model.mlp_head.mlp[0].weight)

    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            # # remove module. prefix
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False)
            model.mlp_head = mlp_head
            print(model.mlp_head.mlp[0].weight)

            for cls_i in range(args.num_cls):
                cls_layer_i = nn.utils.weight_norm(nn.Linear(args.dim, args.new_cls_size[cls_i], bias=False))
                cls_layer_i.weight_g.data.fill_(1)
                setattr(model, "cls_%d" % cls_i, cls_layer_i)

                for param in getattr(model, "cls_%d" % cls_i).parameters():
                    param.requires_grad = False

            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained,
                                                                       checkpoint[
                                                                           'epoch'] if 'epoch' in checkpoint else 'NA'))

        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    params_groups = utils.get_params_groups(model, args)
    if args.sgd:
        optimizer = torch.optim.SGD(params_groups, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params_groups, args.lr,
                                      weight_decay=args.weight_decay)

    if args.lars:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    state = {'epoch': checkpoint['epoch'],
             'arch': args.arch,
             'state_dict': model.state_dict(),
             'best_loss': checkpoint['best_loss'],
             'nn_queue': checkpoint['nn_queue'],
             'optimizer': optimizer.state_dict()
             }
    torch.save(state, os.path.join(args.save_path, 'modified_model.pth.tar'))
    # print(model)


if __name__ == '__main__':
    main()
