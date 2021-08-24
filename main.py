import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from torch.utils.data import DataLoader
from data_manager import EtiDataset
from eti_dataset import ETI
import argparse
import sys
from torch import nn
import torch
from torch.optim import lr_scheduler
from train import BaseTrainer
import time
import os.path as osp
import numpy as np
import datetime
from utils import save_checkpoint, mkdir_if_missing, Logger
from models.resnet3d import r2plus1d_18, r3d_18, mc3_18

parser = argparse.ArgumentParser(description='Training')
# Datasets
parser.add_argument('--root', type=str, default='/home/paul/eti/dataset')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=128)
parser.add_argument('--width', type=int, default=171)
# Augment
parser.add_argument('--seq_len', type=int, default=16,
                    help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8,
                    help="stride of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--train_batch', default=32, type=int)
parser.add_argument('--test_batch', default=32, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--stepsize', default=[20, 40, 60], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float)
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--distance', type=str, default='cosine',
                    help="euclidean or cosine")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet3d',
                    help="resnet3d, lstm")
# Miscs
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--eval_step', type=int, default=10)
parser.add_argument('--start_eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='logs')


if __name__ == "__main__":
    args = parser.parse_args()
    log_name = f"train_{args.arch}_{time.strftime('-%Y-%m-%d-%H-%M-%S')}.log"
    #save_dir = osp.join(args.save_dir, log_name)
    mkdir_if_missing(args.save_dir)
    sys.stdout = Logger(osp.join(args.save_dir, log_name))

    spatial_transform_train = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        #ST.RandomHorizontalFlip(),
        #ST.CenterCrop(256),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    temporal_transform_train = TT.TemporalRandomCrop(size=args.seq_len, stride=args.sample_stride)
    dataset = ETI(root=args.root)
    train_loader = DataLoader(
        EtiDataset(dataset.train,
                   spatial_transform=spatial_transform_train,
                   temporal_transform=temporal_transform_train),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=True, drop_last=True)

    clip, score = (next(iter(train_loader)))
    print(score.shape)
    print(clip.shape)
    trainer = BaseTrainer()

    print("Initializing model: {}".format(args.arch))
    # model = r2plus1d_18(pretrained=False, num_classes=1)
    model = r3d_18(pretrained=False, num_classes=1)
    # model = mc3_18(pretrained=False, num_classes=1)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    start_time = time.time()
    train_time = 0
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        trainer.train(epoch, model, criterion=criterion, optimizer=optimizer,
                      train_loader=train_loader)
        train_time += round(time.time() - start_train_time)

        save_checkpoint({
            'state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'epoch': epoch,
        }, is_best=False, fpath=osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


