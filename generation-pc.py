import argparse
import os
import time
import math
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.utils.data
from dataloaders.kitti_loader_d import load_calib, oheight, owidth, input_options, KittiDepth, get_paths_and_transform
import modelresnetunet as modelnet
from modelresnetunet import DepthCompletionNet
import sys

from matplotlib import pyplot
import cv2
from tensorboardX import SummaryWriter
from liteflownet.run import Network
import flowtoimage as fl
import matplotlib.pyplot as plt
import numpy as np
import flow_util
from dataloaders.pc_loader import project_disp_to_depth
# import pptk
import visz_point
from PIL import Image
from dataloaders.kitti_loader_d import load_calib, oheight, owidth, input_options, KittiDepth, get_paths_and_transform

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--epochsample', default=40000, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2',
#                     choices=criteria.loss_names,
#                     help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-i','--input', type=str, default='gd',
                    choices=input_options, help='input: | '.join(input_options))
parser.add_argument('-l','--layers', type=int, default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained', action="store_true",
                    help='use ImageNet pre-trained weights')
# parser.add_argument('--val', type=str, default="select",
#                     choices= ["select","full"], help='full or select validation set')
parser.add_argument('--val', type=str, default="full",
                    choices= ["select","full"], help='full or select validation set')
parser.add_argument('--jitter', type=float, default=0.1,
                    help = 'color jitter for images')

parser.add_argument('-m', '--train-mode', type=str, default="dense",
                    choices = ["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
                    help = 'dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('.', 'results_experiment_all')
# args.use_rgb = ('rgb' in args.input) or args.use_pose

args.use_rgb = True
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
# print(args)

K ,C2V,R0=load_calib()
fu, fv = float(K[0,0]), float(K[1,1])
cu, cv = float(K[0,2]), float(K[1,2])


# read .bin file
# def load_pc(filename):
#     scan = np.fromfile(filename,dtype= np.float32)
#     scan =scan.reshape(-1,4)
#     return scan

def save_pc(mode, args, loader):
    if not os.path.isdir("pc_data/"+mode+'_d'):
        os.makedirs("pc_data/"+mode+'_d')
    # model.train()
    # lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    print('len(loader)=',len(loader))
    # input("lhj")
    step = 0
    for i, batch_data in enumerate(loader):
        start = time.time()
        # batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
        batch_data = {key:val.to(device) for key,val in batch_data.items() if val is not None}
        if len(loader)-step <4:
            break
        if step == args.epochsample:
            break 

        pcfile0= 'pc_data/'+mode+'_d/'+str(i)+'_0.npy'
        pcfile1= 'pc_data/'+mode+'_d/'+str(i)+'_1.npy'
        pcfile2= 'pc_data/'+mode+'_d/'+str(i)+'_2.npy'
        with torch.no_grad():
            d0 = np.squeeze(batch_data['d'][0,...].data.cpu().numpy())
            d1 = np.squeeze(batch_data['d2'][0,...].data.cpu().numpy())
            d2 = np.squeeze(batch_data['d3'][0,...].data.cpu().numpy())
            point0 = project_disp_to_depth(d0,d0,fu,fv,cu,cv)
            point1 = project_disp_to_depth(d1,d1,fu,fv,cu,cv)
            point2 = project_disp_to_depth(d2,d2,fu,fv,cu,cv)
            pc0=np.array(point0,dtype=np.float32)
            pc1=np.array(point1,dtype=np.float32)
            pc2=np.array(point2,dtype=np.float32)
            np.save(pcfile0,pc0)
            np.save(pcfile1,pc1)
            np.save(pcfile2,pc2)
        print('loading '+mode+' step:',i)
        step=step+1

def main():
    print("KittiDepth('train', args)")
    train_dataset = KittiDepth('train', args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True, sampler=None)
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True) # set batch size to be 1 for validation
    print("=> data loaders created.")
    print(len(train_loader))
  
    save_pc("val", args, val_loader)
    save_pc("train", args, train_loader)
    print('generation ok !!!')

if __name__ == '__main__':
    main()