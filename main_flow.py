import argparse
import os
import time
import math
import torch
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.utils.data

# from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth, get_paths_and_transform
from dataloaders.kitti_loader_d_flow import load_calib, oheight, owidth, input_options, KittiDepth, get_paths_and_transform
# from model import DepthCompletionNet
# from unet_refine import DepthCompletionNet
# import modelresnetunet as modelnet
# from modelresnetunet import DepthCompletionNet
import plin_model as modelnet
from plin_model import DepthCompletionNet
import sys
# sys.path.append('./model/')
# from baseline import DepthCompletionNet
# import baseline as modelnet

from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from
from matplotlib import pyplot
import cv2
from tensorboardX import SummaryWriter
from liteflownet.run import Network
import flowtoimage as fl
import matplotlib.pyplot as plt
import numpy as np
import flow_util
# from dataloaders.pc_loader import project_disp_to_depth
# import pptk
import visz_point
from PIL import Image
parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=16, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--epochsample', default=50000, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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
parser.add_argument('--rank-metric', type=str, default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help = 'metrics for which best result is sbatch_datacted')
parser.add_argument('-m', '--train-mode', type=str, default="dense",
                    choices = ["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
                    help = 'dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')

# cmap = plt.cm.jet
# # cmap = plt.cm.nipy_spectral
# def depth_colorize1(depth):
#     depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
#     depth = 255 * cmap(depth)[:,:,:3] # H, W, C
#     return depth.astype('uint8')
# def merge_into_row1(ele, pred):
#     def preprocess_depth(x):
#         y = np.squeeze(x.data.cpu().numpy())
#         return depth_colorize1(y)
#     # if is gray, transforms to rgb
#     img_list = []
#     if 'rgb' in ele:
#         rgb = np.squeeze(ele['rgb1'][0,...].data.cpu().numpy())
#         rgb = np.transpose(rgb, (1, 2, 0))
#         # img_list.append(rgb)
#         # img_list = []
#     # elif 'g' in ele:
#     #     g = np.squeeze(ele['g'][0,...].data.cpu().numpy())
#     #     g = np.array(Image.fromarray(g).convert('RGB'))
#     #     # img_list.append(g)
#     if 'd2' in ele:
#         img_list.append(preprocess_depth(ele['d'][0,...]))
#         d = np.hstack(img_list)
#         img_list = []
#     img_list.append(preprocess_depth(pred[0,...]))
#     pred = np.hstack(img_list)
#     img_list = []
#     if 'gt' in ele:
#         img_list.append(preprocess_depth(ele['gt'][0,...]))
#         gt = np.hstack(img_list)  
#     # img_merge = np.hstack(img_list)
#     # return img_merge.astype('uint8')
#     return rgb.astype('uint8') ,d.astype('uint8'),pred.astype('uint8'),gt.astype('uint8')
# def save_image1(img_merge, filename):
#     image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(filename, image_to_write)


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


cont =0
writer = SummaryWriter('logs')
# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    # fu, fv = float(K[0,0]), float(K[1,1])
    # cu, cv = float(K[0,2]), float(K[1,2])
    # kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv).cuda()


# moduleNetwork = Network().cuda().eval()
# FlowBackWarp = modelnet.backWarp(1216, 256, device)
# # moduleNetwork = Network().to(device).eval()
# def flownet(tensorPreprocessedFirst, tensorPreprocessedSecond,mode):
#     # torch.set_grad_enabled(False)
#     intWidth = 1216
#     intHeight = 256

#     # tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
#     # tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)
#     with torch.no_grad():
#         intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
#         intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

#         tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
#         tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
#         # nn.Upsample
        
#         tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

#         tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
#         tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
#         # if mode == 'train':
#         #     torch.set_grad_enabled(True)
#     return tensorFlow[:, :, :, :]


def train(mode, args, loader, model, optimizer, logger, epoch):
    global cont
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)

    model.train()
    lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    print('len(loader)=',len(loader))

    step = 0
    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
        # batch_data = {key:val.to(device) for key,val in batch_data.items() if val is not None}
        # if len(loader)-step <5:
        #     break
        if step == args.epochsample:
            break 
'''
        # # rgb1 = batch_data['rgb1']/255.0
        # # rgb = batch_data['rgb']/255.0
        # # rgb3 = batch_data['rgb3']/255.0
        # # #warp flow
        # # F_t_0= flownet(rgb, rgb1,mode) #1 2 240 1216
        # # F_t_1= flownet(rgb, rgb3,mode) #1 2 240 1216

        # # g_I0_F_t_0 = FlowBackWarp(batch_data['d'], F_t_0)
        # # g_I1_F_t_1 = FlowBackWarp(batch_data['d3'], F_t_1)
        # # #warp end


        # # with torch.no_grad():
        # #     d0 = np.squeeze(batch_data['d'][0,...].data.cpu().numpy())
        # #     d2 = np.squeeze(batch_data['d3'][0,...].data.cpu().numpy())
        # #     point0 = project_disp_to_depth(d0,d0,fu,fv,cu,cv)
        # #     point2 = project_disp_to_depth(d2,d2,fu,fv,cu,cv)
        # #     pc1=np.array(point0[:,0:3])
        # #     fig = visz_point.draw_lidar(pc1)
        # #     input("-----")
       
        # gt = batch_data['gt']
        # data_time = time.time() - start
        # start = time.time()

        # depth_flow = batch_data['depth_flow']
        # d2 = batch_data['d2']
        # print('d2.shape',d2.shape)
        # print('gt.shape',gt.shape)
        # print('depth_flow.shape',depth_flow.shape)

        # depth_flow = depth_flow.data.cpu().numpy()
        # print('depth_flow.shape',depth_flow.shape)
        # depth_flow = np.transpose(depth_flow, (0, 2, 3, 1))    
        # depth_flow = np.squeeze(depth_flow)
        # print(depth_flow.shape)
        # # curr_vect = mask1[0] *255 
        # # vect_fn = os.path.join('./results-png','F_t_0'+ '.' + 'png')
        # # flow_im = fl.flow_to_image(curr_vect)
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(depth_flow)
        # plt.show()
       

        # d2 = d2.data.cpu().numpy()
        # d2 = np.transpose(d2, (0, 2, 3, 1))    
        # d2 = np.squeeze(d2)
        # print(d2.shape)
        # # curr_vect = mask1[0] *255 
        # # vect_fn = os.path.join('./results-png','F_t_0'+ '.' + 'png')
        # # flow_im = fl.flow_to_image(curr_vect)
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(d2)
        # plt.show()
        # input('wait---')
        
        # display image
        # edge2 = batch_data['edge']
        # edge2 = edge2.data.cpu().numpy()
        # print('edge2.shape',edge2.shape)
        # edge2 = np.transpose(edge2, (0, 2, 3, 1))    
        # edge2 = np.squeeze(edge2)
        # print(edge2.shape)
        # # curr_vect = mask1[0] *255 
        # # vect_fn = os.path.join('./results-png','F_t_0'+ '.' + 'png')
        # # flow_im = fl.flow_to_image(curr_vect)
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(edge2)
        # plt.show()
               
        
    
        # # read point cloud file
        # pcfile0= 'pc_data/'+mode+'/'+str(i)+'_0.npy'
        # pcfile2= 'pc_data/'+mode+'/'+str(i)+'_2.npy'
        # pc0=np.load(pcfile0).reshape(1,-1,3)
        # pc0=torch.from_numpy(pc0)
        # pc2=np.load(pcfile2).reshape(1,-1,3)
        # pc2=torch.from_numpy(pc2)
        

        # pred , coarse = model(batch_data,pred_flow)
        # pred , coarse = model(batch_data,F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0)
        pred = model(device,batch_data['d'],batch_data['d3'],batch_data['mask1'],batch_data['mask3'],batch_data['rgb'],batch_data['edge'])  
        # pred , coarse = model(batch_data['d'],batch_data['d3'],F_t_0, F_t_1, g_I1_F_t_1, g_I0_F_t_0,batch_data['rgb'],pc0,fu,fv,cu,cv)  
        # if step%100==0:
        #     print('----mask_softmax',mask[0,:,0:226,116:166])
        # writer.add_graph(model,(batch_data['d'],batch_data['d3'],batch_data['pc0'],batch_data['pc2'],))
        # with SummaryWriter(comment='model') as w:
        #     w.add_graph(model,(batch_data['d'],batch_data['d3'],batch_data['pc0'],batch_data['pc2']))

        depth_loss= 0
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d2'])
                mask = (batch_data['d2'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                # depth_loss_coarse = depth_criterion(coarse, gt)
                mask = (gt < 1e-3).float()

            if step % 100 == 0:
                print('depth_loss',depth_loss)
                # print('2',depth_loss_coarse)
            # backprop
            # loss = depth_loss +depth_loss_coarse*0.3 # args.w1*photometric_loss + 
            # loss = depth_loss + args.w1*photometric_loss + args.w2*smooth_loss
            loss = depth_loss

            writer.add_scalar('loss', loss, cont)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            cont = cont + 1

        gpu_time = time.time() - start
        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data)
           
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            # logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
        step=step+1  

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)
    writer.close()
    print('-------here------')
'''
def val(mode, args, loader, model, logger, epoch):

    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    model.eval()       
    lr = 0
    print('len(loader)=',len(loader))
    step = 0 
    list1=[]
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            start = time.time()
            batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
            # # batch_data = {key:val.to(device) for key,val in batch_data.items() if val is not None}
            # if len(loader)-step <5:
            #     break
            # if step == args.epochsample:
            if step == 8000:
                break 
'''            # rgb1 = batch_data['rgb1']/255.0
            # rgb  = batch_data['rgb']/255.0
            # rgb3 = batch_data['rgb3']/255.0
            # # pred_flow = flownet(rgb1, rgb3,mode) #1 2 240 1216
           
            # #warp flow
            # F_0_1= flownet(rgb1, rgb3,mode) #1 2 240 1216
            # F_1_0= flownet(rgb3, rgb1,mode) #1 2 240 1216
            # # fCoeff = modelnet.getFlowCoeff(device)
            # # F_t_0 = -0.25 * F_0_1 + 0.25 * F_1_0
            # # F_t_1 =  0.25 * F_0_1 - 0.25 * F_1_0
            # F_t_0= flownet(rgb, rgb1,mode) #1 2 240 1216
            # F_t_1= flownet(rgb, rgb3,mode) #1 2 240 1216

            # g_I0_F_t_0 = FlowBackWarp(batch_data['d'], F_t_0)
            # g_I1_F_t_1 = FlowBackWarp(batch_data['d3'], F_t_1)
            # #warp end
            gt = batch_data['gt']
            data_time = time.time() - start
            start = time.time()

                    # read point cloud file
            # pcfile0= 'pc_data/'+mode+'/'+str(i)+'_0.npy'
            # pcfile2= 'pc_data/'+mode+'/'+str(i)+'_2.npy'
            # pc0=np.load(pcfile0).reshape(1,-1,3)
            # pc0=torch.from_numpy(pc0)
            # pc2=np.load(pcfile2).reshape(1,-1,3)
            # pc2=torch.from_numpy(pc2)
            # pred = model(batch_data['d'],batch_data['d3'],pc0,pc2,fu,fv,cu,cv)
            # pred = model(device,batch_data['d'],batch_data['d3'],batch_data['pc0'],batch_data['pc2']) 
            pred = model(device,batch_data['d'],batch_data['d3'],batch_data['mask1'],batch_data['mask3'],batch_data['rgb'],batch_data['edge']) 
            # pred = model(device,batch_data['d2'],batch_data['d3'],batch_data['mask2'],batch_data['mask3'],batch_data['rgb'])  



            # pred , coarse ,mask= model(batch_data,pred_flow)
            # pred , coarse = model(batch_data,pred_flow)
            # pred , coarse = model(batch_data,F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0)  
            # pred , coarse = model(batch_data['d'],batch_data['d3'],F_t_0, F_t_1, g_I1_F_t_1, g_I0_F_t_0,batch_data['rgb']) 
            # pred , coarse = model(batch_data['d'],batch_data['d3'],F_t_0, F_t_1, g_I1_F_t_1, g_I0_F_t_0)  
            gpu_time = time.time() - start
            # measure accuracy and record loss
            step=step+1 

            # d0 = np.squeeze(batch_data['d'][0,...].data.cpu().numpy())
            # d2 = np.squeeze(batch_data['d3'][0,...].data.cpu().numpy())
            # point0 = project_disp_to_depth(d0,d0,fu,fv,cu,cv)
            # point2 = project_disp_to_depth(d2,d2,fu,fv,cu,cv)
            # print(point0.shape)
            
            # print(point0.shape)
            # print(point2.shape)
            # lidar = pointcloud.project_disp_to_depth(rgb_viz,d2_viz,pred_viz)
            # pc1=np.array(point0[:,0:3])
            # fig = visz_point.draw_lidar(pc1)
            # input("-----1---")


            # img_merge=merge_into_row1(batch_data, pred) 
            # filename = 'results-png/'+'rgb'+str(i)+'.png'
            # save_image1(img_merge[0], filename)
            # filename = 'results-png/'+'d'+str(i)+'.png'
            # save_image1(img_merge[1], filename)
            # filename = 'results-png/'+'pred'+str(i+1)+'.png'
            # save_image1(img_merge[2], filename)
            # filename = 'results-png/'+'gt'+str(i+1)+'.png'
            # save_image1(img_merge[3], filename)
            # input('---------generation test.png----------')

            # img_merge=merge_into_row1(batch_data, g_I1_F_t_1) 
            # # filename = 'results-png/2rgb.png'
            # # save_image1(img_merge[0], filename)
            # # filename = 'results-png/2d.png'
            # # save_image1(img_merge[1], filename)
            # filename = 'results-png/warp_t_1.png'
            # save_image1(img_merge[2], filename)

            # pred_vect = F_t_0.data.cpu().numpy()
            # pred_vect = np.transpose(pred_vect, (0, 2, 3, 1))    
            # curr_vect = pred_vect[0]   
            # vect_fn = os.path.join('./results-png','F_t_0'+ '.' + 'png')
            # flow_im = fl.flow_to_image(curr_vect)
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(flow_im)
            # plt.show()

            # plt.savefig('flow--t-0.png', bbox_inches='tight',pad_inches=0.0)
            # print('Done!')


            # # cv2.imwrite(vect_fn, flow_im)

            # pred_vect = F_t_1.data.cpu().numpy()
            # pred_vect = np.transpose(pred_vect, (0, 2, 3, 1))    
            # curr_vect = pred_vect[0]   
            # vect_fn = os.path.join('./results-png','F_t_1' + '.' + 'png')
            # flow_im = fl.flow_to_image(curr_vect)
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(flow_im)
            # plt.show()
            # plt.savefig('flow--t-1.png', bbox_inches='tight',pad_inches=0.0)
            # # cv2.imwrite(vect_fn, flow_im)
            # plt.imshow(flow_im)
            # # plt.show()
            # plt.savefig(vect_fn, bbox_inches='tight')
            # temp=[50,100,250,400,750,900,950]
            # temp=[900,950]
            # if i in temp:
            # temp = range(60)
            # if i in temp[::]:

            #     img_merge=merge_into_row1(batch_data, pred) 
            #     filename = 'results-png/'+'rgb'+str(i)+'.png'
            #     save_image1(img_merge[0], filename)
            #     filename = 'results-png/'+'d'+str(i)+'.png'
            #     save_image1(img_merge[1], filename)
            #     filename = 'results-png/'+'pred'+str(i+1)+'.png'
            #     save_image1(img_merge[2], filename)
            #     filename = 'results-png/'+'gt'+str(i+1)+'.png'
            #     save_image1(img_merge[3], filename)


                # img_merge=merge_into_row1(batch_data, g_I0_F_t_0) 
                # name= str(i)
                # # filename = 'results-png/'+name+'rgb.png'
                # # save_image1(img_merge[0], filename)
                # filename = 'results-png/'+name+'d1.png'
                # save_image1(img_merge[1], filename)
                # filename = 'results-png/'+name+'pred0.png'
                # save_image1(img_merge[2], filename)
                # # filename = 'results-png/'+name+'gt.png'
                # # save_image1(img_merge[3], filename)
                # input('---------generation test.png----------')

            #     # gt_viz = np.squeeze(gt[0,...].data.cpu().numpy())
            # d_viz = np.squeeze(batch_data['d'][0,...].data.cpu().numpy())
            #     # d2_viz = np.squeeze(batch_data['d2'][0,...].data.cpu().numpy())
            #     # pred_viz = np.squeeze(pred[0,...].data.cpu().numpy())
            # print('d_viz',d_viz.shape)

            # # rgb_viz = np.squeeze(batch_data['rgb'][0,...].data.cpu().numpy())
            # # rgb_viz = np.transpose(rgb_viz, (1, 2, 0))
            # # rgb_viz=rgb_viz.reshape(256,1216,-1)
               
            # # lidar =project_disp_to_depth(rgb_viz,d_viz,d_viz)
            #     # pc=np.array(lidar[:,0:3])
            #     # fig = visz_point.draw_lidar(pc)
            # input("----0----")
                # lidar = pointcloud.project_disp_to_depth(rgb_viz,d2_viz,pred_viz)
                # pc1=np.array(lidar[:,0:3])
                # fig = visz_point.draw_lidar(pc1)
                # input("-----1---")

                # ## pptk
                # import pptk
                # from colorsys import hsv_to_rgb
              
                # pc_all = np.concatenate((pc,pc1),axis=0)
                # print(pc_all.shape)   #98993*2
                # fig = visz_point.draw_lidar_simple(pc_all)
                # input("-----2---")
                # viewer = pptk.viewer(pc_all)
                # viewer.set(point_size=0.01, show_axis=False, bg_color_bottom=[0.1,0.1,0.1,0.5])

                # n_instances = 2 # 2
                # hues = np.linspace(0,1, n_instances+1)
                # np.random.shuffle(hues)
                # inst_colors = np.array([hsv_to_rgb(h, 0.7, 0.85) for h in hues])
                # inst_colors[0,:] = [0.4, 0.4, 0.4]
                # # inst_colors[1,:] = [0.4, 0.4, 0.4]
                # # inst_colors[2,:] = [0.4, 0.4, 0.4]

                # pc_len = pc.shape[0]
                # pc1_len = pc.shape[0]

                # pc_all_index = np.zeros([pc_len + pc1_len],dtype = int)
                # for i in range(pc_len+pc1_len):
                #     if i<pc_len:
                #         pc_all_index[i] = 1
                #     else:
                #         pc_all_index[i] = 2
                # viewer.attributes(inst_colors[pc_all_index :])

            #     mini_batch_size = next(iter(batch_data.values())).size(0)
            #     result = Result()       
            #     result.evaluate(pred.data, gt.data)        
            #     [m.update(result, data_time, mini_batch_size) for m in meters]
            #     # list1.append(i)
            #     # list1.append(result.rmse)
            #     # result1 = np.array(list1).reshape(-1,2)
            #     # np.savetxt('npresult1.csv',result1,fmt='%d',delimiter=',')
            #     print('--------------',i,result.rmse)
            #     # input('---------continue-----')
            # else:
            #     print('---------continue-----',i)
            #     continue
            


            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()       
            result.evaluate(pred.data, gt.data)        
            [m.update(result, data_time, mini_batch_size) for m in meters]
            # list1.append(i)
            # list1.append(result.rmse)
            # result1 = np.array(list1).reshape(-1,2)
            # np.savetxt('npresult1.csv',result1,fmt='%d',delimiter=',')
            # print('--------------',i,result.rmse)
            # input('---------continue-----')

            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
           

        avg = logger.conditional_save_info(mode, average_meter, epoch)
        is_best = logger.rank_conditional_save_best(mode, avg, epoch)
        if is_best and not (mode == "train"):
            logger.save_img_comparison_as_best(mode, epoch)
        logger.conditional_summarize(mode, avg, is_best)
        
        return avg, is_best
'''
# def save_to_file(file_name, contents):
#     fh = open(file_name, 'w')
#     for i in contents:

#         temp=str(i)
#         fh.write(temp)
#         fh.write('\n')
#     fh.close()

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            args = checkpoint['args']
            is_eval = True
            print("=> checkpoint loaded.")
        else:
            print("=> no model found at '{}'".format(args.evaluate))
            return
    elif args.resume: # optionally resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # # # print("=> creating model and optimizer...")
  
    # # model = DepthCompletionNet(args)
    # #    #lhj one gpu
    # # if torch.cuda.device_count()>1:
    # #     model = torch.nn.DataParallel(model)
    # # model = DepthCompletionNet(args).to(device)
    # # torch.cuda.set_device(0)
    # print("=> creating model and optimizer...")
    # torch.cuda.set_device(0)
    # model = DepthCompletionNet(args).to(device)

    print("=> creating model and optimizer...")
    # torch.cuda.set_device(0)
    model = DepthCompletionNet(args)
    model.cuda()


    # print("=> model transferred to multi-GPU.")
    # if torch.cuda.device_count()>1:
    #     model = torch.nn.DataParallel(model,device_ids=device_ids)


    model_named_params = [p for _,p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)

    print("=> model and optimizer created.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    # Data loading code
    print("=> creating data loaders ...")
    if not is_eval:
        print("KittiDepth('train', args)")
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=args.workers, pin_memory=True, sampler=None)
    
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True) # set batch size to be 1 for validation
    print("=> data loaders created.")
    print(len(val_loader))
    # paths, transform1 = get_paths_and_transform('val', args)
    # filename_d=paths['d']
    # # print(filename_d)
    # # result1 = np.array(filename_d)
    # save_to_file('mobiles.txt', filename_d)
    # input("--------")
    # np.savetxt('filename_d.txt',filename_d)
    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        result, is_best = val("val", args, val_loader, model, logger, checkpoint['epoch'])
        return
    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        
        # train("train", args, train_loader, model, optimizer, logger, epoch) # train for one epoch
    
        result, is_best = val("val", args, val_loader, model, logger, epoch) # evaluate on validation set

        # helper.save_checkpoint({ # save checkpoint
        #     'epoch': epoch,
        #     # 'model': model.module.state_dict(),
        #     'model': model.state_dict(),
        #     'best_result': logger.best_result,
        #     'optimizer' : optimizer.state_dict(),
        #     'args' : args,
        # }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()

