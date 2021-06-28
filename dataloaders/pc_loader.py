import cv2
import numpy as np
# import mayavi.mlab as mlab
import matplotlib.pyplot as plt
# import pptk
# import viz_util
import sys
# sys.path.append('/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/PLin++')

import torch
# from dataloaders.kitti_loader import load_calib
# from dataloaders import kitti_loader
# import kitti_loader as load_calib
# rgb = cv2.imread('5depth.png') 
# gt_depth1 = cv2.imread('5sparse.png',-1)
# gt_depth1 = gt_depth1.astype(np.float64) / 256

# camera_cx = 596.5593
# camera_cy = 149.854
# camera_fx = 721.5377
# camera_fy = 721.5377


 # hard-coded KITTI camera intrinsics
# K ,C2V,R0= kitti_loader.load_calib()
# fu, fv = float(K[0,0]), float(K[1,1])
# cu, cv = float(K[0,2]), float(K[1,2])

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


from colorsys import hsv_to_rgb

def project_image_to_rect(uv_depth,fu,fv,cu,cv):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    # print('uv_depth[:, 0]',uv_depth[:, 0])
    x = ((uv_depth[:, 0] - cu) * uv_depth[:, 2]) / fu
    y = ((uv_depth[:, 1] - cv) * uv_depth[:, 2]) / fv 
    pts_3d_rect = np.zeros((n, 3))

    # # print(rgb.shape)
    # b1 = rgb[:,0]
    # g1 = rgb[:,1]
    # r1 = rgb[:,2]
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    # pts_3d_rect[:, 0] = -y
    # pts_3d_rect[:, 1] = -x
    # pts_3d_rect[:, 2] = uv_depth[:, 2]
    # pts_3d_rect[:, 3] = r1
    # pts_3d_rect[:, 4] = b1
    # pts_3d_rect[:, 5] = g1
    # for i in  range(n):
    #     if pts_3d_rect[i, 2] < 0.5:
    #        del  pts_3d_rect[i, 2]


    return pts_3d_rect

# def project_disp_to_depth(rgb,disp):
#     # disp[disp < 0] = 0
#     # baseline = 0.54
#     mask = disp > 0
#     # depth = calib.f_u * baseline / (disp + 1. - mask)
#     depth = disp
#     # rows, cols = depth.size()
#     rows, cols = 256,1216
#     c, r = np.meshgrid(np.arange(cols), np.arange(rows))
#     points = np.stack([c, r, depth])
#     points = points.reshape((3, -1))
#     points = points.T
#     print(points.size())
#     points = points[mask.reshape(-1)]
#     print(points.size())
#     rgb=rgb[mask]
#     cloud = project_image_to_rect(points,rgb)

#     # valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
#     # return cloud[valid]
#     return cloud
def project_disp_to_depth(gt,depth,fu,fv,cu,cv):
    # disp[disp < 0] = 0
    # baseline = 0.54
    # valid_mask = disp>0.1

        # convert from meters to mm
    # output_mm = 1e3 * output[valid_mask]
  
    mask = gt > 1.0
  
    # depth = calib.f_u * baseline / (disp + 1. - mask)
    # print('1',disp.shape)
    # depth = disp.view(256,-1)
    # print('2',depth.shape)
    # rows, cols = depth.size()
    rows, cols = 256,1216
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    # points = np.stack([c+13, r+119, depth])
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    # print('1',points.shape)
    # print('2',mask.reshape(-1).shape)
    points = points[mask.reshape(-1)]
    # print('3points.shape',points.shape)
    # rgb=rgb[mask]
    # print(rgb.shape)
    # cloud = project_image_to_rect(points,fu,fv,cu,cv)


    # pts_3d_ref=np.transpose(np.dot(np.linalg.inv(R0), np.transpose(cloud[:,0:3])))
    # pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    # cloud=np.dot(pts_3d_ref, np.transpose(C2V))
    # valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    # return cloud[valid]
    return points



# def project_feature_to_image(uv_depth,fu,fv,cu,cv,xyz,feature):
#     ''' Input: nx3 first two channels are uv, 3rd channel
#                is depth in rect camera coord.
#         Output: nx3 points in rect camera coord.
#     '''
#     n = uv_depth.shape[0]
#     # print('uv_depth[:, 0]',uv_depth[:, 0])
#     x = ((uv_depth[:, 0] - cu) * uv_depth[:, 2]) / fu
#     y = ((uv_depth[:, 1] - cv) * uv_depth[:, 2]) / fv 
#     pts_3d_rect = np.zeros((n, 3))

#     # # print(rgb.shape)
#     # b1 = rgb[:,0]
#     # g1 = rgb[:,1]
#     # r1 = rgb[:,2]
#     pts_3d_rect[:, 0] = x
#     pts_3d_rect[:, 1] = y
#     pts_3d_rect[:, 2] = uv_depth[:, 2]

#     #(1,C,256,1216)
#     uv_feature[1,] = (xyz[0,:,0]*fu)/xyz[0,:,2]+cu
#     return pts_3d_rect

# # if __name__=='__main__':


# #     lidar = project_disp_to_depth(rgb,gt_depth1, 1)

# #     # # np.savetxt("pc.txt",lidar)
# #     pc=np.array(lidar[:,0:3])
# #     # # pc = np.loadtxt('pc.txt')
# #     # fig = viz_util.draw_lidar(pc)
# #     # raw_input()
 
# #     v = pptk.viewer(pc)
# #     # v.attributes(lidar[:,3:6]/255)
# #     v.set(point_size=0.001)
# #     input("--------")

# def project_feature_to_image(fu,fv,cu,cv,xyz,feature,uv_feature):
#     ''' Input: nx3 first two channels are uv, 3rd channel
#                is depth in rect camera coord.
#         Output: nx3 points in rect camera coord.
#     '''
#     # print('uv_feature.shape',uv_feature.shape)
#     u=xyz[0,:,0]*fu/xyz[0,:,2]+cu
#     v=xyz[0,:,1]*fv/xyz[0,:,2]+cv
#     # d=xyz[0,:,2]
#     # u=np.rint(u).astype(int)
#     # v=np.rint(v).astype(int)
#     u=torch.round(u-13).type(torch.int64)
#     v=torch.round(v-119).type(torch.int64)
#     #style-1
#     uv_feature[0,:,v,u]=feature[0,:,:]
#     # temp=uv_feature
#     # #style-2
#     # for i in np.arange(xyz.shape[1]):
#     #     uv_feature[0,:,v[i],u[i]]=feature[0,:,i]
#     # result=torch.equal(uv_feature,temp)
#     # print(result)
#     # print('uv',uv_feature.shape)
#     # print('uv',uv_feature)
#     return uv_feature


def project_feature_to_image(fu,fv,cu,cv,xyz,feature,uv_feature):

    u = xyz[0,:,0]*fu/xyz[0,:,2] + cu -13.0
    v = xyz[0,:,1]*fv/xyz[0,:,2] + cv -119.0
    # d=xyz[0,:,2]
    # u=np.rint(u).astype(int)
    # v=np.rint(v).astype(int)
    u = torch.round(u).type(torch.int64)
    v = torch.round(v).type(torch.int64)
    #style-1
    uv_feature[0,:,v,u] = feature[0,:,:]
    return uv_feature


###### gpu 1 

#(x,y,z)
def project_d_feature(fu,fv,cu,cv,xyz,feature,uv_feature):

    u = xyz[0,:,0]
    v = xyz[0,:,1]
    # d=xyz[0,:,2]
    # u=np.rint(u).astype(int)
    # v=np.rint(v).astype(int)
    u = torch.round(u).type(torch.int64)
    v = torch.round(v).type(torch.int64)
    # print(torch.max(u),torch.max(v))
    #style-1
    uv_feature[0,:,v,u] = feature[0,:,:]
    return uv_feature


#(x,y,z)
def project_feature_to_image_gpu1(fu,fv,cu,cv,xyz,feature,uv_feature):

    u = xyz[0,:,0]*fu/xyz[0,:,2] + cu -13.0
    v = xyz[0,:,1]*fv/xyz[0,:,2] + cv -119.0
    # d=xyz[0,:,2]
    # u=np.rint(u).astype(int)
    # v=np.rint(v).astype(int)
    u = torch.round(u).type(torch.int64)
    v = torch.round(v).type(torch.int64)
    #style-1
    uv_feature[0,:,v,u] = feature[0,:,:]
    return uv_feature


###### gpu 1 

#(x,y,d)

# def project_feature_to_image_gpu1(fu,fv,cu,cv,xyz,feature,uv_feature):

#     u = xyz[0,:,0]
#     v = xyz[0,:,1]
#     u = torch.round(u).type(torch.int64)
#     v = torch.round(v).type(torch.int64)
#     #style-1
#     uv_feature[0,:,v,u] = feature[0,:,:]
#     return uv_feature


def project_xyz_to_image(fu,fv,cu,cv,xyz,feature,uv):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    # print('uv_feature.shape',uv_feature.shape)
    u = xyz[0,:,0]*fu/xyz[0,:,2] + cu - 13.0
    v = xyz[0,:,1]*fv/xyz[0,:,2] + cv - 119.0
    d = xyz[0,:,2]
    # u=np.rint(u).astype(int)
    # v=np.rint(v).astype(int)
    u = torch.round(u).type(torch.int64)
    v = torch.round(v).type(torch.int64)
    #style-1
    # print(u[:6],v[:6])
    uv[0,:,v,u] = xyz[0,:,2]
    # if xyz.shape[0]>1:
    #     u=xyz[1,:,0]*fu/xyz[1,:,2]+cu
    #     v=xyz[1,:,1]*fv/xyz[1,:,2]+cv
    #     # d=xyz[0,:,2]
    #     # u=np.rint(u).astype(int)
    #     # v=np.rint(v).astype(int)
    #     u=torch.round(u-13).type(torch.int64)
    #     v=torch.round(v-119).type(torch.int64)
    #     #style-1
    #     uv_feature[1,:,v,u]=feature[1,:,:]

    # temp=uv_feature
    # #style-2
    # for i in np.arange(xyz.shape[1]):
    #     uv_feature[0,:,v[i],u[i]]=feature[0,:,i]
    # result=torch.equal(uv_feature,temp)
    # print(result)
    # print('uv',uv_feature.shape)
    # print('uv',uv_feature)
    # rgb = np.squeeze(ele['rgb1'][0,...].data.cpu().numpy())
    # uv=np.squeeze(uv[0,...].data.cpu().numpy())
    # # uv = np.transpose(uv, (1,2,0))
    # plt.figure()
    # plt.imshow(uv)
    # plt.show()
    # plt.savefig('uv_projection.png', bbox_inches='tight',pad_inches=0.0)
    return uv


# def project_image_to_rect_gpu1(uv_depth,fu,fv,cu,cv):
#     n = uv_depth.shape[0]
#     # print('uv_depth[:, 0]',uv_depth[:, 0])
#     x = ((uv_depth[:, 0] - cu) * uv_depth[:, 2]) / fu
#     y = ((uv_depth[:, 1] - cv) * uv_depth[:, 2]) / fv 
#     pts_3d_rect = np.zeros((n, 3))

#     pts_3d_rect[:, 0] = x
#     pts_3d_rect[:, 1] = y
#     pts_3d_rect[:, 2] = uv_depth[:, 2]

#     return pts_3d_rect

def project_disp_to_depth_gpu1(gt,depth,fu,fv,cu,cv):
    mask = gt > 1.0
    rows, cols = 256,1216
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
   
    points = points[mask.reshape(-1)]
    # print('3points.shape',points.shape)
    # rgb=rgb[mask]
    # print(rgb.shape)
    # cloud = project_image_to_rect(points,fu,fv,cu,cv)


    # pts_3d_ref=np.transpose(np.dot(np.linalg.inv(R0), np.transpose(cloud[:,0:3])))
    # pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    # cloud=np.dot(pts_3d_ref, np.transpose(C2V))
    # valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    # return cloud[valid]
    return points