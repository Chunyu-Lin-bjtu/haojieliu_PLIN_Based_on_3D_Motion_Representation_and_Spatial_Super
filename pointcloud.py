import cv2
import numpy as np
# import mayavi.mlab as mlab
import matplotlib.pyplot as plt
# import pptk
# import viz_util
import torch
from dataloaders.kitti_loader_d import load_calib

# rgb = cv2.imread('5depth.png') 
# gt_depth1 = cv2.imread('5sparse.png',-1)
# gt_depth1 = gt_depth1.astype(np.float64) / 256

# camera_cx = 596.5593
# camera_cy = 149.854
# camera_fx = 721.5377
# camera_fy = 721.5377


 # hard-coded KITTI camera intrinsics
K ,C2V,R0= load_calib()
fu, fv = float(K[0,0]), float(K[1,1])
cu, cv = float(K[0,2]), float(K[1,2])

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


from colorsys import hsv_to_rgb

def project_image_to_rect(uv_depth):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    print('uv_depth[:, 0]',uv_depth[:, 0])
    x = ((uv_depth[:, 0] - cu) * uv_depth[:, 2]) / fu
    y = ((uv_depth[:, 1] - cv) * uv_depth[:, 2]) / fv 
    pts_3d_rect = np.zeros((n, 3))

    # print(rgb.shape)
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
def project_disp_to_depth(d,gt,depth):
    # disp[disp < 0] = 0
    # baseline = 0.54
    # valid_mask = disp>0.1

        # convert from meters to mm
    # output_mm = 1e3 * output[valid_mask]
  
    mask = (gt+d) > 1.0 
  
    # depth = calib.f_u * baseline / (disp + 1. - mask)
    # print('1',disp.shape)
    # depth = disp.view(256,-1)
    # print('2',depth.shape)
    # rows, cols = depth.size()
    rows, cols = 256,1216
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c+13, r+60, depth])
    points = points.reshape((3, -1))
    points = points.T
    # print('1',points.shape)
    # print('2',mask.reshape(-1).shape)
 
    points = points[mask.reshape(-1)]
    # print('3',points.shape)
    # rgb=rgb[mask]
    # print(rgb.shape)
    cloud = project_image_to_rect(points)


    # pts_3d_ref=np.transpose(np.dot(np.linalg.inv(R0), np.transpose(cloud[:,0:3])))
    # pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    # cloud=np.dot(pts_3d_ref, np.transpose(C2V))
    # valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    # return cloud[valid]
    return cloud



# if __name__=='__main__':


#     lidar = project_disp_to_depth(rgb,gt_depth1, 1)

#     # # np.savetxt("pc.txt",lidar)
#     pc=np.array(lidar[:,0:3])
#     # # pc = np.loadtxt('pc.txt')
#     # fig = viz_util.draw_lidar(pc)
#     # raw_input()
 
#     v = pptk.viewer(pc)
#     # v.attributes(lidar[:,3:6]/255)
#     v.set(point_size=0.001)
#     input("--------")