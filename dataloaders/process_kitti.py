import os, sys
import os.path as osp
import numpy as np
import png
from multiprocessing import Pool
import time
from dataloaders.kitti_loader_d_flow import pc_loader
from dataloaders.visualization import visual
import torch 


# calib_root = './utils/calib_cam_to_cam/'
data_root = sys.argv[1]
# disp1_root = osp.join(data_root, 'training/disp_occ_0')
# disp2_root = osp.join(data_root, 'training/disp_occ_1')
# op_flow_root = osp.join(data_root, 'training/flow_occ')

save_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/save_path/'
# flow_pc_path= '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/test/ours_KITTI_train_3000_35m/'
flow_pc_path= '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/HPLFlowNet/checkpoints/data/'

# def pixel2xyz(depth, P_rect, px=None, py=None):
#     assert P_rect[0,1] == 0
#     assert P_rect[1,0] == 0
#     assert P_rect[2,0] == 0
#     assert P_rect[2,1] == 0
#     assert P_rect[0,0] == P_rect[1,1]
#     focal_length_pixel = P_rect[0,0]

#     c_u=P_rect[0,2]
#     c_v=P_rect[1,2]
#     f_u=focal_length_pixel
#     f_v=focal_length_pixel
    
#     height, width = depth.shape[:2]
#     if px is None:
#         px = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1))
#     if py is None:
#         py = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
#     print('P_rect:',P_rect)
#     const_x = c_u * depth + P_rect[0,3]
#     const_y = c_v * depth + P_rect[1,3]


    
#     x = ((px * (depth + P_rect[2,3]) - const_x) / f_u) [:, :, None]
#     y = ((py * (depth + P_rect[2,3]) - const_y) / f_u) [:, :, None]
#     pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)
    
#     pc[..., :2] *= -1.
#     return pc

def pixel2xyz(depth, P_rect, px=None, py=None):

    c_u=P_rect[0,2]
    c_v=P_rect[1,2]
    f_u=P_rect[0,0]
    f_v=P_rect[1,1]
    b_x=P_rect[0,3] / (-f_u)
    b_y=P_rect[1,3] / (-f_v)

    
    height, width = depth.shape[:2]
    if px is None:
        px = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1))
    if py is None:
        py = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    const_x = c_u * depth + P_rect[0,3]
    const_y = c_v * depth + P_rect[1,3]
    x = ((px * (depth + P_rect[2,3]) - const_x) / f_u) [:, :, None]
    y = ((py * (depth + P_rect[2,3]) - const_y) / f_v) [:, :, None]
    pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)
    
    pc[..., :2] *= -1.
    return pc

def xyz2pixel(depth, P_rect,xyz):

    c_u=P_rect[0,2]
    c_v=P_rect[1,2]
    f_u=P_rect[0,0]
    f_v=P_rect[1,1]
    b_x=P_rect[0,3] / (-f_u)
    b_y=P_rect[1,3] / (-f_v)
    
    # start = time.time()
    # print('depth.shape',depth.shape) #(375, 1242)
    height, width = depth.shape[:2]
    xyz[..., :2] *= -1.
    u = (xyz[:,0]-b_x)*f_u / xyz[:,2] + c_u
    v = (xyz[:,1]-b_y)*f_v / xyz[:,2] + c_v
    u=np.round(u).astype(int)
    v=np.round(v).astype(int)
    u_mask = np.logical_and(u < width, u >= 0)  #1224
    v_mask = np.logical_and(v < height, v >= 0) #370
    mask = np.logical_and(u_mask, v_mask)

    depth[v[mask],u[mask]] = xyz[mask,2]
    return depth



def project_image_to_rect(uv_depth,P_rect):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    c_u=P_rect[0,2]
    c_v=P_rect[1,2]
    f_u=P_rect[0,0]
    f_v=P_rect[1,1]
    b_x=P_rect[0,3] / (-f_u)
    b_y=P_rect[1,3] / (-f_v)

    rows, cols = uv_depth.shape
    mask = uv_depth > 0
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, uv_depth])
    points = points.reshape((3, -1))
    points = points.T
    uv_depth = points[mask.reshape(-1)]

    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def load_uint16PNG(fpath):
    reader = png.Reader(fpath)
    pngdata = reader.read()
    px_array = np.vstack( map(np.uint16, pngdata[2]) )
    if pngdata[3]['planes'] == 3:
        width, height = pngdata[:2]
        px_array = px_array.reshape(height, width, 3)
    return px_array


def load_disp(fpath):
    # A 0 value indicates an invalid pixel (ie, no
    # ground truth exists, or the estimation algorithm didn't produce an estimate
    # for that pixel).
    array = load_uint16PNG(fpath)
    valid = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid)] = -1.
    return disp, valid


def load_op_flow(fpath):
    array = load_uint16PNG(fpath)
    valid = array[..., -1] == 1
    array = array.astype(np.float32)
    flow = (array[..., :-1] - 2**15) / 64.
    return flow, valid


def disp_2_depth(disparity, valid_disp, FOCAL_LENGTH_PIXEL):
    BASELINE = 0.54
    depth = FOCAL_LENGTH_PIXEL * BASELINE / (disparity + 1e-5)
    depth[np.logical_not(valid_disp)] = -1.
    return depth


# def process_one_frame(idx):
#     sidx = '{:06d}'.format(idx)

#     calib_path = osp.join(calib_root, sidx + '.txt')
#     with open(calib_path) as fd:
#         lines = fd.readlines()
#         assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
#         P_rect_left = \
#             np.array([float(item) for item in
#                       [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
#                      dtype=np.float32).reshape(3, 4)

#     assert P_rect_left[0, 0] == P_rect_left[1, 1]
#     focal_length_pixel = P_rect_left[0, 0]

#     disp1_path = osp.join(disp1_root, sidx + '_10.png')
#     disp1, valid_disp1 = load_disp(disp1_path)
#     depth1 = disp_2_depth(disp1, valid_disp1, focal_length_pixel)
#     pc1 = pixel2xyz(depth1, P_rect_left)

#     disp2_path = osp.join(disp2_root, sidx + '_10.png')
#     disp2, valid_disp2 = load_disp(disp2_path)
#     depth2 = disp_2_depth(disp2, valid_disp2, focal_length_pixel)

#     valid_disp = np.logical_and(valid_disp1, valid_disp2)

#     op_flow, valid_op_flow = load_op_flow(osp.join(op_flow_root, '{:06d}_10.png'.format(idx)))
#     vertical = op_flow[..., 1]
#     horizontal = op_flow[..., 0]
#     height, width = op_flow.shape[:2]

#     px2 = np.zeros((height, width), dtype=np.float32)
#     py2 = np.zeros((height, width), dtype=np.float32)

#     for i in range(height):
#         for j in range(width):
#             if valid_op_flow[i, j] and valid_disp[i, j]:
#                 try:
#                     dx = horizontal[i, j]
#                     dy = vertical[i, j]
#                 except:
#                     print('error, i,j:', i, j, 'hor and ver:', horizontal[i, j], vertical[i, j])
#                     continue

#                 px2[i, j] = j + dx
#                 py2[i, j] = i + dy

#     pc2 = pixel2xyz(depth2, P_rect_left, px=px2, py=py2)

#     final_mask = np.logical_and(valid_disp, valid_op_flow)

#     valid_pc1 = pc1[final_mask]
#     valid_pc2 = pc2[final_mask]

#     truenas_path = osp.join(save_path, '{:06d}'.format(idx))
#     os.makedirs(truenas_path, exist_ok=True)
#     np.save(osp.join(truenas_path, 'pc1.npy'), valid_pc1)
#     np.save(osp.join(truenas_path, 'pc2.npy'), valid_pc2)
def process_one_frame_one_P(split_trainval,idx,depth1,depth3):
    """ Write image to pc.
        project 2D depth to 3D pc
    """

    print('loading ---{:06d}'.format(idx))
    calib_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calib_cam_to_cam.txt'
    print('loading ',idx)
    # indexstar= int(filename.find('2011_'))
    # indexend= int(filename.find('_drive'))
    # data_calid = filename[indexstar:indexend]
    # calib_path = osp.join(calib_root,data_calid ,'calib_cam_to_cam.txt')
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    # disp1_path = osp.join(disp1_root, sidx + '_10.png')
    # disp1, valid_disp1 = load_disp(disp1_path)
    # depth1 = disp_2_depth(disp1, valid_disp1, focal_length_pixel)

    valid_disp1 = depth1>0
    pc1 = pixel2xyz(depth1, P_rect_left)
    # my_pc1 = project_image_to_rect(depth1, P_rect_left)

    valid_disp2 = depth3>0
    pc2 = pixel2xyz(depth3, P_rect_left)
    
    # valid_pc1 = pc1[final_mask]
    # # my_pc1 = pc1_my[final_mask]
    # valid_pc2 = pc2[final_mask]
    valid_pc1 = pc1[valid_disp1]
    # my_pc1 = pc1_my[final_mask]
    valid_pc2 = pc2[valid_disp2]
    save_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/PLin/pc_data/'
    # truenas_path = osp.join(save_path,'raw_points',split_trainval, '{:06d}'.format(idx))
    truenas_path = osp.join(save_path,'raw_points',split_trainval+'_all')
    os.makedirs(truenas_path, exist_ok=True)

    print('1',depth1.shape,valid_pc1.shape)
    print('3',depth3.shape,valid_pc2.shape)

    # visual(valid_pc1,my_pc1)

    np.save(osp.join(truenas_path,'{:06d}.npy'.format(idx)), valid_pc1)
    # np.save(osp.join(truenas_path, 'my_pc1.npy'), my_pc1)
    np.save(osp.join(truenas_path,'{:06d}.npy'.format(idx+2)), valid_pc2)
    # print('save ok !\n')
    # return depth1


def process_one_frame(split_trainval,idx,filename,depth1,depth3):
    """ Write image to pc.
        project 2D depth to 3D pc
    """

    # sidx = '{:06d}'.format(idx)
    calib_root = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calid'
    print('loading ',idx)
    indexstar= int(filename.find('2011_'))
    indexend= int(filename.find('_drive'))
    data_calid = filename[indexstar:indexend]
    calib_path = osp.join(calib_root,data_calid ,'calib_cam_to_cam.txt')
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    # disp1_path = osp.join(disp1_root, sidx + '_10.png')
    # disp1, valid_disp1 = load_disp(disp1_path)
    # depth1 = disp_2_depth(disp1, valid_disp1, focal_length_pixel)

    valid_disp1 = depth1>0
    pc1 = pixel2xyz(depth1, P_rect_left)
    # my_pc1 = project_image_to_rect(depth1, P_rect_left)

    valid_disp2 = depth3>0
    pc2 = pixel2xyz(depth3, P_rect_left)
    
    # valid_pc1 = pc1[final_mask]
    # # my_pc1 = pc1_my[final_mask]
    # valid_pc2 = pc2[final_mask]
    valid_pc1 = pc1[valid_disp1]
    # my_pc1 = pc1_my[final_mask]
    valid_pc2 = pc2[valid_disp2]

    truenas_path = osp.join(save_path,'raw_points',split_trainval+'_all', '{:06d}'.format(idx))
    os.makedirs(truenas_path, exist_ok=True)

    print('1',valid_pc1.shape)
    print('3',valid_pc2.shape)

    # visual(valid_pc1,my_pc1)

    np.save(osp.join(truenas_path,'pc1.npy'), valid_pc1)
    # np.save(osp.join(truenas_path, 'my_pc1.npy'), my_pc1)
    np.save(osp.join(truenas_path,'pc2.npy'), valid_pc2)
    # print('save ok !\n')
    return depth1

def process_one_pc(split_trainval,idx,filename,depth):
    """ Write pc to image.
    project 3D pc to 2D depth
    """
    # sidx = '{:06d}'.format(idx)
    calib_root = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calid'
    indexstar= int(filename.find('2011_'))
    indexend= int(filename.find('_drive'))
    data_calid = filename[indexstar:indexend]
    calib_path = osp.join(calib_root,data_calid ,'calib_cam_to_cam.txt')
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    if split_trainval == 'train' :
        path = osp.join(flow_pc_path,split_trainval+'_output_17500', 'pc1_t_{}.npy'.format(idx))
    else:
        # path = osp.join(flow_pc_path,split_trainval+'_output_17500', 'pc1_t_{}.npy'.format(idx))
        path = osp.join(flow_pc_path,split_trainval+'_all_output_17500', 'pc1_t_{}.npy'.format(idx))
    # path = osp.join(flow_pc_path,split_trainval+'_output_remove_road', 'pc1_t_{}.npy'.format(idx))

    pc = np.load(path).astype(np.float32)  #.astype(np.float32)
    depth = np.squeeze(depth)
    depth[:,:] = 0.0

    depth_prject = xyz2pixel(depth, P_rect_left,pc)
    depth_prject = np.expand_dims(depth_prject,-1)
    # print('project',depth_prject.shape)
    # truenas_path = osp.join(save_path,'raw_points',split_trainval, '{:06d}'.format(idx))
    # os.makedirs(truenas_path, exist_ok=True)
    # np.save(osp.join(truenas_path,'pc1.npy'), valid_pc1)
    # # np.save(osp.join(truenas_path, 'my_pc1.npy'), my_pc1)
    # np.save(osp.join(truenas_path,'pc2.npy'), valid_pc2)
    return depth_prject


### backup
# def pc_feature2image(xyz,feature,uv_feature):

    
#     calib_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calib_cam_to_cam.txt'
#     with open(calib_path) as fd:
#         lines = fd.readlines()
#         assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
#         P_rect = \
#             np.array([float(item) for item in
#                       [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
#                      dtype=np.float32).reshape(3, 4)

#     assert P_rect_left[0, 0] == P_rect_left[1, 1]

#     focal_length_pixel = P_rect[0, 0]
#     c_u=P_rect[0,2]
#     c_v=P_rect[1,2]
#     f_u=P_rect[0,0]
#     f_v=P_rect[1,1]
#     b_x=P_rect[0,3] / (-f_u)
#     b_y=P_rect[1,3] / (-f_v)



#     u = xyz[0,:,0]*f_u/xyz[0,:,2] + c_u
#     v = xyz[0,:,1]*f_v/xyz[0,:,2] + c_v
#     # d=xyz[0,:,2]
#     # u=np.rint(u).astype(int)
#     # v=np.rint(v).astype(int)
#     u = torch.round(u).type(torch.int64)
#     v = torch.round(v).type(torch.int64)
#     #style-1
#     uv_feature[0,:,v,u] = feature[0,:,:]
#     return uv_feature

def pc_feature2image(xyz,feature,uv_feature):

    # liu torch.Size([2, 10240, 3]) torch.Size([2, 64, 10240])
    # uv_feature torch.FloatTensor(2,64,256,1216)
    
    calib_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calib_cam_to_cam.txt'
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect[0, 0] == P_rect[1, 1]

    focal_length_pixel = P_rect[0, 0]
    c_u=P_rect[0,2]
    c_v=P_rect[1,2]
    f_u=P_rect[0,0]
    f_v=P_rect[1,1]
    b_x=P_rect[0,3] / (-f_u)
    b_y=P_rect[1,3] / (-f_v)



    u = xyz[:,:,0]*f_u/xyz[:,:,2] + c_u
    v = xyz[:,:,1]*f_v/xyz[:,:,2] + c_v

    u = torch.round(u).type(torch.int64)
    v = torch.round(v).type(torch.int64)
    # print(u.shape,v.shape,max(u),max(v)) #torch.Size([2, 10240]) torch.Size([2, 10240])
    # print(v[0].shape,v[0,:].shape,v[0,0:10])

    uv_feature[0,:,v[0],u[0]] = feature[0,:,:]
    uv_feature[1,:,v[1],u[1]] = feature[1,:,:]
    return uv_feature




if __name__ == '__main__':

    main()


