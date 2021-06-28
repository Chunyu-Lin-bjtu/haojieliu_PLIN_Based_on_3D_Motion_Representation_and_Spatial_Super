import torch
import torch.nn as nn
import numpy as np
from chamferdist.chamferdist import ChamferDistance
loss_names = ['l1', 'l2']

chamferDist = ChamferDistance()

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

    
    height, width = depth.shape[2:]
    if px is None:
        px = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1))
    if py is None:
        py = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    px = torch.from_numpy(px).cuda()
    py = torch.from_numpy(py).cuda()

    const_x = c_u * depth + P_rect[0,3]
    const_y = c_v * depth + P_rect[1,3]
    x = ((px * (depth + P_rect[2,3]) - const_x) / f_u) [:,:,:, :, None]
    y = ((py * (depth + P_rect[2,3]) - const_y) / f_v) [:,:,:, :, None]
    pc = torch.cat((x, y, depth[:,:,:, :, None]), dim=-1)
    return pc
def process_one_frame(pred,target):

    calib_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calib_cam_to_cam.txt'
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    valid_mask = (target>0).detach()
    P_rect_left = torch.from_numpy(P_rect_left)
    pc1 = pixel2xyz(target, P_rect_left)
    pc2 = pixel2xyz(pred, P_rect_left)

    pc1_0 = pc1[0][valid_mask[0]].unsqueeze(0)
    # pc1_1 = pc1[1][valid_mask[1]].unsqueeze(0)
    ## print('pc1_0.shape',pc1_0.shape,pc1_1.shape) #pc1_0.shape torch.Size([1,88575, 3]) torch.Size([1,90074, 3])
    pc2_0 = pc2[0][valid_mask[0]].unsqueeze(0)
    # pc2_1 = pc2[1][valid_mask[1]].unsqueeze(0)
   
    # return pc1_0, pc2_0, pc1_1, pc2_1
    return pc1_0, pc2_0
def process_one_frame_(pred,target):

    calib_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calib_cam_to_cam.txt'
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    valid_mask = (target>0).detach()
    P_rect_left = torch.from_numpy(P_rect_left)
    pc1 = pixel2xyz(target, P_rect_left)
    pc2 = pixel2xyz(pred, P_rect_left)

    pc1_0 = pc1[0][valid_mask[0]]
    # pc1_1 = pc1[1][valid_mask[1]].unsqueeze(0)
    ## print('pc1_0.shape',pc1_0.shape,pc1_1.shape) #pc1_0.shape torch.Size([1,88575, 3]) torch.Size([1,90074, 3])
    pc2_0 = pc2[0][valid_mask[0]]
    # pc2_1 = pc2[1][valid_mask[1]].unsqueeze(0)
   
    # return pc1_0, pc2_0, pc1_1, pc2_1
    return pc1_0, pc2_0
# class CD_loss(nn.Module):
#     def __init__(self):
#         super(CD_loss, self).__init__()

#     def forward(self, pred, target):
#         assert pred.dim() == target.dim(), "inconsistent dimensions"
#         pc1_0, pc2_0, pc1_1, pc2_1 = process_one_frame(pred,target) #492.3053
#         dist12_0, dist21_0, idx1, idx2 = chamferDist(pc1_0, pc2_0)
#         dist12_1, dist21_1, idx1, idx2 = chamferDist(pc1_1, pc2_1)
#         self.loss = 0.5 * (dist12_0.mean() + dist21_0.mean()) + 0.5 * (dist12_1.mean() + dist21_1.mean())
#         # print(self.loss) ##492.3053
#         return self.loss
class CD_loss(nn.Module):
    def __init__(self):
        super(CD_loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        pc1_0, pc2_0= process_one_frame(pred,target) #492.3053
        dist12_0, dist21_0, idx1, idx2 = chamferDist(pc1_0, pc2_0)
        # dist12_1, dist21_1, idx1, idx2 = chamferDist(pc1_1, pc2_1)
        self.loss =  dist12_0.mean() + dist21_0.mean()
        # print(self.loss) ##492.3053
        return self.loss

def process_one_frame_train(pred,target):

    calib_path = '/media/lhj/693746ff-7ed5-42d2-b9d6-3e7b290b0533/DocuClassLin/LHJ/self-supervised-depth-completion-master/dataloaders/calib_cam_to_cam.txt'
    with open(calib_path) as fd:
        lines = fd.readlines()
        assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
        P_rect_left = \
            np.array([float(item) for item in
                      [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                     dtype=np.float32).reshape(3, 4)

    assert P_rect_left[0, 0] == P_rect_left[1, 1]
    focal_length_pixel = P_rect_left[0, 0]

    valid_mask = (target>0).detach()
    P_rect_left = torch.from_numpy(P_rect_left)
    pc1 = pixel2xyz(target, P_rect_left)
    pc2 = pixel2xyz(pred, P_rect_left)

    pc1_0 = pc1[0][valid_mask[0]].unsqueeze(0)
    pc1_1 = pc1[1][valid_mask[1]].unsqueeze(0)
    ## print('pc1_0.shape',pc1_0.shape,pc1_1.shape) #pc1_0.shape torch.Size([1,88575, 3]) torch.Size([1,90074, 3])
    pc2_0 = pc2[0][valid_mask[0]].unsqueeze(0)
    pc2_1 = pc2[1][valid_mask[1]].unsqueeze(0)
   
    return pc1_0, pc2_0, pc1_1, pc2_1
    # return pc1_0, pc2_0

class CD_loss_train(nn.Module):
    def __init__(self):
        super(CD_loss_train, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        pc1_0, pc2_0, pc1_1, pc2_1= process_one_frame_train(pred,target) #492.3053
        dist12_0, dist21_0, idx1, idx2 = chamferDist(pc1_0, pc2_0)
        dist12_1, dist21_1, idx1, idx2 = chamferDist(pc1_1, pc2_1)
        self.loss =  dist12_0.mean() + dist21_0.mean() +dist12_1.mean() + dist21_1.mean()
        # print(self.loss) ##492.3053
        return self.loss
class bc(nn.Module):
    def __init__(self):
        super(bc, self).__init__()

    def forward(self, pred, target):

        assert pred.dim() == target.dim(), "inconsistent dimensions"
        target = target
        valid_mask = (target>0).detach()
        diff = target - pred*255.0
        # diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss



class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()

        return self.loss

class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, target, recon, mask=None):

        assert recon.dim()==4, "expected recon dimension to be 4, but instead got {}.".format(recon.dim())
        assert target.dim()==4, "expected target dimension to be 4, but instead got {}.".format(target.dim())
        assert recon.size()==target.size(), "expected recon and target to have the same size, but got {} and {} instead"\
            .format(recon.size(), target.size())
        diff = (target - recon).abs()
        diff = torch.sum(diff, 1) # sum along the color channel

        # compare only pixels that are not black
        valid_mask = (torch.sum(recon, 1)>0).float() * (torch.sum(target, 1)>0).float()
        if mask is not None:
            valid_mask = valid_mask * torch.squeeze(mask).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                print("warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size).")
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim() == 4, "expected 4-dimensional data, but instead got {}".format(x.dim())
            horizontal = 2 * x[:,:,1:-1,1:-1] - x[:,:,1:-1,:-2] - x[:,:,1:-1,2:]
            vertical = 2 * x[:,:,1:-1,1:-1] - x[:,:,:-2,1:-1] - x[:,:,2:,1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()
        self.loss = second_derivative(depth)
        return self.loss