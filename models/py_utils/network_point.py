import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torch.nn.functional import interpolate
import math
import torch.nn.functional as F
import cv2
from config import system_configs
import time
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from .KLD import jd_loss, xy_wh_r_2_xy_sigma



import torchvision.transforms.functional as Ftt
from scipy import ndimage


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)



def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU6(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class Output_mask(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output_mask, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 1, 0, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs



class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, 
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                    nn.BatchNorm2d(n_filters),
                                    #nn.ReLU(inplace=True),)
                                    nn.PReLU(),)
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class network_point(nn.Module):
    def __init__(self, hidden_dim = system_configs.attn_dim):
        super(network_point, self).__init__()
        hidden_dim = system_configs.attn_dim

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.out_confidence = Output(hidden_dim*2, 2)     
        self.out_offset = Output(hidden_dim*2, 2)      
        self.out_scope = Output(hidden_dim*2, 1)
        self.out_length = Output(hidden_dim*2, 1)

        self.convlayer1 = BaseConv(hidden_dim*2, hidden_dim*2, 1, 1, act='relu')
        self.upsample1 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim*2, 2, stride=2)
        self.out_mask = Output_mask(hidden_dim, 1)
        self.C3P4_1 = CSPLayer(hidden_dim*2, hidden_dim, n=1, shortcut=True, depthwise=False, act="relu")
 
    def forward(self, encoded_feature, imgC):

        encoded_feature, p40 = encoded_feature
        encoded_feature = self.conv1(encoded_feature)

        out_mask = self.upsample1(encoded_feature)
        out_mask = self.C3P4_1(out_mask)
        out_mask = self.out_mask(out_mask)

        
        pooled_hs =  roi_align(encoded_feature, imgC, output_size=1, spatial_scale=1/32, sampling_ratio=128)
        
        out_confidence = self.out_confidence(pooled_hs)
   
        out_offset = self.out_offset(pooled_hs)
    
        out_scope = self.out_scope(pooled_hs)
        out_length = self.out_length(pooled_hs)

        result = [out_confidence, out_offset, out_scope, out_length, out_mask]

        return result



class loss_point_onlyNeg(nn.Module):
    def __init__(self):
        super(loss_point_onlyNeg, self).__init__()
   
    def get_neg_bbox(self, target_points, gt_nums_lane):

        

        image_index_for_boxes = []
        gt_nums_lane_for_boxes = [[i] * gt_nums_lane[i] * system_configs.neg_sampling_rate for i in range(len(gt_nums_lane))]
        for x in gt_nums_lane_for_boxes:
            for y in x:
                image_index_for_boxes.append(y)


 

        r = system_configs.roi_r
        centers = torch.rand(target_points.shape[0] * system_configs.neg_sampling_rate, 2) * 640
        boxes = torch.zeros((target_points.shape[0] * system_configs.neg_sampling_rate, 5))
        boxes[:, 0] = torch.Tensor(image_index_for_boxes)
        boxes[:, 1] = centers[:, 0] - r
        boxes[:, 2] = centers[:, 1] - r
        boxes[:, 3] = centers[:, 0] + r
        boxes[:, 4] = centers[:, 1] + r
        valid = (boxes[:,1] > 0) & (boxes[:,1] < 640) & (boxes[:,2] >= 0) & (boxes[:,2] < 640) & (boxes[:,3] >= 0) & (boxes[:,3] < 640) & (boxes[:,4] >= 0) & (boxes[:,4] < 640) 
        valid_boxes = boxes[valid]
        valid_centers = centers[valid]
        valid_index_per_img = [0 for i in range(1+int(valid_boxes[:, 0].max().item()))]
        for i in valid_boxes[:, 0]: 
            valid_index_per_img[int(i.item())] += 1
   

        valid_index_per_img = valid_index_per_img
        new_valid_index_per_img = []
        old = 0
        for index in valid_index_per_img:
            new_valid_index_per_img.append((old, old+index))
            old = old+index

        gt_points_per_img = target_points.split(gt_nums_lane)
        gt_points_per_img = [gt_points.view(-1, 2) for gt_points in gt_points_per_img] 
        gt_points_per_img = [x.shape[0] for x in gt_points_per_img]

       
        new_gt_points_per_img = []
        old = 0
        for index in gt_points_per_img:
            new_gt_points_per_img.append((old, old+index))
            old = old+index


        target_points = target_points.view(-1, 2).unsqueeze(0).repeat(valid_centers.shape[0], 1, 1)
        valid_centers = valid_centers.unsqueeze(1).repeat(1, target_points.shape[1], 1)
        masks = torch.zeros(valid_centers.shape).bool()
        log = []
        for i, j in zip(new_valid_index_per_img, new_gt_points_per_img):
            masks[i[0]:i[1], j[0]:j[1], :] = True
            log += [j[1]-j[0] for _ in range(i[1]-i[0])]
        dist = torch.norm((640*target_points[masks[:,:,0]]-valid_centers[masks[:,:,0]]), p=2, dim=-1)
        dist = dist.split(log)
        min_dist = [x.min(-1)[0] for x in dist]
        selected = []
        for index, value in enumerate(min_dist):
            if 130>value > (math.sqrt(2)*r):
                selected.append(index)
        
        imgC_Neg = torch.index_select(valid_boxes, 0, torch.Tensor(selected).int())
        del dist
        return imgC_Neg.cuda()

     

    def get_gt(self, tgt, imgC, valid_position, gt_idx):
        
        tgt = [y[0] for y in tgt[1:]]
        gt_nums_lane = [y.shape[0] for y in tgt]
        tgt = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)

  
        target_points = tgt[:,5:] 

        target_xs = target_points[:, :target_points.shape[1] // 2]
        target_ys = target_points[:, target_points.shape[1] // 2:]
        assert ((target_ys>-1) != (target_xs>-1)).sum().item() == 0
        points_num = torch.where(target_xs>-100, 1, 0).sum(-1)
        target_xs = target_xs[target_xs>-1]
        target_xs = target_xs.split(points_num.tolist()) # points for each lane
        target_ys = target_ys[target_ys>-1]
        target_ys = target_ys.split(points_num.tolist()) # points for each lane

        target_points = torch.ones((len(target_xs), 830, 2)) 
        target_points_sparse = torch.ones((len(target_xs), 180, 2)) 

        for index, line_x in enumerate(target_xs):
            line_x = line_x.unsqueeze(0).unsqueeze(0)
            line_x_dense = interpolate(line_x, size=830, mode='linear', align_corners=True)
            line_x_dense = line_x_dense[0,0,:]
            target_points[index, :, 0] = line_x_dense[:]

            line_x_sparse = interpolate(line_x, size=180, mode='linear', align_corners=True)
            line_x_sparse = line_x_sparse[0,0,:]
            target_points_sparse[index, :, 0] = line_x_sparse[:]


            
        for index, line_y in enumerate(target_ys):
            line_y = line_y.unsqueeze(0).unsqueeze(0)
            line_y_dense = interpolate(line_y, size=830, mode='linear', align_corners=True)
            line_y_dense = line_y_dense[0,0,:]
            target_points[index, :, 1] = line_y_dense[:]

            line_y_sparse = interpolate(line_y, size=180, mode='linear', align_corners=True)
            line_y_sparse = line_y_sparse[0,0,:]
            target_points_sparse[index, :, 1] = line_y_sparse[:]
        # target_points: for each GT lane, 830 interpolated points x 2 coord

   
        return target_points_sparse, gt_nums_lane   
    

    def forward(self, imgC, xs, ys, valid_position, gt_idx, encoded_feature, Net):
        loss = 0
        target_points, gt_nums_lane = self.get_gt(ys, imgC, valid_position, gt_idx)
        
        imgC_Neg = self.get_neg_bbox(target_points, gt_nums_lane)

        point_out_Neg = Net(encoded_feature, imgC_Neg)
        pr_confidence_Neg = point_out_Neg[0][:,:,0,0]
        gt_confidence_Neg = torch.zeros(pr_confidence_Neg.shape[0]).cuda()
        loss_ce_Neg = F.cross_entropy(pr_confidence_Neg.unsqueeze(0).permute(0, 2, 1), gt_confidence_Neg.long().unsqueeze(-1).permute(1,0))
        print('[POINT] loss_ce_Neg:', loss_ce_Neg.item())
        loss += loss_ce_Neg
  
        return loss, []




class loss_point(nn.Module):
    def __init__(self):
        super(loss_point, self).__init__()

    def get_gt(self, tgt, imgC, valid_position, gt_idx):
        
        # get gt_lanes_points
        tgt = [y[0] for y in tgt[1:]]
        gt_nums_lane = [y.shape[0] for y in tgt]
        tgt = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        target_points = tgt[:,5:].cuda() 
        target_xs = target_points[:, :target_points.shape[1] // 2]
        target_ys = target_points[:, target_points.shape[1] // 2:]
        assert ((target_ys>-1) != (target_xs>-1)).sum().item() == 0
        points_num = torch.where(target_xs>-100, 1, 0).sum(-1)
        target_xs = target_xs[target_xs>-1]
        target_xs = target_xs.split(points_num.tolist()) # points for each lane
        target_ys = target_ys[target_ys>-1]
        target_ys = target_ys.split(points_num.tolist()) # points for each lane
        target_points = torch.ones((len(target_xs), 830, 2)).cuda() 
        target_points_sparse = torch.ones((len(target_xs), 180, 2)).cuda() 
        for index, line_x in enumerate(target_xs):
            line_x = line_x.unsqueeze(0).unsqueeze(0)
            line_x_dense = interpolate(line_x, size=830, mode='linear', align_corners=True)
            line_x_dense = line_x_dense[0,0,:]
            target_points[index, :, 0] = line_x_dense[:]
            line_x_sparse = interpolate(line_x, size=180, mode='linear', align_corners=True)
            line_x_sparse = line_x_sparse[0,0,:]
            target_points_sparse[index, :, 0] = line_x_sparse[:]
        for index, line_y in enumerate(target_ys):
            line_y = line_y.unsqueeze(0).unsqueeze(0)
            line_y_dense = interpolate(line_y, size=830, mode='linear', align_corners=True)
            line_y_dense = line_y_dense[0,0,:]
            target_points[index, :, 1] = line_y_dense[:]
            line_y_sparse = interpolate(line_y, size=180, mode='linear', align_corners=True)
            line_y_sparse = line_y_sparse[0,0,:]
            target_points_sparse[index, :, 1] = line_y_sparse[:]
        # target_points: for each GT lane, 830 interpolated points x 2 coord


        gt_IDTmasks = self.get_gt_mask(target_points, gt_nums_lane)

        
        imgBox = imgC[:, 1:] # for each predicted boxes, 4 coord
        imgCenter = torch.zeros(imgBox.shape[0], 2).cuda()
        imgCenter[:, 0] = imgBox[:, 0] / 2 + imgBox[:, 2] / 2
        imgCenter[:, 1] = imgBox[:, 1] / 2 + imgBox[:, 3] / 2
        imgCenter = imgCenter.unsqueeze(1).repeat(1, 830, 1)
        
        imgCln = imgC[:, 0]
        newimgCln = []
      
        for index in range(int(imgCln.max().item())+1):
            cnt = 0
            for l in imgCln.tolist():
                if l == index:
                    cnt += 1
            newimgCln.append(cnt)
     
        imgCenter_imgs = imgCenter[:, 0, :].split(newimgCln)
        
        newtgtln = []
        for index in range(int(gt_idx[0].max().item())+1):
            cnt = 0
            for l in gt_idx[0].tolist():
                if l == index:
                    cnt += 1
            newtgtln.append(cnt)
        target_points_imgs = target_points.split(newtgtln)

        
        
        r = system_configs.roi_r
        GT_confidence, GT_offset, GT_angle, GT_length = [], [], [], []
        
        for index, (imgCenter_img, target_points_img) in enumerate(zip(imgCenter_imgs, target_points_imgs)):
            target_points_img = target_points_img.view(-1, 2).unsqueeze(0).repeat(imgCenter_img.shape[0], 1, 1) * 640
            if target_points_img.shape[0] == 0:
                continue
            imgCenter_img = imgCenter_img.unsqueeze(1).repeat(1, target_points_img.shape[1], 1)
            dist = torch.norm(target_points_img-imgCenter_img, p=2, dim=-1)
            min_dist_each_row = dist.min(-1)[0]
            min_indices = dist.min(-1)[1]
            min_indices = torch.clamp(min_indices, min = 1, max = target_points_img.shape[1]-3)
            min_indices_index = min_indices.t().unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)
            min_points = torch.gather(target_points_img, dim=1, index = min_indices_index)
            gt_confidence =  (torch.abs(min_points[:,0,0] - imgCenter_img[:,0,0]) < r) & (torch.abs(min_points[:,0,1] - imgCenter_img[:,0,1]) < r)
            gt_offset = min_points.squeeze(1)
            image_lt = imgCenter_img[:,0,:] - r
            offset = (gt_offset - image_lt) / r / 2


            dist_inside_mask = (torch.abs(target_points_img[:,:,0] - imgCenter_img[:,:,0]) < r) & (torch.abs(target_points_img[:,:,1] - imgCenter_img[:,:,1]) < r)
            dist_inside = (dist * dist_inside_mask)
            max_dist = dist_inside.max(-1)[0] # TODO: inside box
            min_dist = min_dist_each_row
            gt_length = torch.sqrt(max_dist**2 - min_dist**2)



            min_indices_before = min_indices - 1
            min_indices_index_before = min_indices_before.t().unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)
            min_points_before = torch.gather(target_points_img, dim=1, index = min_indices_index_before).squeeze(1)

            min_indices_after = min_indices + 1
            min_indices_index_after = min_indices_after.t().unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)
            min_points_after = torch.gather(target_points_img, dim=1, index = min_indices_index_after).squeeze(1)

            gt_scope = (min_points_after[:, 1] - gt_offset[:, 1]) / (min_points_after[:, 0] - gt_offset[:, 0] + 1e-12)
            gt_scope += (gt_offset[:, 1] - min_points_before[:, 1]) / (gt_offset[:, 0] - min_points_before[:, 0] + 1e-12)
            gt_scope /= 2.0
            gt_angle = torch.atan(gt_scope)
      
            GT_confidence.append(gt_confidence)
            GT_offset.append(offset)
            GT_angle.append(gt_angle)
            GT_length.append(gt_length)
        GT_confidence = torch.cat(GT_confidence, 0)
        GT_offset = torch.cat(GT_offset, 0)
        GT_angle = torch.cat(GT_angle, 0)
        GT_length = torch.cat(GT_length, 0)
        

        return GT_confidence, GT_offset, GT_angle, target_points_sparse, gt_nums_lane, GT_length, gt_IDTmasks   # TODO
    
    def get_gt_mask(self, gt_points, gt_nums_lane):
  
        gt_points = torch.clamp((gt_points * 640).floor(), 0, 639)
        gt_masks = torch.zeros(len(gt_nums_lane), 640, 640).cuda()
        gt_points = gt_points.split(gt_nums_lane)
        for index, img_gt_points in enumerate(gt_points):
            img_gt_points = img_gt_points.view(-1, 2)
            
           
            img_gt_points = torch.unique_consecutive(img_gt_points, dim=0).long()
    
            gt_masks[index, img_gt_points[:, 1].tolist(), img_gt_points[:, 0].tolist()] = True
            
      

        gt_masksc = gt_masks.cpu().numpy()
        gt_IDTmasks = torch.zeros(len(gt_nums_lane), 640//16, 640//16).cuda()
        for index, gt_mask in enumerate(gt_masksc):
            gt_IDTmask = 18 - np.clip(ndimage.distance_transform_edt(1 - gt_mask),0,18)
            gt_IDTmask = np.expand_dims(gt_IDTmask, -1)
            gt_IDTmask = cv2.resize(gt_IDTmask, (640//16, 640//16))
            # inspect mask
            # if index == 0:
            #     cv2.imshow('a', gt_IDTmask)
            #     cv2.waitKey()
            gt_IDTmasks[index] = torch.Tensor(gt_IDTmask).cuda()
    
        return gt_IDTmasks


    def get_neg_bbox(self, target_points, gt_nums_lane):

        

        image_index_for_boxes = []
        gt_nums_lane_for_boxes = [[i] * gt_nums_lane[i] * system_configs.neg_sampling_rate for i in range(len(gt_nums_lane))]
        for x in gt_nums_lane_for_boxes:
            for y in x:
                image_index_for_boxes.append(y)

        r = system_configs.roi_r
 
        Centers = []
        splice = [len(a)//system_configs.neg_sampling_rate for a in gt_nums_lane_for_boxes]
        Target_points = target_points.split(splice)
        for target_point in Target_points:
            margin_x = torch.min((1-target_point[:, :, 0].max()), target_point[:, :, 0].min()) * 640 - 30
            centers_x = (torch.rand(target_point.shape[0] * system_configs.neg_sampling_rate, 1).cuda()) * (640 - 2 * margin_x.item()) + margin_x.item() 
            margin_y = torch.min((1-target_point[:, :, 1].max()), target_point[:, :, 1].min()) * 640 - 30
            centers_y = (torch.rand(target_point.shape[0] * system_configs.neg_sampling_rate, 1).cuda()) * (640 - 2 * margin_y.item()) + margin_y.item() 
            centers = torch.stack((centers_x, centers_y), -1)[:, 0, :]
            Centers.append(centers)
        centers = torch.cat(Centers, 0)

        boxes = torch.zeros((target_points.shape[0] * system_configs.neg_sampling_rate, 5)).cuda()
        boxes[:, 0] = torch.Tensor(image_index_for_boxes)
        boxes[:, 1] = centers[:, 0] - r
        boxes[:, 2] = centers[:, 1] - r
        boxes[:, 3] = centers[:, 0] + r
        boxes[:, 4] = centers[:, 1] + r
        valid = (boxes[:,1] > 0) & (boxes[:,1] < 640) & (boxes[:,2] >= 0) & (boxes[:,2] < 640) & (boxes[:,3] >= 0) & (boxes[:,3] < 640) & (boxes[:,4] >= 0) & (boxes[:,4] < 640) 
        valid_boxes = boxes[valid]
        valid_centers = centers[valid]
        valid_index_per_img = [0 for i in range(1+int(valid_boxes[:, 0].max().item()))]
        for i in valid_boxes[:, 0]: 
            valid_index_per_img[int(i.item())] += 1


        
        valid_centers_per_img = valid_centers.split(valid_index_per_img)
        boxes_per_img = valid_boxes.split(valid_index_per_img)
        gt_points_per_img = target_points.split(gt_nums_lane)
        gt_points_per_img = [gt_points.view(-1, 2) for gt_points in gt_points_per_img] 
        
        imgC_Neg = []
        for centers, tgts, boxes in zip(valid_centers_per_img, gt_points_per_img, boxes_per_img):
            centers = centers.cuda().unsqueeze(1).repeat(1, tgts.shape[0], 1)
            tgts = tgts.cuda().unsqueeze(0).repeat(centers.shape[0], 1, 1) * 640
            dist = torch.norm(centers-tgts, p=2, dim=-1)
            min_dist_each_row = dist.min(-1)[0] > (math.sqrt(2)*r + 1)
            neg_box_per_img = boxes[min_dist_each_row]
            imgC_Neg.append(neg_box_per_img)
        imgC_Neg = torch.cat(imgC_Neg, 0)

        return imgC_Neg.cuda()



    def forward(self, point_out, imgC, xs, ys, valid_position, gt_idx, encoded_feature, Net):
        loss = 0
        pr_confidence, pr_offset, pr_theta, pr_length, pr_mask = point_out
        pr_confidence, pr_offset, pr_theta, pr_length, pr_mask = pr_confidence[:,:,0,0], pr_offset[:,:,0,0], pr_theta[:,0,0,0], pr_length[:,0,0,0], pr_mask[:, 0, :, :]
        gt_confidence, gt_offset, gt_theta, target_points, gt_nums_lane, gt_length, gt_IDTmasks = self.get_gt(ys, imgC, valid_position, gt_idx)
        gt_confidence, gt_offset, gt_theta, target_points, gt_length = gt_confidence.cuda(), gt_offset.cuda(), gt_theta.cuda(), target_points.cuda(), gt_length.cuda()
        
        cur = 0
        cnt = 0
        splice = []
        for i in valid_position[0]:
            if i == cur:
                cnt += 1
            else:
                splice.append(cnt)
                cnt = 1
                cur += 1
        splice.append(cnt)

        
        imgC_Neg = self.get_neg_bbox(target_points, gt_nums_lane)
        loss_seg = F.l1_loss(pr_mask, gt_IDTmasks)*0.6
        loss += loss_seg
        print("[POINT]loss_seg:"+ str(loss_seg.item()))


        point_out_Neg = Net(encoded_feature, imgC_Neg)
        pr_confidence_Neg = point_out_Neg[0][:,:,0,0]
        gt_confidence_Neg = torch.zeros(pr_confidence_Neg.shape[0]).cuda()
        
        loss_ce = 1 * F.cross_entropy(pr_confidence.unsqueeze(0).permute(0, 2, 1), gt_confidence.cuda().long().unsqueeze(-1).permute(1,0))
        pr_confidence_focal = torch.cat((pr_confidence, pr_confidence_Neg))
        gt_confidence_focal = torch.cat((gt_confidence, gt_confidence_Neg))
        loss_ce_focal = 25 * sigmoid_focal_loss(torch.max(pr_confidence_focal, -1)[0], gt_confidence_focal.float().cuda(), alpha=0.8, reduction='mean')
        loss += loss_ce

        loss += loss_ce_focal
        print("[POINT]loss_ce:"+ str(loss_ce.item())+ "    loss_focal:" +  str(loss_ce_focal.item()))
        print("[POINT]loss_focal:" +  str(loss_ce_focal.item()))
        if gt_confidence.sum().item():
            loss_offset = 40 * F.mse_loss(gt_offset[gt_confidence].cuda(), pr_offset[gt_confidence])
            loss_theta = 1 * F.l1_loss(gt_theta[gt_confidence].cuda(), pr_theta[gt_confidence])
            loss_length =  F.l1_loss(gt_length[gt_confidence].cuda() / system_configs.roi_r, pr_length[gt_confidence])
            loss += loss_offset
            loss += loss_theta
            loss += loss_length

            xywhr_pr = torch.stack((pr_offset[gt_confidence][:, 0], pr_offset[gt_confidence][:, 1], pr_length[gt_confidence], pr_length[gt_confidence]/4, pr_theta[gt_confidence]), -1)
            xywhr_gt = torch.stack((gt_offset[gt_confidence][:, 0], gt_offset[gt_confidence][:, 1], gt_length[gt_confidence], gt_length[gt_confidence]/4, gt_theta[gt_confidence]), -1)
            GT = xy_wh_r_2_xy_sigma(xywhr_gt)
            PR = xy_wh_r_2_xy_sigma(xywhr_pr)
            loss_kld = 0.5 * torch.mean(jd_loss(PR, GT))
            loss += loss_kld

            print("[POINT]loss_offset:"+ str(loss_offset.item())+ "    loss_theta:" +  str(loss_theta.item())+ "    loss_length:" +  str(loss_length.item())+ "    loss_kld:" +  str(loss_kld.item()))
        else:
            print('all sampled imgC are Neg')
        
        if gt_confidence.sum().item():
            imgC_splice = imgC.split(splice)
            gt_confidence_splice = gt_confidence.split(splice)
            pr_offset_splice = pr_offset.split(splice)
            pr_theta_splice = pr_theta.split(splice)

            loss_smooth_theta = []
            loss_smooth_offset = []
           
            for imgC_splice_lane, gt_confidence_splice_lane, pr_offset_splice_lane, pr_theta_splice_lane in zip(imgC_splice, gt_confidence_splice, pr_offset_splice, pr_theta_splice):
                if gt_confidence_splice_lane.sum().item()>1:
                    pr_pos_xy = (imgC_splice_lane[:, 1:3] + system_configs.roi_r * 2 * pr_offset_splice_lane)[gt_confidence_splice_lane]
                    dist_adjasant_point = torch.norm(pr_pos_xy[1:, :] - pr_pos_xy[:-1, :], 2,-1)      
                    dist_weight = dist_adjasant_point / (system_configs.roi_r * 2 )


                    pr_theta_splice_lane = pr_theta_splice_lane[gt_confidence_splice_lane]
                    dist_theta_point_change = (pr_theta_splice_lane[1:] - pr_theta_splice_lane[:-1]) / (dist_weight+1e-12)
                    dist_theta_point_change_change = abs(dist_theta_point_change[1:] - dist_theta_point_change[:-1])
                    if dist_theta_point_change_change.shape[0]:
                        dist_theta_point_loss = dist_theta_point_change_change.mean() / 5
                        loss_smooth_theta.append(dist_theta_point_loss)
                    
                    


                    dist_point_change = (pr_pos_xy[1:, :] - pr_pos_xy[:-1, :]) / (dist_weight.unsqueeze(-1).repeat(1,2)+1e-12)
                    dist_point_change_change = abs(dist_point_change[1:, :] - dist_point_change[:-1, :])
                    if dist_point_change_change.shape[0]:
                        dist_point_loss = dist_point_change_change.mean() / 5
                        loss_smooth_offset.append(dist_point_loss)


                    

                    
                    

            loss_smooth_theta = (sum(loss_smooth_theta) / len(loss_smooth_theta))
            loss_smooth_offset = sum(loss_smooth_offset) / len(loss_smooth_offset)
            # loss += 0.35 * loss_smooth_theta
            loss += loss_smooth_offset

   
            # loss += loss_continuity
            if isinstance(loss_smooth_theta, int) or isinstance(loss_smooth_offset, int):
                pass
            else:
                print("[POINT]loss_smooth_theta:"+ str(loss_smooth_theta.item()) + "    loss_smooth_offset:"+ str(loss_smooth_offset.item()))
            



     
        return loss, gt_confidence.cuda().sum()