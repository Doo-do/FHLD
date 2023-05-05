# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import random

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)

# SBC_colors = [ORANGE, RED, CYAN, DARK_GREEN, GREEN, BLUE, YELLOW, PURPLE, PINK]
SBC_colors = [ORANGE, ORANGE, ORANGE, RED, RED, RED, CYAN, CYAN, CYAN]

KPS_colors = [DARK_GREEN, DARK_GREEN, YELLOW, YELLOW, PINK]

def save_batch_image_with_curves(batch_image,
                                batch_curves,
                                batch_labels,
                                file_name,
                                nrow=2,
                                padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            boxes = batch_curves[k]
            labels = batch_labels[k]
            num_box = boxes.shape[0]
            i = 0
            for n in range(num_box):
                lane = boxes[:, 5:][n]
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:]
                ys = ys[xs >= 0] * H
                xs = xs[xs >= 0] * W
                cls = labels[n]
                if (cls > 0 and cls < 10):
                    for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                        j_x = x * width + padding + xcoord
                        j_y = y * height + padding + ycoord
                        cv2.circle(ndarr, (int(j_x), int(j_y)), 2, PINK, 10)
                    i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_dbs(batch_image,
                              batch_curves,
                              batch_labels,
                              file_name,
                              nrow=2,
                              padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    ofile_name = file_name[:-4]
    B, C, H, W = batch_image.size()
    images = batch_image.mul(255).clamp(0, 255).permute(0,2,3,1).cpu().numpy()


    for k in range(images.shape[0]):
        file_name = ofile_name + '_' + str(k) + '.jpg'
        image = images[k].copy().astype(np.uint8)
        pred = batch_curves[k].cpu().numpy()  # 10 7
        labels = batch_labels[k].cpu().numpy()  # 10
        pred = pred[labels == 1] # only draw lanes
        num_pred = pred.shape[0]
        if num_pred > 0:
            for n, lane in enumerate(pred):

                color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
                x1 = lane[0]
                y1 = lane[1]
                x2 = lane[2]
                y2 = lane[3]
                a3 = lane[4]
                a2 = lane[5]
                b3 = lane[6]
                b2 = lane[7]
                a1 = x2 - a3 - a2 - x1
                a0 = x1
                b1 = y2 - b3 - b2 - y1
                b0 = y1

                p1 = (x1, y1)
                p2 = (x2, y2)
                lamda = np.array([i/100 for i in range(0, 100, 2)])
                xs = a3 * lamda**3 + a2 * lamda**2 + a1 * lamda + a0
                ys = b3 * lamda**3 + b2 * lamda**2 + b1 * lamda + b0
                points = np.stack((xs*image.shape[0], ys*image.shape[1]), -1) 
                
                cv2.circle(np.array(image), (int(p1[0]*image.shape[0]), int(p1[1]*image.shape[1])), 3, color, -1)
                cv2.circle(image, (int(p2[0]*image.shape[0]), int(p2[1]*image.shape[1])), 3, color, -1)
                
                for point in points:
                    cv2.circle(image, (int(point[0]), int(point[1])), 0, color, 0)

                    
        cv2.imwrite(file_name, image)

def save_debug_images_boxes(input, tgt_curves, tgt_labels,
                            pred_curves, pred_labels, prefix=None):
    # save_batch_image_with_curves(
    #     input, tgt_curves, tgt_labels,
    #     '{}_gt.jpg'.format(prefix)
    # )

    save_batch_image_with_dbs(
        input, pred_curves, pred_labels,
        '{}_pred.jpg'.format(prefix)
    )

