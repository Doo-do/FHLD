#!/usr/bin/env python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.nn.functional as F
import json
import torch
import pprint
import argparse
import cv2
import importlib
import sys
import numpy as np
from nnet.py_factory import NetworkFactory
import matplotlib
import shutil
import time
from models.py_utils.network_point import network_point
from config import system_configs


matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
from db.utils.evaluator import Evaluator
from tqdm import tqdm
torch.backends.cudnn.benchmark = False
import random
random.seed(13)


def clamp(min, max, para):
    if para >= min and para <= max:
        return para
    elif para > max:
        return max
    else:
        return min



RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)
def normalize_(image, mean, std):
    image -= mean
    image /= std

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")

    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--modality", dest="modality",
                        default=None, type=str)
    parser.add_argument("--image_root", dest="image_root",
                        default="/data/duduguan/LSTR-Custom-Base (copy 1)/raws", type=str)
    parser.add_argument("--batch", dest='batch',
                        help="select a value to maximum your FPS",
                        default=1, type=int)
    parser.add_argument("--debugEnc", action="store_true")
    parser.add_argument("--debugDec", action="store_true")
    args = parser.parse_args()
    return args


def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def kp_decode(nnet,
              images,):

    # print(images.shape,inmasks.shape)
    # checkpoints = system_configs.snapshot_file.format(280000)
    # print(checkpoints)
    # model = torch.load(checkpoints)

    # exit()
    out = nnet.test(images)
    # out_joints = out['pred_boxes'].cpu()
    return out
def inference(test_dir,model_path,result_dir,decode_func=kp_decode):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    make_dirs([result_dir])
    # test_iter = system_configs.max_iter if testiter is None else testiter
    # print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    nnet = NetworkFactory()




    print("loading parameters...")
    color = [RED, GREEN, DARK_GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, PINK]
    nnet.load_params(model_path)
    nnet.cuda()
    nnet.network.eval()
    nnet.network_point.eval()
    input_size = (640, 640)

    files_train = [os.path.join('test_images', p) for p in os.listdir(os.path.join(test_dir, 'test_images')) if os.path.exists(os.path.join(test_dir, 'test_labels', p[:-4]+'.txt'))] 
    files_test = [os.path.join('train_images', p) for p in os.listdir(os.path.join(test_dir, 'train_images')) if os.path.exists(os.path.join(test_dir, 'train_labels', p[:-4]+'.txt'))]
    random.shuffle(files_train)
    random.shuffle(files_test)
    files =  files_test[:350] + files_train[:150]
    for file in tqdm(files):
        
        image         = cv2.imread(os.path.join(test_dir, file))
        orimage = cv2.resize(image, (640, 640))
        height, width = image.shape[0:2] # 800 x 800

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)

        pad_image     = image.copy() # 800 x 800
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        check_image = resized_image.copy()

        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))


        _mean = np.array([0.04129752, 0.07141293, 0.18842927], dtype=np.float32)
        _std = np.array([0.10884721, 0.09943989, 0.19230546], dtype=np.float32)
        

        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255. # 640 x 640
        normalize_(resized_image, _mean, _std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
    
        outputs, weights, encoded_feature      = nnet.test([images, masks])
        pred_labels = outputs['pred_logits'].detach()
  
        prob = F.softmax(pred_labels, -1)
        scores, batch_labels = prob.max(-1)  # 4 10

        pred_curves = outputs['pred_curves'].detach()
        batch_curves = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)
        pred1 = batch_curves[0].cpu().numpy()  # 10 7
        labels = batch_labels[0].cpu().numpy()  # 10

        pred_valid = pred1[labels>0]

        

            
        Boxes = []
        for n, lane in enumerate(pred_valid):
            lane = lane[1:]




            ############

            
            color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            color2 = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
           
            x1 = lane[2]
            y1 = lane[0]
            x2 = lane[3]
            y2 = lane[1]
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
            points = np.stack((xs*640, ys*640), -1) 
            cv2.circle(orimage, (int(clamp(0,1,p1[0])*640), int(clamp(0,1,p1[1])*640)), 4, color, -1)
            cv2.circle(orimage, (int(clamp(0,1,p2[0])*640), int(clamp(0,1,p2[1])*640)), 4, color, -1)
            for point in points:
                point_x, point_y = int(point[0]), int(point[1])
                cv2.circle(orimage, (point_x, point_y), 1, (66, 66, 66), -1)
        
            lamda_box = np.array([i/100 for i in range(0, 100, 10)])
            points = np.zeros((len(lamda_box), 2))
            points[:, 0] = ((a3 * lamda_box**3 + a2 * lamda_box**2 + a1 * lamda_box + a0) * 640)
            points[:, 1] = ((b3 * lamda_box**3 + b2 * lamda_box**2 + b1 * lamda_box + b0) * 640)
            
            points = points[(points[:, 0] > 0) & (points[:, 0] < 640) & (points[:, 1] > 0) & (points[:, 1] < 640)]
            # for point in points:
            #     cv2.circle(orimage, (int(point[0]), int(point[1])), 4, color, 1)
            if not points.shape[0]:
                continue
            center_points = points
            boxes = np.zeros((center_points.shape[0], 5))
            r = system_configs.roi_r
            boxes[:, 1] = center_points[:, 0] - r
            boxes[:, 2] = center_points[:, 1] - r
            boxes[:, 3] = center_points[:, 0] + r
            boxes[:, 4] = center_points[:, 1] + r
            boxes = np.clip(boxes, 0, 639)
            for box in boxes:
                cv2.rectangle(orimage, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), 155, 2)
            
            Boxes.append(torch.Tensor(boxes))
            
         
        # cv2.imshow('a', orimage)
        
        # cv2.imshow('a', check_image)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(result_dir,file[:-4]+'_1stage.jpg'), orimage)


        if not Boxes:
            continue
        test_imgC = torch.cat(Boxes, 0).cuda()
        with torch.no_grad():
            point_out = nnet.model_point(encoded_feature, test_imgC)
        out_confidence, out_offset, out_scope, out_length, Mask = point_out
        out_confidence = out_confidence[:, :, 0, 0].detach().cpu()
        valid_index = torch.softmax(out_confidence, -1).numpy()[:, -1] > 0.5
        out_offset = out_offset[:, :, 0, 0].detach().cpu().numpy()
        out_scope = out_scope[:, 0, 0, 0].detach().cpu().numpy()
        valid_offset = out_offset[valid_index>0]
        valid_scope = out_scope[valid_index>0]
        valid_length = out_length[valid_index>0]
        for index, (point, theta, length) in enumerate(zip(valid_offset, valid_scope, valid_length)):
            length = system_configs.roi_r * length
            off_x, off_y = point
            theta = np.rad2deg(theta)
            box = test_imgC[valid_index>0][index][1:]
            newpoint = (box[0] + off_x *2 * r, box[1] + off_y * 2 * r)
            cv2.ellipse(orimage, (int(newpoint[0]), int(newpoint[1])), (int(length), int(length/5*2)), theta, 0, 360, 225, 2)
            cv2.circle(orimage, (int(newpoint[0]), int(newpoint[1])), 1, (255,255, 255), 1)

        
        cv2.imwrite(os.path.join(result_dir,file[:-4]+'_pred.jpg'), orimage)
        # cv2.imshow('a', orimage)
        # cv2.waitKey()
        
        
####################################################################################################################################################
    if os.path.exists('compare'):
        shutil.rmtree('compare')
    Paths_pr = []
    os.mkdir('compare')
    sums = 0
    nums = 0
    files_test = [os.path.join('test_images', p) for p in os.listdir(os.path.join(test_dir, 'test_images')) if os.path.exists(os.path.join(test_dir, 'test_labels', p[:-4]+'.txt'))]
    for file in tqdm(files_test): 
        if not os.path.exists('/data/duduguan/LSTR-Custom-Base (copy 1)/compare_gt/'+file.split('/')[1][:-4] + '_gt.json'):
            print('/data/duduguan/LSTR-Custom-Base (copy 1)/compare_gt/'+file.split('/')[1][:-4] + '_gt.json')
            continue
        dest = shutil.copyfile('/data/duduguan/LSTR-Custom-Base (copy 1)/compare_gt/'+file.split('/')[1][:-4] + '_gt.json', 'compare/'+file.split('/')[1][:-4] + '_gt.json')
        imgs_dic = {}

        imgs_dic["task_name"] = file.split('/')[-1]
        imgs_dic["lane_mark"] = []


        
        image         = cv2.imread(os.path.join(test_dir, file))
        orimage = image
        height, width = image.shape[0:2] # 800 x 800

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)

        pad_image     = image.copy() # 800 x 800
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        

        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))


        _mean = np.array([0.04129752, 0.07141293, 0.18842927], dtype=np.float32)
        _std = np.array([0.10884721, 0.09943989, 0.19230546], dtype=np.float32)
        

        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255. # 640 x 640
        normalize_(resized_image, _mean, _std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        images        = torch.from_numpy(images).cuda(non_blocking=True)
        masks         = torch.from_numpy(masks).cuda(non_blocking=True)
        
        
        time0 = time.time()
        outputs, weights, encoded_feature      = nnet.test([images, masks])
        time1 = time.time()
        
        pred_labels = outputs['pred_logits'].detach()
  
        prob = F.softmax(pred_labels, -1)
        scores, batch_labels = prob.max(-1)  # 4 10

        pred_curves = outputs['pred_curves'].detach()
        batch_curves = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)
        pred1 = batch_curves[0].cpu().numpy()  # 10 7
        labels = batch_labels[0].cpu().numpy()  # 10

        pred_valid = pred1[labels>0]

        

        # get imgC from stage 1   
        Boxes = []
        for n, lane in enumerate(pred_valid):
            lane = lane[1:]
            
            
            
            
            
            
            color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            color2 = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
           
            x1 = lane[2]
            y1 = lane[0]
            x2 = lane[3]
            y2 = lane[1]
            a3 = lane[4]
            a2 = lane[5]
            b3 = lane[6]
            b2 = lane[7]
            a1 = x2 - a3 - a2 - x1
            a0 = x1
            b1 = y2 - b3 - b2 - y1
            b0 = y1

     
            
            lamda_box = np.array([i/300 for i in range(0, 300, 10)])
            points = np.zeros((len(lamda_box), 2))
            points[:, 0] = ((a3 * lamda_box**3 + a2 * lamda_box**2 + a1 * lamda_box + a0) * 640)
            points[:, 1] = ((b3 * lamda_box**3 + b2 * lamda_box**2 + b1 * lamda_box + b0) * 640)
            
            points = points[(points[:, 0] > 0) & (points[:, 0] < 640) & (points[:, 1] > 0) & (points[:, 1] < 640)]
            # for point in points:
            #     cv2.circle(orimage, (int(point[0]), int(point[1])), 4, color, 1)
            if not points.shape[0]:
                continue
            center_points = points
            boxes = np.zeros((center_points.shape[0], 5))
            r = system_configs.roi_r
            boxes[:, 1] = center_points[:, 0] - r
            boxes[:, 2] = center_points[:, 1] - r
            boxes[:, 3] = center_points[:, 0] + r
            boxes[:, 4] = center_points[:, 1] + r
            boxes = np.clip(boxes, 0, 639)
            
            Boxes.append(torch.Tensor(boxes))
            
       
        

        if not Boxes:
            lane = []

            dic = {}
            dic["index"] = lane_index
            dic["node_list"] = lane
            dic["acce_line_info"] = "x"
            dic["lane_mark_type"] = "x"
            dic["lane_mark_color"] = "x"
            dic["index_uniq"] = 0
            imgs_dic["lane_mark"].append(dic)

            save_pr_path = 'compare/' + file.split('/')[-1][:-4] + '_pr.json'
            if os.path.exists(save_pr_path):  # 如果文件存在
                os.remove(save_pr_path)
            with open(save_pr_path, 'a') as file:
                json.dump(imgs_dic, file)
                file.write('\n')
                continue


        test_imgC = torch.cat(Boxes, 0).cuda()
        


        # get stage2 result
        img_lane_points = [Boxes[i].shape[0] for i in range(len(Boxes))]
        time2 = time.time()
        with torch.no_grad():
            point_out = nnet.model_point(encoded_feature, test_imgC)
        time3 = time.time()
        # print(time1-time0 )
        # print(time3-time2+time1-time0 )
        # print(1/(time1-time0) )
        # print(1/(time3-time2 +time1-time0) )
        out_confidences, out_offsets, out_scopes, out_lengths, _ = point_out
        out_confidences = out_confidences[:, :, 0, 0].detach().cpu()
        out_confidences = out_confidences.split(img_lane_points)
        out_offsets = out_offsets.split(img_lane_points)
        out_scopes = out_scopes.split(img_lane_points)
        out_lengths = out_lengths.split(img_lane_points)
        test_imgCs = test_imgC.split(img_lane_points)
      
        for lane_index, (out_confidence, out_offset, out_scope, out_length, test_imgC) in enumerate(zip(out_confidences, out_offsets, out_scopes, out_lengths, test_imgCs)):
            valid_index = torch.softmax(out_confidence, -1).numpy()[:, -1] > 0.5
            out_offset = out_offset[:, :, 0, 0].detach().cpu().numpy()
            out_scope = out_scope[:, 0, 0, 0].detach().cpu().numpy()
            valid_offset = out_offset[valid_index>0]
            valid_scope = out_scope[valid_index>0]
            valid_length = out_length[valid_index>0]
            lane_points = []
            for index, (point, theta, length) in enumerate(zip(valid_offset, valid_scope, valid_length)):
                length = system_configs.roi_r * length
                off_x, off_y = point
                theta = np.rad2deg(theta)
                box = test_imgC[valid_index>0][index][1:]
                newpoint = (box[0] + off_x *2 * r, box[1] + off_y * 2 * r)
                lane_points.append(newpoint)
            

        
            lane = [[x.item()/640*800, y.item()/640*800, 0] for (x, y) in lane_points]

            dic = {}
            dic["index"] = lane_index
            dic["node_list"] = lane
            dic["acce_line_info"] = "x"
            dic["lane_mark_type"] = "x"
            dic["lane_mark_color"] = "x"
            dic["index_uniq"] = 0
            imgs_dic["lane_mark"].append(dic)
        
        
        # timeend = time.time()
        # print('all', timeend - timestart)
        # print('aole:', time0 - timestart)
        # sums += (timeend - timestart)
        # nums += 1

        save_pr_path = 'compare/' + file.split('/')[-1][:-4] + '_pr.json'
        Paths_pr.append(file.split('/')[-1][:-4] + '_gt.json')
        if os.path.exists(save_pr_path):  # 如果文件存在
            os.remove(save_pr_path)
        with open(save_pr_path, 'a') as file:
            json.dump(imgs_dic, file)
            file.write('\n')
    # print(1/(sums/nums))
    
    return Paths_pr
            



if __name__ == "__main__":

    filenames = os.walk('/data/duduguan/LSTR-Custom-Base (copy 1)/compare_gt')
    filenames_gt = [n for (_,_,n) in filenames][0]

    if os.path.exists('results/test_images'):
        shutil.rmtree('results/test_images')
        shutil.rmtree('results/train_images')
    os.makedirs('results/test_images')
    os.makedirs('results/train_images')

    args = parse_args()
    if args.testiter == None:
        names = os.walk('cache/nnet/LSTR')
        names = [n for (_,_,n) in names][0]
        names = [n for n in names if n[-9:]=='model.pkl']
        names = sorted([int(n[5:-10]) for n in names])
        args.testiter = str(names[-1])
    args.cfg_file = 'LSTR'


    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))
    print('testiter:', args.testiter, time.ctime(os.path.getmtime(os.path.join('cache/nnet/LSTR', 'LSTR_'+args.testiter+'_model.pkl'))))
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    test_dir=args.image_root
    test_model=args.testiter
    result_dir='results'

    Paths_pr = inference(test_dir,test_model,result_dir)

