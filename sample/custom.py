import cv2
import numpy as np
import torch
from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from imgaug.augmentables.lines import LineString, LineStringsOnImage
import random

def myclip_out_of_image(db, LS):

    flag = False
    shape = LS.shape
    lanes = db.linestrings_to_lanes(LS)

    out_lanes = []
    


    for lane in lanes:
        deletes = []

        for index, point in enumerate(lane):
            # cv2.circle(check1, (int(point[0]), int(point[1])), 1, (255, 255, 255), -1)
            if point[0] > shape[1] or point[0] < 0 or point[1] > shape[0] or point[1] < 0:
                deletes.append(index)

        lane = np.delete(lane, deletes, axis=0)

        if lane.shape[0] > 3:
            out_lanes.append(lane)


    out_strings = db.lane_to_linestrings(out_lanes)
    if not out_lanes:
        flag = True
    return LineStringsOnImage(out_strings, shape=shape), flag



def kp_detection(db, k_ind):
    data_rng     = system_configs.data_rng
    batch_size   = system_configs.batch_size
    input_size   = db.configs["input_size"]
    lighting     = db.configs["lighting"]
    rand_color   = db.configs["rand_color"]
    images   = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32) # b, 3, H, W
    masks    = np.zeros((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)  # b, 1, H, W
    gt_lanes = []

    db_size = db.db_inds.size # 3268 | 2782

    discard = []
    firstcheck = True
    for b_ind in range(batch_size):
        
        if k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading ground truth
        item  = db.detections(db_ind) # all in the raw coordinate
        img   = cv2.imread(item['old_anno']['path'])
        mask  = np.ones((1, img.shape[0], img.shape[1], 1), dtype=np.bool)
        line_strings = db.lane_to_linestrings(item['old_anno']['raw_lanes'])
        line_strings = LineStringsOnImage(line_strings, shape=img.shape)

  

        img, line_strings, mask = db.transform(image=img, line_strings=line_strings, segmentation_maps=mask)


        
        line_strings, isnan = myclip_out_of_image(db, line_strings)
        if isnan:
            discard.append(b_ind)
             
        old_anno = {'path': item['old_anno']['path'],
                    'raw_lanes': db.linestrings_to_lanes(line_strings),
                    'categories': [1] * len(line_strings)}
        label = db._transform_annotation(old_anno, img_wh=(input_size[1], input_size[0]))['label']


        tgt_ids   = label[:, 0]
        label     = label[tgt_ids > 0]

        label = np.stack([label] * batch_size, axis=0)
        if not isnan:
            gt_lanes.append(torch.from_numpy(label.astype(np.float32)))

        img = (img / 255.).astype(np.float32)

        normalize_(img, db.mean, db.std)
        
        images[b_ind]   = img.transpose((2, 0, 1))
        masks[b_ind]    = np.logical_not(mask[:, :, :, 0])

 
    images = np.delete(images, discard, axis=0)
    masks = np.delete(masks, discard, axis=0)
    for i in range(len(discard)):
        gt_lanes.append(gt_lanes[i])
        images = np.concatenate((images, np.expand_dims(images[i], 0)), axis=0)
        masks = np.concatenate((masks, np.expand_dims(masks[i], 0)), axis=0)
        
    images = torch.from_numpy(images)
    masks = torch.from_numpy(masks)

    return {
               "xs": [images, masks],
               "ys": [images, *gt_lanes]
           }, k_ind


def sample_data(db, k_ind):
    return globals()[system_configs.sampling_function](db, k_ind)


