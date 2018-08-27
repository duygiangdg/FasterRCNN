# calculate score for validation set

import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
import airbus

def iou(mask_true, mask_pred):
    i = np.sum((mask_true*mask_pred) >0)
    u = np.sum((mask_true + mask_pred) >0) + 0.0000000000000000001  # avoid division by zero
    return i/u

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def f2(masks_true, masks_pred):
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(masks_true) == np.sum(masks_pred) == 0:
        return 1.0
    
    f2_total = 0
    for t in thresholds:
        tp,fp,fn = 0,0,0
        ious = {}
        for i,mt in enumerate(masks_true):
            found_match = False
            for j,mp in enumerate(masks_pred):
                miou = iou(mt, mp)
                ious[100*i+j] = miou # save for later
                if miou >= t:
                    found_match = True
            if not found_match:
                fn += 1
                
        for j,mp in enumerate(masks_pred):
            found_match = False
            for i, mt in enumerate(masks_true):
                miou = ious[100*i+j]
                if miou >= t:
                    found_match = True
                    break
            if found_match:
                tp += 1
            else:
                fp += 1
        f2 = (5*tp)/(5*tp + 4*fn + fp)
        f2_total += f2
    
    return f2_total/len(thresholds)

def f2_img(img_id):
    masks_true = airbus.read_masks(img_id, 'val')
    masks_pred = airbus.read_masks(img_id, 'pred')
    return f2(masks_true, masks_pred)

def f2_all():
    score = 0
    img_ids = airbus.read_all_img_ids('pred')
    for img_id in img_ids:
        score += f2_img(img_id)
    avg_score = score/len(img_ids)
    return avg_score