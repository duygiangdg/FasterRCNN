# common functions for airbus ship detection challenge

import numpy as np # linear algebra
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_PATH = '../input/train'
VAL_PATH = '../input/val'
TEST_PATH = '../input/test'
TRAIN_SEG_PATH = '../input/train_ship_segmentations.csv'
VAL_SEG_PATH = '../input/val_seg.csv'
TEST_SEG_PATH = '../input/test/subsample.csv'
PRED_SEG_PATH = '../input/pred_seg.csv'

def get_path(type='train'):
    if type=='train':
        return TRAIN_PATH
    elif type=='val':
        return VAL_PATH
    else:
        return TEST_PATH

def get_seg_path(type='train'):
    if type=='train':
        return TRAIN_SEG_PATH
    elif type=='val':
        return VAL_SEG_PATH
    elif type=='test':
        return TEST_SEG_PATH
    else:
        return PRED_SEG_PATH

def read_all_img_ids(type='train'):
    path = get_seg_path(type)
    df = pd.read_csv(path)
    img_ids = df['ImageId'].values
    _, idx = np.unique(img_ids, return_index=True)
    img_ids = img_ids[np.sort(idx)]
    return img_ids

def rle_encode(mask):
    '''
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rles_to_mask(rles, all_masks=None):
    # Take the individual image rles and create a single mask array for all ships
    mask = np.zeros((768, 768), dtype = np.int16)
    for rle in rles:
        if isinstance(rle, str):
            mask += rle_decode(rle)
    return mask

def read_img(img_id, type='train'):
    if type=='train':
        path = TRAIN_PATH
    elif type=='val':
        path = VAL_PATH
    else:
        path = TEST_PATH
    img = cv.imread(path + '/' + img_id)
    return img

def read_rles(img_id, type='train'):
    path = get_seg_path(type)
    df = pd.read_csv(path)
    _rles = df.loc[df['ImageId'] == img_id, 'EncodedPixels'].tolist()
    rles = [rle for rle in _rles if isinstance(rle, str)]
    return rles

def read_masks(img_id, type='train'):
    rles = read_rles(img_id, type)
    masks = []
    for rle in rles:
        masks.append(rle_decode(rle))
    return masks

def read_mask(img_id, type='train'):
    rles = read_rles(img_id, type)
    mask = rles_to_mask(rles)
    return mask

def visualize(img_id, type='train'):
    mask = read_mask(img_id, type)
    img = read_img(img_id, type)
    _, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    axarr[2].imshow(img)
    axarr[2].imshow(mask, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()

def visualize_all(type='train'):
    img_ids = read_all_img_ids(type)
    for img_id in img_ids:
        visualize(img_id, type)