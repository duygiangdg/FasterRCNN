import numpy as np # linear algebra
from tqdm import tqdm
import json
import cv2 as cv
import pandas as pd
from airbus import read_all_img_ids, get_path, read_masks

def main():
    root = {}
    root['info'] = {
        'contributor': 'crowdAI.org',
        'about': 'Dataset for crowdAI Mapping Challenge',
        'date_created': '',
        'description': 'crowdAI mapping-challenge dataset',
        'url': 'https://www.crowdai.org/challenges/mapping-challenge',
        'version': '1.0',
        'year': 2018
    }
                        
    root['categories'] = [
        {
            'id': 101,
            'name': 'ship',
            'supercategory': 'ship'
        }
    ]

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    all_image_names = read_all_img_ids()
    np.random.shuffle(all_image_names)
    
    for image_name in tqdm(all_image_names[:1000]):
        image_id += 1
        masks = read_masks(image_name)
        image_info = {
            'id': image_id,
            'file_name': image_name,
            'width': 768,
            'height': 768,
            'date_captured': '',
            'license': 1,
            'coco_url': '',
            'flickr_url': ''
        }
        images.append(image_info)
        for mask in masks:
            annotation_id = annotation_id + 1
            _mask = mask.copy().astype(np.uint8)

            im2, contours, hierarchy = cv.findContours(_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]

            area = cv.contourArea(cnt)

            x, y, w, h = cv.boundingRect(cnt)
            bbox = [y, x, h, w]

            (x,y), (width, height), angle = cv.minAreaRect(cnt)
            rect = x, y, width, height, angle

            segmentation = [list(cnt.astype(np.float64).reshape(-1))]
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 101,
                'iscrowd': 0,
                'area': area,
                'bbox': bbox,
                'min_area_rect': rect,
                'segmentation': segmentation,
                'width': 768,
                'height': 768,
            }
            annotations.append(annotation)

    root['images'] = images
    root['annotations'] = annotations

    print('Writing JSON...')
    fp = open(get_path('test') + '/annotation.json', 'w')
    fp.write(json.dumps(root))
    print('Done')

if __name__ == '__main__':
    main()