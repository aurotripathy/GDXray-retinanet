import pandas as pd 
import cv2
import numpy as np
from draw_poly import draw_poly
import os
import glob
from random import shuffle
from pudb import set_trace

def get_group_stats(group,
                    root_dir='/media/tempuser/RAID 5/gdxray'):
    image_files = glob.glob(os.path.join(root_dir, group) + '/*.png')
    image_count = len(image_files)
    grp_min_w = grp_min_h = 1000
    grp_max_w = grp_max_h = -1 
    for image_file in image_files:
        img = cv2.imread(image_file)
        height, width = img.shape[:2]
        if height > grp_max_h:
            grp_max_h = height
        if width > grp_max_w:
            grp_max_w = width
        if height < grp_min_h:
            grp_min_h = height
        if width < grp_min_w:
            grp_min_w = width
    
    return grp_min_w, grp_max_w, grp_min_h, grp_max_h, image_count



def get_groups_with_annotations(root_dir='/media/tempuser/RAID 5/gdxray'):

    groups = glob.glob(root_dir + '/Castings/*/ground_truth.*')
    return [group.strip(root_dir).strip('/ground_truth.txt') for group in groups]
    

def write_annotations(file_name, annotations_list):
    with open(file_name, 'w') as f:
        for annotation in annotations_list:
            if annotation['class'] == 'negative':
                f.write('{},,,,,\n'.format(annotation['img_file']))  # per the spec
            else:
                f.write('{},{},{},{},{},{}\n'.format(annotation['img_file'], 
                                                     annotation['x1'], annotation['y1'], 
                                                     annotation['x2'], annotation['y2'], 
                                                     annotation['class']))


root_dir = '/media/tempuser/RAID 5/gdxray'
ground_truth_file = 'ground_truth.txt'
annotations_dir = 'annotations-with-negatives'
annotation_file_name = 'annotations/annotations_all.csv'
annotation_file_name = os.path.join(annotations_dir, 'annotations_all.csv')
groups = get_groups_with_annotations()
# Delete any groups that don't make sense
set_trace()

groups.remove('Castings/C0055')
groups.sort()
print('Total groups', len(groups))

for group in groups:
    grp_min_w, grp_max_w, grp_min_h, grp_max_h, count = get_group_stats(group)
    print('grp={}:min W={}, max W={}, min H={}, max H={}, count={}'.format(group, grp_min_w, grp_max_w, grp_min_h, grp_max_h, count))


