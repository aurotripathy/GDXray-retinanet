import pandas as pd 
import cv2
import numpy as np
from draw_poly import draw_poly
import os
import glob
from random import shuffle
from pudb import set_trace
"""
Input:
Directory of the images and the ground-truth
Number of images starting with index 1
Output:
Directory is prepended with 'superimposed'
Also generates the annotation file in the local folder
"""

def display_superimposed_boxes(group, image_count, annotations_file, annotations_list,
                               root_dir='/media/tempuser/RAID 5/gdxray',
                               ground_truth_file='ground_truth.txt'): 
    in_dir = os.path.join(root_dir, group)
    gt_file = os.path.join(in_dir, ground_truth_file)
    out_dir = os.path.join(root_dir, 'superimposed', group)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ground_truth = pd.read_table(gt_file, 
                                 delim_whitespace=True, 
                                 names=('img_id', 'x1', 'x2', 'y1', 'y2'))
    if ground_truth is not None:
        print(ground_truth.head())
    else:
        print('Cannot read file', gt_file)

    for image_id in range(1, image_count+1):
        print('image id', image_id)
        locations = ground_truth[ground_truth['img_id'] == image_id]
        locations = locations[['x1', 'x2', 'y1', 'y2']]
        x1_locations, x2_locations, y1_locations, y2_locations = locations['x1'].values, locations['x2'].values, locations['y1'].values, locations['y2'].values
        print(x1_locations, x2_locations, y1_locations, y2_locations)

        img_file = os.path.join(in_dir, in_dir.split('/')[-1] + '_{0:04d}.png'.format(image_id))
        img = cv2.imread(img_file)

        print('Shape of image', img.shape)

        for x1, x2, y1, y2 in zip(x1_locations, x2_locations, y1_locations, y2_locations):
            draw_poly(img, int(x1), int(y1), int(x2), int(y2))
            annotations_file.write('{},{},{},{},{},defect\n'.format(img_file, int(x1), int(y1), int(x2), int(y2)))
            annotation_dict = {'img_file' : img_file,
                               'x1' : int(x1), 'y1' : int(y1),
                               'x2' : int(x2), 'y2' : int(y2),
                               'class' : 'defect'}
            annotations_list.append(annotation_dict)

        cv2.imwrite(os.path.join(out_dir, 'super' + out_dir.split('/')[-1] + '_{0:04d}.png'.format(image_id)), img)
        
    return annotations_file, annotations_list


def get_image_count(group,
                    root_dir='/media/tempuser/RAID 5/gdxray'):
    return(len(glob.glob(os.path.join(root_dir, group) + '/*.png')))


def get_groups_with_annotations(root_dir='/media/tempuser/RAID 5/gdxray'):

    groups = glob.glob(root_dir + '/Castings/*/ground_truth.*')
    return [group.strip(root_dir).strip('/ground_truth.txt') for group in groups]
    

def write_annotations(file_name, annotations_list):
    with open(file_name, 'w') as f:
        for annotation in annotations_list:
            f.write('{},{},{},{},{},{}\n'.format(annotation['img_file'], 
                                                 annotation['x1'], annotation['y1'], 
                                                 annotation['x2'], annotation['y2'], 
                                                 annotation['class']))


root_dir = '/media/tempuser/RAID 5/gdxray'
ground_truth_file = 'ground_truth.txt'
annotation_file_name = 'annotations/annotations_all.csv'

groups = get_groups_with_annotations()
print('Total groups', len(groups))
set_trace()

if os.path.isfile(annotation_file_name):
    os.remove(annotation_file_name)

# create an empty annotation file
annotations_file = open(annotation_file_name, 'w')
# keep appending annotations for all the groups
total_images = 0
annotations_list = []
for group in groups:
    image_count = get_image_count(group)
    total_images += image_count
    annotations_file, annotations_list = display_superimposed_boxes(group, image_count, annotations_file, annotations_list)
annotations_file.close()
print('Total images:', total_images)
print('Total annotations', len(annotations_list))

set_trace()
shuffle(annotations_list)  # in place
train_split_mark = int(len(annotations_list) * 0.8)
train_annotations_list = annotations_list[:train_split_mark]
test_annotations_list = annotations_list[train_split_mark:]
write_annotations('annotations/train_annotations.csv', train_annotations_list)
write_annotations('annotations/test_annotations.csv', test_annotations_list)             

