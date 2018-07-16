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

def etch_image_metadata(image_id, group, img):
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(image_id) + ' ' + str(group.split('/')[1]), (10, 20), font, 0.8, green, 2, cv2.LINE_AA)
    cv2.putText(img, '{}'.format(img.shape[0:2]), (10, 50), font, 0.8, green, 2, cv2.LINE_AA)
    return img


def process_annotations(group, image_count, annotations_list, images_list,
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
        img_file = os.path.join(in_dir, in_dir.split('/')[-1] + '_{0:04d}.png'.format(image_id))
        images_list.append(img_file)
        img = cv2.imread(img_file)
        orig_img = img.copy()  # make a copy for easy visualization
        print('Shape of image', img.shape)
        
        locations = ground_truth[ground_truth['img_id'] == image_id]

        if locations.shape[0] == 0:
            annotation_dict = {'img_file' : img_file,
                               'x1' : -1, 'y1' : -1,
                               'x2' : -1, 'y2' : -1,
                               'class' : 'negative'}
            annotations_list.append(annotation_dict)
        else:
            locations = locations[['x1', 'x2', 'y1', 'y2']]
            x1_locations, x2_locations, y1_locations, y2_locations = locations['x1'].values, locations['x2'].values, locations['y1'].values, locations['y2'].values
            print(x1_locations, x2_locations, y1_locations, y2_locations)

            for x1, x2, y1, y2 in zip(x1_locations, x2_locations, y1_locations, y2_locations):
                draw_poly(img, int(x1), int(y1), int(x2), int(y2))
                annotation_dict = {'img_file' : img_file,
                                   'x1' : int(x1), 'y1' : int(y1),
                                   'x2' : int(x2), 'y2' : int(y2),
                                   'class' : 'defect'}
                annotations_list.append(annotation_dict)

        etch_image_metadata(image_id, group, img)
        cv2.imwrite(os.path.join(out_dir, out_dir.split('/')[-1] + '_{0:04d}_02.png'.format(image_id)), img)
        cv2.imwrite(os.path.join(out_dir, out_dir.split('/')[-1] + '_{0:04d}_01.png'.format(image_id)), orig_img)
        
    return annotations_list, images_list


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

def remove_groups(groups, remove_list):
    for remove_group in remove_list:
        try:
            groups.remove(remove_group)
        except ValueError:
            print('Missing group cannot be removed.', remove_group)
            exit(2)
    return(groups)


def split_train_test_by_images(images_list, annotations_list, split_fraction=0.8):
    shuffle(images_list)
    train_split_mark = int(len(images_list) * split_fraction)
    train_images_list = images_list[:train_split_mark]
    test_images_list = images_list[train_split_mark:]

    train_annotations_list = []
    for image_file in train_images_list:
        for annotation in annotations_list:
            if annotation['img_file'] == image_file:
                train_annotations_list.append(annotation)


    test_annotations_list = []
    for image_file in test_images_list:
        for annotation in annotations_list:
            if annotation['img_file'] == image_file:
                test_annotations_list.append(annotation)


    return train_annotations_list, test_annotations_list

root_dir = '/media/tempuser/RAID 5/gdxray'
ground_truth_file = 'ground_truth.txt'
annotations_dir = 'annotations-with-negatives'
annotation_file_name = 'annotations/annotations_all.csv'
annotation_file_name = os.path.join(annotations_dir, 'annotations_all.csv')
groups = get_groups_with_annotations()

# Remove any groups that are extremely low-contrast in this phase
groups = remove_groups(groups, 
                       ['Castings/C0054', 'Castings/C0055', 'Castings/C0057', 
                        'Castings/C0060', 'Castings/C0062', 'Castings/C0065'])

print('Total groups', len(groups))


if os.path.isfile(annotation_file_name):
    os.remove(annotation_file_name)

# keep appending annotations for all the groups
total_images = 0
annotations_list = []
images_list = []
for group in groups:
    grp_min_w, grp_max_w, grp_min_h, grp_max_h, image_count = get_group_stats(group)
    print('grp={}:min W={}, max W={}, min H={}, max H={}, count={}'.format(group, 
                                                                           grp_min_w, grp_max_w, 
                                                                           grp_min_h, grp_max_h, 
                                                                           image_count))
    total_images += image_count
    annotations_list, images_list = process_annotations(group, image_count, 
                                                        annotations_list, images_list)
# set_trace()

print('Total images:', total_images)
print('Total annotations', len(annotations_list))
print('Negative annotations', len([x for x in annotations_list if x['class'] == 'negative']))
print('Positive annotations', len([x for x in annotations_list if x['class'] == 'defect']))

train_annotations_list, test_annotations_list = split_train_test_by_images(images_list, annotations_list, 0.8)
write_annotations(os.path.join(annotations_dir, 'train_annotations.csv'), train_annotations_list)
write_annotations(os.path.join(annotations_dir, 'test_annotations.csv'), test_annotations_list)             

write_annotations(os.path.join(annotations_dir, 'all_annotations.csv'), annotations_list)

