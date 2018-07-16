import pandas as pd 
import cv2
import numpy as np
from draw_poly import draw_poly

"""
Mimics the matlab code shipped with the data to display superimposed boxes
uses image id, 44
"""

gt_file = '/media/tempuser/RAID 5/gdxray/Castings/C0002/ground_truth.txt'
ground_truth = pd.read_table(gt_file, 
                             delim_whitespace=True, 
                             names=('img_id', 'x1', 'x2', 'y1', 'y2'))
if ground_truth is not None:
    print(ground_truth.head())
else:
    print('Cannot read file', gt_file)

locations = ground_truth[ground_truth['img_id'] == 44]
locations = locations[['x1', 'x2', 'y1', 'y2']]
x1_locations, x2_locations, y1_locations, y2_locations = locations['x1'].values, locations['x2'].values, locations['y1'].values, locations['y2'].values
print(x1_locations, x2_locations, y1_locations, y2_locations)

img_file = '/media/tempuser/RAID 5/gdxray/Castings/C0002/C0002_0044.png'
img = cv2.imread(img_file)
cv2.imwrite('original.png', img)
print('Shape of image', img.shape)

for x1, x2, y1, y2 in zip(x1_locations, x2_locations, y1_locations, y2_locations):
    draw_poly(img, int(x1), int(y1), int(x2), int(y2))

cv2.imwrite('box-superimposed.png', img)



