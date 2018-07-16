# cerate 10 more via cropping
from pudb import set_trace
import numpy as np
import cv2
from random import randint

image_file = "original.png"
reduction_fraction = 0.1
image = cv2.imread(image_file)
height = image.shape[0]
width = image.shape[1]
print(height, width)

max_delta_x = int(width * reduction_fraction/2)
max_delta_y = int(height * reduction_fraction/2)

delta_x = randint(0, max_delta_x)
delta_y = randint(0, max_delta_y)
print(delta_y, delta_x)

new_width = int(width * (1 - reduction_fraction))
new_height = int(height * (1 - reduction_fraction))
print(new_height, new_width)

set_trace()
new_image = image[delta_y : delta_y + new_height, delta_x : delta_x + new_width, :]

cv2.imwrite('cropped.png', new_image)



