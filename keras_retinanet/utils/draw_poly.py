# needed format 
# path/to/image.jpg,x1,y1,x2,y2,class_name

import numpy as np
import cv2

def draw_poly(img, x1, y1, x2, y2):
    pts = np.array([ [x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1] ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img,[pts], True, (0, 255, 255))


if __name__ == "__main__":
    # Create a black image
    img = np.zeros((512,512,3), np.uint8)
    x1, y1 = 0, 0
    x2, y2 = 100, 100
    # Draw a polygon
    draw_poly(img, x1, y1, x2, y2)
    #write the image
    cv2.imwrite('poly.png', img) 
