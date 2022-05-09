# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:25:19 2021

@author: W10
"""
from PIL import Image
import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import Output






#%%


from boxdetect import config
config = config.PipelinesConfig()

#to determine configuration settings
config.width_range = (40,175)
config.height_range = (20,90)
config.scaling_factors = [0.9,0.85,0.99,0.99,0.98,0.97,0.89,0.6,0.5,0.4,0.78,0.80,0.3,0.2,0.1,0.33,0.44,0.22,0.11]
#config.dilation_iterations = 0
config.wh_ratio_range = (0.6, 5) 
config.group_size_range = (2, 100)
#config.horizontal_max_distance_multiplier = 2

#to import libraries
import numpy
from boxdetect.pipelines import get_boxes

#to get rectangular bounding boxes of the paper
image_path = cv2.imread("apo.jpeg")
rects, grouped_rects, org_image, output_image = get_boxes(image_path, cfg=config, plot=False)

plt.figure(figsize=(10,10))
plt.imshow(output_image)
plt.show()

from boxdetect.pipelines import get_checkboxes

checkboxes = get_checkboxes(
    image_path, cfg=config, px_threshold=0.1, plot=False, verbose=True)

coord_list=[]

for checkbox in checkboxes:
#    print("(x,y,width,height): ", checkbox[0])
    coord_list.append(checkbox[0])

coord_list = numpy.array(coord_list)
x1 = coord_list[:,0]
y1= coord_list[:,1]
x2 = x1 + coord_list[:,2]
y2 = y1 + coord_list[:,3]

ocr_result_list=[]

for i in range(len(coord_list)):
    image_roi = image_path[y1[i]:y2[i],x1[i]:x2[i]]
    ocr_result=pytesseract.image_to_string(image_roi, lang='tur+eng',)
    ocr_result_list.append(ocr_result.strip())


print(f"{ocr_result_list[15]}  {ocr_result_list[16]}")

