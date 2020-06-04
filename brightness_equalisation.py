import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

root = "C:\Users\..."
images = "Snapshots of all STEREO CMEs in HI1"
new_images = "new_images"


for filename in os.listdir(images):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(root, images, filename),0)
        equ = cv2.equalizeHist(img)
        print(os.path.join(root, new_images, filename))
        cv2.imwrite(os.path.join(root, new_images, filename), equ)










