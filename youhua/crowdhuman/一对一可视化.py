import os
import cv2
img_path = r'D:\songjiahao\DATA\crowhuman\images\Train/273275,22d8800006cb17cb.jpg'
label_path = r'D:\songjiahao\DATA\crowhuman\labels\Train/273275,22d8800006cb17cb.txt'
img = cv2.imread(img_path)
img_h, img_w, _ = img.shape
with open(label_path, 'r') as f:
    obj_lines = [l.strip() for l in f.readlines()]
for obj_line in obj_lines:
    cls, cx, cy, nw, nh = [float(item) for item in obj_line.split(' ')]
    color = (0, 0, 255) if cls == 0.0 else (0, 255, 0)
    x_min = int((cx - (nw / 2.0)) * img_w)
    y_min = int((cy - (nh / 2.0)) * img_h)
    x_max = int((cx + (nw / 2.0)) * img_w)
    y_max = int((cy + (nh / 2.0)) * img_h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
img = cv2.resize(img,(640,640))
cv2.imshow('Ima', img)
cv2.waitKey(0)