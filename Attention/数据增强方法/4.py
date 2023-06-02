import torch
import cv2
from PIL import Image
import numpy as np

# Load Yolov5 model
ptpath = r'D:\yanyi\xianyu\1\facedetector_facemask3\facedetector_facemask3\untitled\weights\yolo.pt'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define input image path
img_path = 'example.jpg'

# Load input image
im = cv2.imread(img_path)

# Convert image to RGB format
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Convert numpy array to PIL image
im = Image.fromarray(im)

# Get predictions
results = model(im)

# Extract predicted bounding boxes, confidences, classes and mask ids
bboxes = results.xyxy[0].numpy()
confidences = results.xyxy[0][:, 4].numpy()
classes = results.xyxy[0][:, 5].numpy().astype(np.int)
mask_ids = results.mask[0].numpy()

# Print the results
print("Bounding Boxes:\n", bboxes)
print("Confidences:\n", confidences)
print("Classes:\n", classes)
print("Mask IDs:\n", mask_ids)