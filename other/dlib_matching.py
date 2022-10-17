import sys
import numpy as np
import dlib
import cv2

if len(sys.argv) < 2:
    print("Usage: %s <image file>" % sys.argv[0])
    sys.exit(1)

image_file = sys.argv[1]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
image = cv2.imread(image_file)
image = resize(image, width=1200)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)