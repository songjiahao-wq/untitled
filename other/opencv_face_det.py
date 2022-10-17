import cv2
import numpy
cam = cv2.VideoCapture(0)
face_det = cv2.CascadeClassifier('./apply/haarcascade_frontalface_default.xml')
success = cam.isOpened()

while success and cv2.waitKey(1) == -1:
    ret, img = cam.read()
    face =  face_det.detectMultiScale(img,1.3,5)
    if len(face):
        for(x, y, w, h) in face:
            img = cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
            img_face = img[y :y+h , x:x+w]
            cv2.imshow('1', img_face)

    # print(img_face)
    print(img.shape)
    
    cv2.imshow('0',img)
