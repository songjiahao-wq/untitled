from cvzone.FaceDetectionModule import FaceDetector
import cv2
import imutils
cap = cv2.VideoCapture(r'data/song.jpg')
detector = FaceDetector()

while True:
    success, img = cap.read()
    img = imutils.resize(img, width=1024)
    img, bboxs = detector.findFaces(img)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
    # img = cv2.putText(img, "中国", (50,50), cv2.FONT_HERSHEY_PLAIN,2, (0,255,255))
    cv2.imshow("Image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
