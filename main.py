import cv2 as cv

# cat = cv.imread("media/cat.jpg")
# cv.imshow("CAT", cat)
# cv.waitKey(0)

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

success, frame = cap.read()
if success:
    cv.imshow("Webcam", frame)
    cv.waitKey(0)