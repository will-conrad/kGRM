import cv2 as cv


webcam_ID = 0

# Show cat
# cat = cv.imread("media/cat.jpg")
# cv.imshow("CAT", cat)
# cv.waitKey(0)

# Setup Capture Device
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    success, frame = cap.read()
    if not success:
        break
    cv.imshow("Webcam", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break