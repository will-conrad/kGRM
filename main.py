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

success, frame = cap.read()
if success:
    cv.imshow("Webcam", frame)
    cv.waitKey(0)
    cap.release()

# Save frame
def take_photo():
    print("Taking Photo")
    cap = cv.VideoCapture(webcam_ID)
    success, frame = cap.read()
    cv.imwrite('photos/test.jpg', frame)
    cap.release()

take_photo()