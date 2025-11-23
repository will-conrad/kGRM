import os
import cv2 as cv
import mediapipe as mp
import numpy as np



os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

WEBCAM_ID = 0
WIDTH = 480
HEIGHT = 640

INDEX = 8
THUMB = 4

# Show cat
# cat = cv.imread("media/cat.jpg")
# cv.imshow("CAT", cat)
# cv.waitKey(0)

# Setup Capture Device
cap = cv.VideoCapture(WEBCAM_ID)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv.CAP_PROP_FPS, 60) # Request 60 FPS

# init hand trackers
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame.flags.writeable = False
    
    # Convert to RGB for processing
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # process frames to get hands
    results = hands.process(frame)

    frame.flags.writeable = True

    distance = 0
    
    if results.multi_hand_landmarks: 
        for landmarks in results.multi_hand_landmarks: # Also changing 'hands' to 'results' here
            
            wrist_x = landmarks.landmark[THUMB].x
            wrist_y = landmarks.landmark[THUMB].y
            wrist_z = landmarks.landmark[THUMB].z
            
            index_x = landmarks.landmark[INDEX].x
            index_y = landmarks.landmark[INDEX].y
            index_z = landmarks.landmark[INDEX].z
            
            # Create NumPy arrays for easier calculation
            point_a = np.array([wrist_x, wrist_y, wrist_z])
            point_b = np.array([index_x, index_y, index_z])
            
            # Calculate Euclidean Distance
            distance = np.linalg.norm(point_a - point_b)
            
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Custom point style
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2) # Custom connection style
            )

    # Convert back to correct colorspace
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow("Webcam", frame)
    distance = (distance / 0.5) * 100
    print(distance)

    # Kill program
    if cv.waitKey(1) & 0xFF == ord('q'):
        break