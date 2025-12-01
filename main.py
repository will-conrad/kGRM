import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv
import mediapipe as mp
import numpy as np
import time

import csv
import sys
import pandas as pd
from threading import Thread
from sklearn.neighbors import KNeighborsClassifier


FILE_NAME = 'gestures.csv'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Colors (B, G, R)
COL_BG = (50, 50, 50)      # Dark Gray
COL_TXT = (255, 255, 255)  # White
COL_ACCENT = (255, 255, 0) # Cyan/Teal
COL_WARN = (0, 0, 255)     # Red
COL_GOOD = (0, 255, 0)     # Green


class ThreadedCamera:
    def __init__(self, source=0):
        self.capture = cv.VideoCapture(source)
        
        # set params for capture
        self.capture.set(cv.CAP_PROP_FPS, 60)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 500) 
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
        
        self.status, self.frame = self.capture.read()
        self.stopped = False
        
    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.stopped:
                self.capture.release()
                return
            
            # This read is blocking, but it's in a separate thread
            # so it doesn't slow down your main script
            status, frame = self.capture.read()
            if status:
                self.frame = frame
                self.status = status

    def read(self):
        # Return the latest frame instantly
        return self.status, self.frame

    def stop(self):
        self.stopped = True


class ValueStabilizer:
    def __init__(self, required_frames=45):
        self.required_frames = required_frames
        self.current_value = None
        self.streak = 0
    
    def update(self, new_value):
        # 1. If value matches, increment streak
        if new_value == self.current_value:
            self.streak += 1
        else:
            # 2. Value changed! Reset streak and track new value
            self.current_value = new_value
            self.streak = 0 # Start fresh
            
        # 3. Check if we reached the target
        if self.streak >= self.required_frames:
            return self.current_value
        
        return None

    def get_progress(self):
        """Returns float 0.0 to 1.0 representing stability progress."""
        return min(self.streak / self.required_frames, 1.0)

    def reset(self):
        self.current_value = None
        self.streak = 0




def get_features(landmarks):
    points = []
    for lm in landmarks.landmark:
        points.append([lm.x, lm.y]) # XY Only
    points = np.array(points)
    points = points - points[0]
    max_val = np.max(np.abs(points))
    if max_val == 0: max_val = 1
    return (points / max_val).flatten().reshape(1, -1)
def calc_dist(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def count_fingers(landmarks):
    count = 0

    # get wrist
    wrist = landmarks.landmark[0]
    
    # ========== FINGER COUNTING ========== #
    # Finger IDs: [Tip, PIP]
    # index - middle - ring - pinky
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_id, pip_id in fingers:
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[pip_id]
        
        # check orientation of finger to wrist (curledness)
        if calc_dist(tip, wrist) > calc_dist(pip, wrist):
            count += 1

    # ========== THUMB CATCHER ========== #
    ring_knuckle = landmarks.landmark[13]
    index_knuckle = landmarks.landmark[5]
    thumb_tip = landmarks.landmark[4]

    # Optimized for model uncertainty in thumb position when hand is facing away from camera
    if calc_dist(ring_knuckle, thumb_tip) > calc_dist(ring_knuckle, index_knuckle):
        count += 1

    return count

def draw_header_bg(img):
    """Draws the big gray header box."""
    cv.rectangle(img, (0, 0), (WINDOW_WIDTH, 150), COL_BG, -1)
    cv.line(img, (0, 150), (WINDOW_WIDTH, 150), (100, 100, 100), 2)

def draw_centered_text(img, text, y, font_scale=1.0, color=COL_TXT, thickness=2):
    """Centers text horizontally at height y."""
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (WINDOW_WIDTH - text_size[0]) // 2
    cv.putText(img, text, (text_x, y), font, font_scale, color, thickness, cv.LINE_AA)

def draw_progress_bar(img, progress):
    """Draws a progress bar at the bottom of the header."""
    if progress > 0:
        bar_width = int(WINDOW_WIDTH * progress)
        # Background slot
        cv.rectangle(img, (0, 140), (WINDOW_WIDTH, 150), (30, 30, 30), -1)
        # Fill
        cv.rectangle(img, (0, 140), (bar_width, 150), COL_ACCENT, -1)

FILE_NAME = 'gestures.csv'

print("--- MATH GESTURE CALCULATOR ---")

# CRITICAL CHECK: Does training data exist?
if not os.path.exists(FILE_NAME):
    print("\nError: Training data not found!")
    print(f"   Missing file: {FILE_NAME}")
    print("   Please run 'trainer.py' first to record your gestures.")
    sys.exit(1)

# Load Model
try:
    print("Loading Gesture Model...")
    df = pd.read_csv(FILE_NAME)
    if len(df) < 5: 
        raise ValueError("Data file is empty or too small.")
        
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    print("Model Loaded Successfully!")

except Exception as e:
    print(f"\nERROR LOADING MODEL: {e}")
    print("   Your training data might be corrupt. Please re-run 'trainer.py'.")
    sys.exit(1)



mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# init hand model parameters
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

stream = ThreadedCamera(0).start()
time.sleep(1.0)

MODE = "GET_NUM_1"

last_valid_finger_count = 0
num1, op, num2, result = None, None, None, None
stabilizer = ValueStabilizer(required_frames=20)


try:
    while True:
        status, frame = stream.read()
        if not status: break # ignore bad frame
        
        frame = cv.flip(frame, 1)
        display_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        
        draw_header_bg(display_frame)
        
        # --- PER-FRAME VARIABLES ---
        detected_gesture = None
        fingers = 0
        hands_present = False
        highest_confidence = 0.0

        if results.multi_hand_landmarks:
            hands_present = True
            
            temp_detected_gesture = None
            temp_highest_confidence = 0.0

            for landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # 1. Sum Fingers
                fingers += count_fingers(landmarks)
                
                # 2. Check for Operator Gestures (Take best match)
                feats = get_features(landmarks)
                pred = knn.predict(feats)[0]
                conf = np.max(knn.predict_proba(feats))
                
                if conf > 0.7 and conf > temp_highest_confidence:
                    temp_detected_gesture = pred
                    temp_highest_confidence = conf

            
            detected_gesture = temp_detected_gesture
            highest_confidence = temp_highest_confidence
        else:
            total_fingers = 0 
            stabilizer.reset() # Reset if hands leave to prevent accidental locks
            

        print(fingers)
        
        # --- STATE MACHINE (RUNS ONCE PER FRAME USING TOTALS) ---
        
        # Global Reset Check
        if detected_gesture == '_': # Reset gesture
            res = stabilizer.update('_')
            if res:
                if MODE != "GET_NUM_1":
                    print("Reset Triggered via Gesture")
                    MODE = "GET_NUM_1"
                    num1, op, num2, result = None, None, None, None

                print("stab Reset")
                stabilizer.reset()



        if MODE == "GET_NUM_1":
            draw_centered_text(display_frame, f"{fingers}", 100, 2.0, COL_ACCENT)
            draw_centered_text(display_frame, "Show First Number", 135, 0.6, COL_TXT)

            if hands_present:
                print("hand present")
                res = stabilizer.update(fingers)
                print("stab")
                print(stabilizer.current_value)


                if res is not None:
                    num1 = res
                    MODE = "GET_OP"
                    print("stab Reset")
                    stabilizer.reset()
            else:
                # If hands leave frame, break the streak
                
                print("stab Reset")
                stabilizer.reset()

        elif MODE == "GET_OP":
            valid_ops = ['+', '-', '*', '/']
            curr_op = detected_gesture if detected_gesture in valid_ops else None
            
            display_op = curr_op if curr_op else "?"
            draw_centered_text(display_frame, f"{num1}  {display_op}", 100, 2.0, COL_TXT)
            draw_centered_text(display_frame, "Show Operator (+ - * /)", 135, 0.6, COL_TXT)
            
            if curr_op:
                res = stabilizer.update(curr_op)
                if res:
                    op = res
                    MODE = "GET_NUM_2"
                    stabilizer.reset()

        elif MODE == "GET_NUM_2":
            draw_centered_text(display_frame, f"{num1} {op} {fingers}", 100, 2.0, COL_ACCENT)
            draw_centered_text(display_frame, "Show Second Number", 135, 0.6, COL_TXT)
            
            res = stabilizer.update(fingers)
            if res is not None:
                num2 = res
                MODE = "SHOW_RESULT"
                try:
                    if op == "+": result = num1 + num2
                    elif op == "-": result = num1 - num2
                    elif op == "*": result = num1 * num2
                    elif op == "/": 
                        result = round(num1 / num2, 2) if num2 != 0 else "Err"
                except: result = "Err"
                stabilizer.reset()
        
        elif MODE == "SHOW_RESULT":
            draw_centered_text(display_frame, f"{num1} {op} {num2} = {result}", 100, 2.0, COL_GOOD)
            draw_centered_text(display_frame, "Use Start/Reset gesture to Reset", 135, 0.6, COL_TXT)

        # Progress Bar Logic
        # Draw Progress Bar in Header
        if MODE in ["GET_NUM_1", "GET_OP", "GET_NUM_2"]:
            prog = stabilizer.get_progress()
            draw_progress_bar(display_frame, prog)

        # Show Window
        cv.imshow("Math Calculator", display_frame)
        if cv.waitKey(1) & 0xFF == 27: break
finally:
    stream.stop()
    cv.destroyAllWindows()
    print("Program exited cleanly.")