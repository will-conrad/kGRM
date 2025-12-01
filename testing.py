import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import os
from threading import Thread
from collections import deque
import math

# --- 0. NEAREST NEIGHBOR IMPLEMENTATION (k=1) ---
def NN(train_features, train_labels, test_features, ord=2): 
  """
  Nearest neighbor algorithm (k=1) for gesture recognition.
  train_labels must be strings ('+', '-', etc.).
  """
  ntest = test_features.shape[0]
  ntrain = train_features.shape[0]
  # test_kNN will hold the labels assigned by the kNN algorithm
  test_kNN = np.zeros(ntest, dtype=object) # Use object dtype for strings
  min_index = np.zeros(ntest, dtype=int) - 1

  for i in range(ntest):
    distances = []
    test = test_features[i]
    for j in range(ntrain):
      train = train_features[j]

      # Calculate Euclidean distance (L2 norm since ord=2 by default)
      norm = np.linalg.norm((test - train), ord)
      distances.append(norm)
      
    minI = np.argmin(distances)
    min_index[i] = minI
    # Ensure label is treated as a string
    test_kNN[i] = str(train_labels[minI])

  return test_kNN, min_index

# --- 1. STABILIZER CLASS (The "Certainty" Logic) ---
class ValueStabilizer:
    def __init__(self, required_frames=45):
        self.required_frames = required_frames
        self.current_value = None
        self.streak = 0
    
    def update(self, new_value):
        if new_value == self.current_value and new_value is not None:
            self.streak += 1
        else:
            self.current_value = new_value
            self.streak = 0
            
        if self.streak >= self.required_frames:
            return self.current_value
        return None

    def reset(self):
        self.current_value = None
        self.streak = 0

# --- 2. FAST CAPTURE CLASS ---
class ThreadedCamera:
    def __init__(self, source=0):
        self.capture = cv.VideoCapture(source)
        self.capture.set(cv.CAP_PROP_FPS, 60)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.status, self.frame = self.capture.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.status, self.frame = self.capture.read()

    def read(self):
        return self.status, self.frame

    def stop(self):
        self.stopped = True

# --- 3. MATH & FEATURE HELPERS ---
def calc_dist(p1, p2):
    """Calculates Euclidean distance between two landmarks (x, y coords)."""
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_features(landmarks):
    """Generates the 63-dimensional feature vector for kNN."""
    points = []
    for lm in landmarks.landmark:
        # Store X, Y, and Z
        points.append([lm.x, lm.y, lm.z]) 
    points = np.array(points)
    
    # 1. Make relative to wrist (Landmark 0)
    base = points[0] 
    points = points - base
    
    # 2. Normalize by scale
    max_val = np.max(np.abs(points))
    # Avoid division by zero if hand is not detected fully
    if max_val == 0: max_val = 1 
    points = points / max_val
    
    # Reshape for NN
    return points.flatten().reshape(1, -1)

def count_fingers(landmarks):
    """Robust finger counter using distance-based logic."""
    count = 0
    wrist = landmarks.landmark[0]
    
    # Fingers (Tip vs PIP)
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_id, pip_id in finger_pairs:
        if calc_dist(landmarks.landmark[tip_id], wrist) > calc_dist(landmarks.landmark[pip_id], wrist):
            count += 1

    # Thumb (Linearity Check: Direct Path vs Segment Path)
    p1, p2, p3, p4 = landmarks.landmark[1], landmarks.landmark[2], landmarks.landmark[3], landmarks.landmark[4]
    
    segment_length = calc_dist(p1, p2) + calc_dist(p2, p3) + calc_dist(p3, p4)
    direct_length = calc_dist(p1, p4)
    
    # Avoid division by zero
    if segment_length != 0:
        linearity = direct_length / segment_length
        if linearity > 0.9:
            count += 1
            
    return count

# --- 4. MODEL LOADING ---
FILE_NAME = 'math_gestures.csv'
try:
    df = pd.read_csv(FILE_NAME)
    TRAIN_FEATURES = df.iloc[:, 1:].values
    TRAIN_LABELS = df.iloc[:, 0].values
    print("Nearest Neighbor Model Ready!")
    HAS_MODEL = True
except:
    print("⚠️ WARNING: No gesture data found. Run collector script first!")
    TRAIN_FEATURES = np.zeros((1, 63))
    TRAIN_LABELS = np.array(['+'])
    HAS_MODEL = False

# --- 5. MAIN APPLICATION SETUP ---
stream = ThreadedCamera(0).start()
time.sleep(1.0) # Allow camera to warm up

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# STATE MACHINE VARIABLES
MODE = "IDLE" # IDLE, GET_NUM_1, GET_OP, GET_NUM_2, SHOW_RESULT
num1, op, num2, result = None, None, None, None
result_timer = 0
stabilizer = ValueStabilizer(required_frames=45) 

# --- 6. MAIN LOOP ---
while True:
    if not stream.status: break
    frame = stream.frame.copy() 
    
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    cv.rectangle(frame, (0,0), (640, 80), (30, 30, 30), -1) # UI Bar
    
    if results.multi_hand_landmarks and HAS_MODEL:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- Primary Hand Analysis ---
            fingers_open_states = get_finger_state(landmarks)
            finger_count = sum(fingers_open_states)
            
            # Global Reset Gesture (Gesture A: Index + Pinky Open, others closed)
            is_gesture_a = fingers_open_states == [False, True, False, False, True] or fingers_open_states == [True, True, False, False, True]
            
            if is_gesture_a and MODE != "GET_NUM_1": # Rock gesture resets
                MODE = "GET_NUM_1"
                stabilizer.reset()
                num1, op, num2, result = None, None, None, None
                cv.putText(frame, "RESETTING...", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # --- STATE MACHINE ---
            
            if MODE == "IDLE":
                txt = "Show 'Rock' Sign to Start Math Mode"
                cv.putText(frame, txt, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif MODE == "GET_NUM_1":
                cv.putText(frame, f"1st Number: {finger_count}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                stable_val = stabilizer.update(finger_count)
                if stable_val is not None:
                    num1 = stable_val
                    MODE = "GET_OP"
                    stabilizer.reset() 

            elif MODE == "GET_OP":
                # kNN Prediction for Operator
                feats = get_features(landmarks)
                predicted_ops, _ = NN(TRAIN_FEATURES, TRAIN_LABELS, feats)
                current_op = predicted_ops[0]
                
                txt = f"{num1} [Op: {current_op or '?'}]"
                cv.putText(frame, txt, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                # Stability Check for Operator
                if current_op in ['+', '-', '*', '/']:
                    res = stabilizer.update(current_op)
                    if res is not None:
                        op = res
                        MODE = "GET_NUM_2"
                        stabilizer.reset()

            elif MODE == "GET_NUM_2":
                cv.putText(frame, f"{num1} {op} [2nd Num]: {finger_count}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                stable_val = stabilizer.update(finger_count)
                if stable_val is not None:
                    num2 = stable_val
                    MODE = "SHOW_RESULT"
                    
                    # PERFORM MATH
                    try:
                        if op == "+": result = num1 + num2
                        elif op == "-": result = num1 - num2
                        elif op == "*": result = num1 * num2
                        elif op == "/": 
                            result = round(num1 / num2, 2) if num2 != 0 else "Div By Zero"
                    except:
                        result = "Error"
                        
                    result_timer = time.time()

            elif MODE == "SHOW_RESULT":
                # Final Display
                cv.putText(frame, f"{num1} {op} {num2} = {result}", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv.putText(frame, "Result Locked. Show 'Rock' to reset.", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    # Progress Bar (Visual Feedback for Stabilizer)
    if MODE in ["GET_NUM_1", "GET_OP", "GET_NUM_2"] and stabilizer.current_value is not None:
        progress = stabilizer.streak / stabilizer.required_frames
        bar_width = int(progress * 200)
        cv.rectangle(frame, (20, 90), (20 + bar_width, 110), (0, 255, 0), -1)
        cv.rectangle(frame, (20, 90), (220, 110), (255, 255, 255), 2)

    cv.imshow("Math App", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        stream.stop()
        break

stream.stop()
cv.destroyAllWindows()