import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import os
import csv
import sys
import pandas as pd
from threading import Thread
from sklearn.neighbors import KNeighborsClassifier

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


    print("--- GESTURE TRAINER STARTED ---")
    
    # Initialize file with header
    with open(file_name, 'w', newline='') as f:
        csv.writer(f).writerow(['label'] + [f'v{i}' for i in range(42)])

    samples = {'+': 0, '-': 0, '*': 0, '/': 0, '_': 0, '=': 0}
    
    recording_active = False
    target_char = None
    frames_to_record = 0
    countdown_start_time = 0
    countdown_duration = 3.0
    
    # Buffer to hold data in RAM before writing (Prevents I/O lag/hangs)
    data_buffer = []

    while True:
        status, frame = stream.read()
        if not status: break
        
        display_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # --- UI DRAWING ---
        cv.rectangle(display_frame, (0,0), (640, 80), (30, 30, 30), -1)
        cv.putText(display_frame, "TRAINING MODE", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        y = 100
        for op, count in samples.items():
            col = (0, 255, 0) if count >= 40 else (0, 0, 255)
            # Friendly names
            name = "SPACE (Start)" if op == '_' else op
            name = "EQUALS (End)" if op == '=' else name
            
            cv.putText(display_frame, f"'{name}': {count}", (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            y += 35

        current_features = None
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            current_features = get_features(landmarks).flatten()

        # --- BURST LOGIC ---
        if recording_active:
            elapsed = time.time() - countdown_start_time
            
            # Phase 1: Countdown
            if elapsed < countdown_duration:
                sec = int(countdown_duration - elapsed) + 1
                cv.putText(display_frame, f"READY: {sec}", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            
            # Phase 2: Capture to RAM Buffer
            elif frames_to_record > 0:
                cv.putText(display_frame, "CAPTURING!", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                if current_features is not None:
                    # Save to RAM buffer instead of Disk
                    data_buffer.append([target_char] + list(current_features))
                    
                    # Mirror (Data Augmentation for opposite hand)
                    feats_mirror = current_features.copy()
                    feats_mirror[0::2] = feats_mirror[0::2] * -1
                    data_buffer.append([target_char] + list(feats_mirror))
                    
                    samples[target_char] += 2
                    frames_to_record -= 1
                    cv.rectangle(display_frame, (0,0), (640, 480), (255, 255, 255), 10)
            
            # Phase 3: Write to Disk (Once)
            else:
                print(f"Writing {len(data_buffer)} samples to disk...")
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_buffer)
                
                # Cleanup
                data_buffer = [] 
                recording_active = False
                target_char = None
        
        else:
            cv.putText(display_frame, "Keys: +, -, *, /, SPACE, =", (20, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            # Only show "Q to Finish" if we have enough data
            if sum(samples.values()) >= 20:
                 cv.putText(display_frame, "Press 'Q' to Finish", (20, y+50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow("Math App", display_frame)
        
        # --- CRITICAL FIX: Update Window ---
        # Moving waitKey outside ensures the window updates every frame
        key = cv.waitKey(1) & 0xFF
        
        if key == 27: # ESC force quit
            return False

        if not recording_active:
            # Map Spacebar(32) to '_'
            char = '_' if key == 32 else chr(key)
            
            if char in samples:
                recording_active = True
                target_char = char
                frames_to_record = 20
                countdown_start_time = time.time()
                data_buffer = [] # Reset buffer
            elif key == ord('q'):
                # Simple check: do we have at least SOME data?
                if sum(samples.values()) < 20:
                    print("⚠️ Record more samples!")
                else:
                    return True # Signal Success
    print("--- GESTURE TRAINER STARTED ---")
    
    with open(file_name, 'w', newline='') as f:
        csv.writer(f).writerow(['label'] + [f'v{i}' for i in range(42)])

    samples = {'+': 0, '-': 0, '*': 0, '/': 0, '_': 0, '=': 0}
    
    recording_active = False
    target_char = None
    frames_to_record = 0
    countdown_start_time = 0
    countdown_duration = 3.0
    
    # Buffer to hold data before writing (Prevents I/O lag)
    data_buffer = []

    while True:
        status, frame = stream.read()
        if not status: break
        display_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        cv.rectangle(display_frame, (0,0), (640, 80), (30, 30, 30), -1)
        cv.putText(display_frame, "TRAINING MODE", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        y = 100
        total_samples = sum(samples.values())
        for op, count in samples.items():
            col = (0, 255, 0) if count >= 40 else (0, 0, 255)
            name = "SPACE" if op == '_' else ("EQUALS" if op == '=' else op)
            cv.putText(display_frame, f"'{name}': {count}", (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            y += 35

        current_features = None
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            current_features = get_features(landmarks).flatten()

        # --- LOGIC ---
        if recording_active:
            elapsed = time.time() - countdown_start_time
            if elapsed < countdown_duration:
                sec = int(countdown_duration - elapsed) + 1
                cv.putText(display_frame, f"READY: {sec}", (220, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            elif frames_to_record > 0:
                cv.putText(display_frame, "CAPTURING!", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                if current_features is not None:
                    # Save to RAM buffer instead of Disk
                    data_buffer.append([target_char] + list(current_features))
                    
                    # Mirror
                    feats_mirror = current_features.copy()
                    feats_mirror[0::2] = feats_mirror[0::2] * -1
                    data_buffer.append([target_char] + list(feats_mirror))
                    
                    samples[target_char] += 2
                    frames_to_record -= 1
                    cv.rectangle(display_frame, (0,0), (640, 480), (255, 255, 255), 10)
            else:
                # Burst Finished: Write buffer to disk now
                print(f"Writing {len(data_buffer)} samples to disk...")
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_buffer)
                data_buffer = [] # Clear buffer
                
                recording_active = False
                target_char = None
        else:
            cv.putText(display_frame, "Keys: +, -, *, /, SPACE, =", (20, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv.putText(display_frame, "ESC to Quit App", (20, y+50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            if total_samples >= 20:
                cv.putText(display_frame, "Press 'Q' to Finish Training", (20, y+80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow("Math App", display_frame)
        
        # --- KEY HANDLING ---
        key = cv.waitKey(1) & 0xFF
        
        if key == 27: # ESC key
            print("Force Quit requested.")
            return False # Signal to Exit App

        if not recording_active:
            char = '_' if key == 32 else chr(key)
            if char in samples:
                recording_active = True
                target_char = char
                frames_to_record = 20
                countdown_start_time = time.time()
                data_buffer = [] # Reset buffer
            elif key == ord('q'):
                if sum(samples.values()) < 20:
                    print("⚠️ Record more samples!")
                else:
                    return True # Signal Success
    print("--- GESTURE TRAINER STARTED ---")
    
    # Overwrite file with new header
    with open(file_name, 'w', newline='') as f:
        csv.writer(f).writerow(['label'] + [f'v{i}' for i in range(42)])

    # Added '_' for Space (Start) and '=' for Equals (End)
    samples = {'+': 0, '-': 0, '*': 0, '/': 0, '_': 0, '=': 0}
    
    recording_active = False
    target_char = None
    frames_to_record = 0
    countdown_start_time = 0
    countdown_duration = 3.0

    while True:
        status, frame = stream.read()
        if not status: break
        
        display_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # --- UI DRAWING ---
        cv.rectangle(display_frame, (0,0), (640, 80), (30, 30, 30), -1)
        cv.putText(display_frame, "TRAINING MODE", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        y = 100
        for op, count in samples.items():
            col = (0, 255, 0) if count >= 40 else (0, 0, 255)
            # Friendly names
            name = "SPACE (Start)" if op == '_' else op
            name = "EQUALS (End)" if op == '=' else name
            
            cv.putText(display_frame, f"'{name}': {count}", (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            y += 35

        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            current_features = get_features(landmarks).flatten()

        # --- BURST LOGIC ---
        if recording_active:
            elapsed = time.time() - countdown_start_time
            
            if elapsed < countdown_duration:
                seconds_left = int(countdown_duration - elapsed) + 1
                cv.putText(display_frame, f"READY: {seconds_left}", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            
            elif frames_to_record > 0:
                cv.putText(display_frame, "CAPTURING!", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                if current_features is not None:
                    with open(file_name, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # Original
                        writer.writerow([target_char] + list(current_features))
                        # Mirror
                        feats_mirror = current_features.copy()
                        feats_mirror[0::2] = feats_mirror[0::2] * -1
                        writer.writerow([target_char] + list(feats_mirror))
                    
                    samples[target_char] += 2
                    frames_to_record -= 1
                    cv.rectangle(display_frame, (0,0), (640, 480), (255, 255, 255), 10)
            
            else:
                recording_active = False
                target_char = None
        
        else:
            cv.putText(display_frame, "Keys: +, -, *, /, SPACE, =", (20, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv.imshow("Math App", display_frame)
        
        # --- CRITICAL FIX: Update Window ---
        key = cv.waitKey(1) & 0xFF
        
        # --- KEY HANDLING ---
        if not recording_active:
            # Map Spacebar(32) to '_'
            char = '_' if key == 32 else chr(key)
            
            if char in samples:
                recording_active = True
                target_char = char
                frames_to_record = 20
                countdown_start_time = time.time()
            elif key == ord('q'):
                if sum(samples.values()) < 20:
                    print("⚠️ Record more samples!")
                else:
                    break


FILE_NAME = 'math_gestures.csv'

print("--- MATH GESTURE CALCULATOR ---")

# CRITICAL CHECK: Does training data exist?
if not os.path.exists(FILE_NAME):
    print("\n❌ CRITICAL ERROR: Training data not found!")
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
    print("✅ Model Loaded Successfully!")

except Exception as e:
    print(f"\n❌ ERROR LOADING MODEL: {e}")
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

MODE = "IDLE"
num1, op, num2, result = None, None, None, None
stabilizer = ValueStabilizer(required_frames=45)


try:
    while True:
        status, frame = stream.read()
        if not status: break
        
        frame = cv.flip(frame, 1)
        display_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        cv.rectangle(display_frame, (0,0), (640, 80), (30, 30, 30), -1)
        
        # --- PER-FRAME VARIABLES ---
        total_fingers = 0
        detected_gesture = None
        highest_confidence = 0.0

        # --- PROCESS ALL HANDS FIRST ---
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # 1. Sum Fingers
                total_fingers += count_fingers(landmarks)
                
                # 2. Check for Operator Gestures (Take best match)
                feats = get_features(landmarks)
                pred = knn.predict(feats)[0]
                conf = np.max(knn.predict_proba(feats))
                
                # Keep the gesture with highest confidence in this frame
                if conf > 0.7 and conf > highest_confidence:
                    detected_gesture = pred
                    highest_confidence = conf

        # --- STATE MACHINE (RUNS ONCE PER FRAME USING TOTALS) ---
        
        # Global Reset Check
        if detected_gesture == '_': # SPACE gesture
            res = stabilizer.update('_')
            if res:
                if MODE != "GET_NUM_1":
                    print("Reset Triggered via Gesture")
                    MODE = "GET_NUM_1"
                    num1, op, num2, result = None, None, None, None
                stabilizer.reset()

        # Logic Flow
        if MODE == "IDLE":
            cv.putText(display_frame, "Do 'SPACE' Gesture to Start", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif MODE == "GET_NUM_1":
            cv.putText(display_frame, f"1st Number: {total_fingers}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Stabilize on the TOTAL finger count
            res = stabilizer.update(total_fingers)
            if res is not None:
                num1 = res
                MODE = "GET_OP"
                stabilizer.reset() 

        elif MODE == "GET_OP":
            valid_ops = ['+', '-', '*', '/']
            curr_op = detected_gesture if detected_gesture in valid_ops else None
            
            txt = f"{num1} [Op: {curr_op or '?'}] {int(highest_confidence*100)}%"
            cv.putText(display_frame, txt, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            if curr_op:
                res = stabilizer.update(curr_op)
                if res:
                    op = res
                    MODE = "GET_NUM_2"
                    stabilizer.reset()

        elif MODE == "GET_NUM_2":
            cv.putText(display_frame, f"{num1} {op} [Num2: {total_fingers}]", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            res = stabilizer.update(total_fingers)
            if res is not None:
                num2 = res
                MODE = "WAIT_EQUALS" 
                stabilizer.reset()

        elif MODE == "WAIT_EQUALS":
            cv.putText(display_frame, f"{num1} {op} {num2} ... [Show =]", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if detected_gesture == '=':
                res = stabilizer.update('=')
                if res:
                    MODE = "SHOW_RESULT"
                    try:
                        if op == "+": result = num1 + num2
                        elif op == "-": result = num1 - num2
                        elif op == "*": result = num1 * num2
                        elif op == "/": 
                            result = round(num1 / num2, 2) if num2 != 0 else "Err"
                    except: result = "Err"
        
        elif MODE == "SHOW_RESULT":
            cv.putText(display_frame, f"{num1} {op} {num2} = {result}", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv.putText(display_frame, "Do 'SPACE' to Reset", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Progress Bar Logic
        if MODE in ["GET_NUM_1", "GET_OP", "GET_NUM_2", "WAIT_EQUALS"] and stabilizer.current_value is not None:
            prog = int((stabilizer.streak / stabilizer.required_frames) * 200)
            cv.rectangle(display_frame, (20, 90), (20 + prog, 110), (0, 255, 0), -1)

        cv.imshow("Math Calculator", display_frame)
        if cv.waitKey(1) & 0xFF == 27: break

finally:
    stream.stop()
    cv.destroyAllWindows()
    print("Program exited cleanly.")

# frame processing loop
# while False:
#     # get frame from camera
#     success, frame = stream.read()
#     if not success:
#         break
    
#     # performance fix, convert color for processing to RGB
#     frame.flags.writeable = False
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

#     # process frame through hands model
#     results = hands.process(frame)

#     # performance fix, convert color for processing to BGR for viewing
#     frame.flags.writeable = True
#     frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
#     finger_count = 0
    
#     if results.multi_hand_landmarks: 
#         for landmarks in results.multi_hand_landmarks:
            
            
            
#             finger_count += count_fingers(landmarks)




#             mp_drawing.draw_landmarks(
#                 frame, 
#                 landmarks, 
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
#             )

#     cv.putText(frame, f'Count: {finger_count}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#     cv.imshow("Webcam", frame)

#     # break on Q
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         stream.stop()
#         break

# # clean up
# cv.destroyAllWindows()