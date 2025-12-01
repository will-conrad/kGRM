import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import csv
import os
import pandas as pd
from threading import Thread
from queue import Queue

# --- CONFIGURATION ---
FILE_NAME = 'gestures.csv'
TARGET_SAMPLES = 25 # Real captures (results in 40 rows due to mirroring)

# ==========================================
# 1. BACKGROUND WRITER THREAD
# ==========================================
class DataWriter(Thread):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.queue = Queue()
        self.daemon = True 
        self.running = True
        self.start()
    
    def run(self):
        while self.running:
            data_batch = self.queue.get()
            if data_batch is None: break 
            
            try:
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_batch)
            except Exception as e:
                print(f"Write error: {e}")
                
            self.queue.task_done()
    
    def save(self, batch):
        self.queue.put(batch)
        
    def stop(self):
        self.running = False
        self.queue.put(None)
        self.join()

# ==========================================
# 2. FAST CAMERA THREAD
# ==========================================
class ThreadedCamera:
    def __init__(self, source=0):
        self.capture = cv.VideoCapture(source)
        self.capture.set(cv.CAP_PROP_FPS, 60)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.status, self.frame = self.capture.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            if not self.capture.isOpened(): break
            self.status, self.frame = self.capture.read()
        self.capture.release()
    
    def stop(self):
        self.stopped = True
    
    def read(self):
        return self.status, self.frame

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_features(landmarks):
    points = []
    for lm in landmarks.landmark:
        points.append([lm.x, lm.y])
    points = np.array(points)
    points = points - points[0] 
    max_val = np.max(np.abs(points))
    if max_val == 0: max_val = 1
    return (points / max_val).flatten()

def load_existing_counts(filename):
    """Reads the CSV and returns a dictionary of counts per label."""
    counts = {'+': 0, '-': 0, '*': 0, '/': 0, '_': 0, '=': 0}
    if os.path.exists(filename):
        try:
            # Read CSV to count samples
            df = pd.read_csv(filename)
            if not df.empty:
                val_counts = df['label'].value_counts()
                for label, count in val_counts.items():
                    if label in counts:
                        counts[label] = count
                print(f"Loaded existing data: {counts}")
        except Exception as e:
            print(f"Error reading file, starting fresh: {e}")
    return counts

def reset_file(filename):
    """Wipes the file and writes the header."""
    with open(filename, 'w', newline='') as f:
        csv.writer(f).writerow(['label'] + [f'v{i}' for i in range(42)])

# ==========================================
# 4. MAIN TRAINER LOOP
# ==========================================
def main():
    print("--- LAUNCHING GESTURE TRAINER ---")
    
    # 1. Setup File & Counts
    if not os.path.exists(FILE_NAME):
        reset_file(FILE_NAME)
    
    samples = load_existing_counts(FILE_NAME)

    # 2. Start Threads
    writer_thread = DataWriter(FILE_NAME)
    stream = ThreadedCamera(0).start()
    time.sleep(1.0) 

    # 3. MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # 4. State Variables
    recording_active = False
    target_char = None
    frames_to_record = 0
    countdown_start = 0
    countdown_duration = 3.0
    
    # Feedback message (e.g., "DATA CLEARED")
    feedback_msg = ""
    feedback_timer = 0

    print("Ready. Switch to window.")

    try:
        while True:
            status, frame = stream.read()
            if not status: break
            
            # Flip for mirror effect
            display_frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # --- UI LAYOUT ---
            cv.rectangle(display_frame, (0,0), (640, 80), (30, 30, 30), -1)
            cv.putText(display_frame, "GESTURE TRAINER", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw Sample Counts
            y = 120
            all_green = True
            for op, count in samples.items():
                # We want 40 total samples (20 real + 20 mirror)
                is_ready = count >= (TARGET_SAMPLES * 2)
                if not is_ready: all_green = False
                
                col = (0, 255, 0) if is_ready else (0, 0, 255)
                name = "SPACE" if op == '_' else ("EQUALS" if op == '=' else op)
                cv.putText(display_frame, f"'{name}': {count}", (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                y += 35

            # Process Hand
            current_features = None
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display_frame, landmarks, mp_hands.HAND_CONNECTIONS)
                current_features = get_features(landmarks)

            # --- RECORDING LOGIC ---
            if recording_active:
                elapsed = time.time() - countdown_start
                
                if elapsed < countdown_duration:
                    sec = int(countdown_duration - elapsed) + 1
                    cv.putText(display_frame, f"HOLD POSE: {sec}", (220, 240), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                elif frames_to_record > 0:
                    cv.putText(display_frame, "CAPTURING...", (200, 240), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    if current_features is not None:
                        batch = []
                        # 1. Real Hand
                        batch.append([target_char] + list(current_features))
                        # 2. Mirror Hand (Data Augmentation)
                        feats_mirror = current_features.copy()
                        feats_mirror[0::2] = feats_mirror[0::2] * -1
                        batch.append([target_char] + list(feats_mirror))
                        
                        writer_thread.save(batch)
                        
                        samples[target_char] += 2
                        frames_to_record -= 1
                        
                        cv.rectangle(display_frame, (0,0), (640, 480), (255, 255, 255), 10)
                
                else:
                    recording_active = False
                    target_char = None
            
            else:
                # IDLE INSTRUCTIONS
                cv.putText(display_frame, "Keys: +, -, *, /, SPACE, =", (20, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv.putText(display_frame, "Press 'C' to Clear All Data", (20, y+50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                if all_green:
                    cv.putText(display_frame, "DONE! PRESS 'Q' TO EXIT", (20, y+85), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv.putText(display_frame, "Finish all red items", (20, y+85), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # --- FEEDBACK OVERLAY ---
            if time.time() - feedback_timer < 2.0:
                cv.putText(display_frame, feedback_msg, (150, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            cv.imshow("Gesture Trainer", display_frame)
            
            # --- INPUT HANDLING ---
            key = cv.waitKey(1) & 0xFF
            
            if key == 27: # ESC
                print("Force Quit")
                break
                
            if not recording_active:
                char = '_' if key == 32 else chr(key) # Space = '_'
                
                if char in samples:
                    recording_active = True
                    target_char = char
                    frames_to_record = TARGET_SAMPLES
                    countdown_start = time.time()
                
                elif key == ord('c') or key == ord('C'):
                    # CLEAR DATA LOGIC
                    print("Clearing all data...")
                    reset_file(FILE_NAME) # Wipe file
                    for k in samples: samples[k] = 0 # Reset RAM counts
                    feedback_msg = "ALL DATA CLEARED"
                    feedback_timer = time.time()
                
                elif key == ord('q'):
                    if all_green:
                        print("Training Complete.")
                        break
                    else:
                        feedback_msg = "COLLECT MORE SAMPLES!"
                        feedback_timer = time.time()
                        # Flash Red
                        cv.rectangle(display_frame, (0,0), (640, 480), (0, 0, 255), 20)
                        cv.imshow("Gesture Trainer", display_frame)
                        cv.waitKey(100)

    finally:
        writer_thread.stop()
        stream.stop()
        cv.destroyAllWindows()
        print("Trainer Closed.")

if __name__ == "__main__":
    main()