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
TARGET_SAMPLES = 20
WINDOW_WIDTH = 800  # Expanded for better UI
WINDOW_HEIGHT = 600

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
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
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
    # REMOVED '='
    counts = {'+': 0, '-': 0, '*': 0, '/': 0, '_': 0}
    if os.path.exists(filename):
        try:
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
    with open(filename, 'w', newline='') as f:
        csv.writer(f).writerow(['label'] + [f'v{i}' for i in range(42)])

# --- UI HELPER: Draw Text with Background ---
def draw_text_with_bg(img, text, x, y, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1, padding=5):
    """Draws text with a filled rectangle background for visibility."""
    font = cv.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)
    
    # Draw Background Rectangle
    cv.rectangle(img, (x - padding, y - text_h - padding), (x + text_w + padding, y + baseline + padding), bg_color, -1)
    
    # Draw Text
    cv.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv.LINE_AA)

# ==========================================
# 4. MAIN TRAINER LOOP
# ==========================================
def main():
    print("--- LAUNCHING GESTURE TRAINER ---")
    
    if not os.path.exists(FILE_NAME):
        reset_file(FILE_NAME)
    
    samples = load_existing_counts(FILE_NAME)

    writer_thread = DataWriter(FILE_NAME)
    stream = ThreadedCamera(0).start()
    time.sleep(1.0) 

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    recording_active = False
    target_char = None
    frames_to_record = 0
    countdown_start = 0
    countdown_duration = 3.0
    feedback_msg = ""
    feedback_timer = 0

    print("Ready. Switch to window.")

    try:
        while True:
            status, frame = stream.read()
            if not status: break
            
            display_frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # --- UI HEADER ---
            # Main Title Box
            draw_text_with_bg(display_frame, "GESTURE TRAINER", 20, 50, 1.0, (255, 255, 255), (50, 50, 50), 2, 10)
            draw_text_with_bg(display_frame, "Press keys to train gestures", 20, 90, 0.6, (200, 200, 200), (50, 50, 50), 1, 5)
            
            # --- SAMPLE LIST ---
            y_start = 140
            line_spacing = 40 
            all_green = True
            
            # REMOVED '='
            ui_items = [
                ('+', '+', 'Plus'),
                ('-', '-', 'Minus'),
                ('*', '*', 'Multiply'),
                ('/', '/', 'Divide'),
                ('_', 'Space', 'Start/Reset')
            ]

            for i, (char, key_name, label) in enumerate(ui_items):
                count = samples.get(char, 0)
                is_ready = count >= (TARGET_SAMPLES * 2)
                if not is_ready: all_green = False
                
                # Dynamic Colors for text and background
                text_col = (0, 255, 0) if is_ready else (0, 0, 255)
                bg_col = (20, 20, 20) # Dark grey bg for list items
                
                text = f"{label} (Press {key_name}): {count}"
                
                # Draw list item with background
                draw_text_with_bg(display_frame, text, 20, y_start + (i * line_spacing), 0.7, text_col, bg_col, 2)

            # Process Hand
            current_features = None
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display_frame, landmarks, mp_hands.HAND_CONNECTIONS)
                current_features = get_features(landmarks)

            # --- RECORDING LOGIC ---
            if recording_active:
                elapsed = time.time() - countdown_start
                
                # Center calculation for prompts
                center_x = WINDOW_WIDTH // 2
                center_y = WINDOW_HEIGHT // 2
                
                if elapsed < countdown_duration:
                    sec = int(countdown_duration - elapsed) + 1
                    msg = f"HOLD POSE: {sec}"
                    # Calculate text width to center it manually if needed, or guess x
                    # Simple centering for monospaced look
                    draw_text_with_bg(display_frame, msg, center_x - 100, center_y, 1.5, (0, 255, 255), (0, 0, 0), 3, 20)
                
                elif frames_to_record > 0:
                    msg = "CAPTURING..."
                    draw_text_with_bg(display_frame, msg, center_x - 120, center_y, 1.5, (0, 0, 255), (0, 0, 0), 3, 20)
                    
                    if current_features is not None:
                        batch = []
                        batch.append([target_char] + list(current_features))
                        # Mirror
                        feats_mirror = current_features.copy()
                        feats_mirror[0::2] = feats_mirror[0::2] * -1
                        batch.append([target_char] + list(feats_mirror))
                        
                        writer_thread.save(batch)
                        samples[target_char] += 2
                        frames_to_record -= 1
                        
                        # Border Flash
                        cv.rectangle(display_frame, (0,0), (WINDOW_WIDTH, WINDOW_HEIGHT), (255, 255, 255), 20)
                else:
                    recording_active = False
                    target_char = None
            
            # --- FOOTER INSTRUCTIONS ---
            else:
                footer_y = WINDOW_HEIGHT - 60
                bg_col = (50, 50, 50)
                
                draw_text_with_bg(display_frame, "Press 'C' to Clear All | 'Q' to Check & Quit", 20, footer_y, 0.6, (200, 200, 200), bg_col, 1, 5)
                draw_text_with_bg(display_frame, "Press 'ESC' to Force Quit", 20, footer_y + 30, 0.6, (200, 200, 200), bg_col, 1, 5)

            # --- FEEDBACK OVERLAY ---
            if time.time() - feedback_timer < 2.0:
                draw_text_with_bg(display_frame, feedback_msg, (WINDOW_WIDTH // 2) - 100, WINDOW_HEIGHT - 120, 1.0, (0, 255, 255), (0,0,0), 2, 10)

            cv.imshow("Gesture Trainer", display_frame)
            
            key = cv.waitKey(1) & 0xFF
            
            if key == 27: # ESC
                print("Force Quit")
                break
                
            if not recording_active:
                char = '_' if key == 32 else chr(key)
                
                # Check for keys (REMOVED '=')
                if char in ['+', '-', '*', '/', '_']:
                    recording_active = True
                    target_char = char
                    frames_to_record = TARGET_SAMPLES
                    countdown_start = time.time()
                
                elif key == ord('c') or key == ord('C'):
                    print("Clearing all data...")
                    reset_file(FILE_NAME)
                    for k in samples: samples[k] = 0
                    feedback_msg = "ALL DATA CLEARED"
                    feedback_timer = time.time()
                
                elif key == ord('q') or key == ord('Q'):
                    if all_green:
                        print("Training Complete. Exiting.")
                        break
                    else:
                        feedback_msg = "COLLECT MORE SAMPLES!"
                        feedback_timer = time.time()
                        # Red Border Flash
                        cv.rectangle(display_frame, (0,0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 255), 20)
                        cv.imshow("Gesture Trainer", display_frame)
                        cv.waitKey(100)

    finally:
        writer_thread.stop()
        stream.stop()
        cv.destroyAllWindows()
        print("Trainer Closed.")

if __name__ == "__main__":
    main()