import cv2 as cv
import mediapipe as mp
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class HandVisualizer3D:
    def __init__(self):
        # 1. Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1, # Start with 1 hand for clarity in 3D
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hand_connections = list(self.mp_hands.HAND_CONNECTIONS)

        # 2. Initialize Camera
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # 3. Initialize PyQtGraph 3D OpenGL Widget
        self.app = pg.mkQApp("3D Hand Visualizer")
        self.view = gl.GLViewWidget()
        self.view.show()
        self.view.setWindowTitle('Real-time 3D Gesture Visualization')
        self.view.setCameraPosition(distance=0.5, elevation=30, azimuth=45)

        # Add a grid for spatial context
        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1) # Scale grid down to match hand metrics
        self.view.addItem(grid)

        # Create Scatter Plot for Joints (Dots)
        # Initial positions at 0,0,0
        pos = np.zeros((21, 3))
        color = (0, 1, 0, 1) # Green, opaque
        size = 5 # Point size
        self.scatter_item = gl.GLScatterPlotItem(pos=pos, color=color, size=size, pxMode=True)
        self.view.addItem(self.scatter_item)

        # Create Line Plot for Bones (Connections)
        self.line_item = gl.GLLinePlotItem(mode='lines', color=(1, 1, 0, 1), width=2, antialias=True)
        self.view.addItem(self.line_item)
        
        # Variables to store data between frames
        self.landmarks_3d = None

        # Setup update timer (runs the main loop)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        # Update as fast as possible (0ms delay implies execute when idle)
        self.timer.start(0) 

    def update(self):
        """Main loop tick: Read camera, process MP, update 3D view."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process hand
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hand_detected = False
        
        # Use WORLD landmarks for true 3D geometry (meters relative to wrist)
        if results.multi_hand_world_landmarks:
            hand_detected = True
            landmarks = results.multi_hand_world_landmarks[0]
            
            # 1. Extract N,3 numpy array of coordinates
            points = []
            for lm in landmarks.landmark:
                # Note: We might flip axes here depending on preference. 
                # MediaPipe World: X=right, Y=up, Z=towards camera (from wrist)
                # We negate Z to make it feel more natural in the GL view
                points.append([lm.x, -lm.y, -lm.z]) 
            
            self.landmarks_3d = np.array(points)

            # 2. Update Scatter Plot (Joints)
            self.scatter_item.setData(pos=self.landmarks_3d)

            # 3. Update Line Plot (Bones)
            # We need to define start and end points for segments
            lines_start = []
            lines_end = []
            for connection in self.hand_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                lines_start.append(self.landmarks_3d[start_idx])
                lines_end.append(self.landmarks_3d[end_idx])
            
            # Interleave start and end points for GLLinePlotItem 'lines' mode
            # Format: p1_start, p1_end, p2_start, p2_end, ...
            interleaved_lines = np.empty((len(lines_start) * 2, 3), dtype=np.float32)
            interleaved_lines[0::2] = lines_start
            interleaved_lines[1::2] = lines_end
            
            self.line_item.setData(pos=interleaved_lines)

        # Optional: Show 2D webcam feed for comparison
        cv.imshow('2D Webcam Feed (For Reference)', frame)

    def run(self):
        # Start Qt event loop
        pg.exec()
        # Cleanup on exit
        self.cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    visualizer = HandVisualizer3D()
    print("3D Visualization running. Use your mouse to rotate the 3D view.")
    visualizer.run()