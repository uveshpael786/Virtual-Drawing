import cv2
import numpy as np
import mediapipe as mp
import os
import time
import math

# --- Configuration & Constants ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
HEADER_HEIGHT = 100
SIDEBAR_WIDTH = 100

# Colors (BGR Format)
COLORS = {
    "magenta": (255, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0), # Eraser
}

# Default Settings
draw_color = COLORS["magenta"]
brush_thickness = 15
eraser_thickness = 50
smoothening = 5 # Higher = smoother but more lag

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)

# --- Canvas Setup ---
img_canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)

# State Variables
xp, yp = 0, 0 # Previous coordinates
curr_x, curr_y = 0, 0 # Current smoothed coordinates
prev_time = 0

def draw_ui(img, current_color, current_thickness):
    # --- Top Header (Colors) ---
    # Define color boxes
    box_width = (WINDOW_WIDTH - SIDEBAR_WIDTH) // len(COLORS)
    
    for i, (name, color) in enumerate(COLORS.items()):
        x1 = i * box_width
        x2 = x1 + box_width
        y1 = 0
        y2 = HEADER_HEIGHT
        
        # Highlight selected color
        if color == current_color:
            cv2.rectangle(img, (x1, y1), (x2, y2 + 10), color, cv2.FILLED)
            cv2.rectangle(img, (x1, y1), (x2, y2 + 10), (255, 255, 255), 3) # White border
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
            
        # Add labels for special tools
        if name == "black":
            cv2.putText(img, "Eraser", (x1 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Side Bar (Tools) ---
    # Clear Button
    cv2.rectangle(img, (WINDOW_WIDTH - SIDEBAR_WIDTH, 0), (WINDOW_WIDTH, 100), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, "CLEAR", (WINDOW_WIDTH - SIDEBAR_WIDTH + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save Button
    cv2.rectangle(img, (WINDOW_WIDTH - SIDEBAR_WIDTH, 100), (WINDOW_WIDTH, 200), (100, 100, 100), cv2.FILLED)
    cv2.putText(img, "SAVE", (WINDOW_WIDTH - SIDEBAR_WIDTH + 15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Size Indicators
    cv2.putText(img, f"Size: {current_thickness}", (WINDOW_WIDTH - SIDEBAR_WIDTH + 10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

print("Starting AI Virtual Painter...")
print("Controls:")
print("- Index Finger UP: Draw")
print("- Index + Middle UP: Hover/Select")
print("- Index + Middle + Ring UP: Change Brush Size (Hold to cycle)")
print("- Select colors from top bar")
print("- 'CLEAR' to reset canvas")
print("- 'SAVE' to save artwork")
print("- Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image
    img = cv2.flip(img, 1)
    
    # Hand Landmarks
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    lm_list = []
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * WINDOW_WIDTH), int(lm.y * WINDOW_HEIGHT)
                lm_list.append([id, cx, cy])

    if len(lm_list) != 0:
        # Tip coordinates
        x1, y1 = lm_list[8][1:]  # Index
        x2, y2 = lm_list[12][1:] # Middle
        x3, y3 = lm_list[16][1:] # Ring

        # Check fingers up
        fingers = []
        # Thumb
        if lm_list[4][1] < lm_list[3][1]: fingers.append(1)
        else: fingers.append(0)
        # 4 Fingers
        for id in [8, 12, 16, 20]:
            if lm_list[id][2] < lm_list[id - 2][2]: fingers.append(1)
            else: fingers.append(0)

        # --- Logic ---

        # 1. Selection Mode (Index + Middle)
        if fingers[1] and fingers[2] and not fingers[3]:
            xp, yp = 0, 0 # Reset drawing
            
            # Smoothing for selection cursor
            curr_x = int(xp + (x1 - xp) / smoothening) if xp != 0 else x1
            curr_y = int(yp + (y1 - yp) / smoothening) if yp != 0 else y1
            
            # Draw Selection Cursor
            cv2.rectangle(img, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), draw_color, cv2.FILLED)
            
            # --- Header Interaction ---
            if y1 < HEADER_HEIGHT:
                # Color Selection
                if x1 < WINDOW_WIDTH - SIDEBAR_WIDTH:
                    box_width = (WINDOW_WIDTH - SIDEBAR_WIDTH) // len(COLORS)
                    index = x1 // box_width
                    key = list(COLORS.keys())[index]
                    draw_color = COLORS[key]
            
            # --- Sidebar Interaction ---
            if x1 > WINDOW_WIDTH - SIDEBAR_WIDTH:
                # Clear
                if y1 < 100:
                    img_canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
                    cv2.putText(img, "CLEARED!", (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                # Save
                elif 100 < y1 < 200:
                    filename = f"artwork_{int(time.time())}.jpg"
                    # Merge canvas and image for saving
                    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
                    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                    img_final = cv2.bitwise_and(img, img_inv)
                    img_final = cv2.bitwise_or(img_final, img_canvas)
                    
                    cv2.imwrite(filename, img_final)
                    cv2.putText(img, "SAVED!", (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    time.sleep(0.5) # Debounce

        # 2. Drawing Mode (Index Only)
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            
            # Smoothing
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            # Smooth the line
            # We interpolate between previous point and current point
            # But for simple smoothing, we can just draw line from xp,yp to x1,y1
            # To make it even smoother, we could use a deque of points, but simple line is usually fast enough
            
            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            
            xp, yp = x1, y1

        # 3. Brush Size Mode (Index + Middle + Ring)
        elif fingers[1] and fingers[2] and fingers[3]:
            # Simple gesture: Hold 3 fingers up to cycle size
            # Or use distance between thumb and index?
            # Let's use distance between Index and Thumb for dynamic sizing!
            
            # Calculate distance
            length = math.hypot(x1 - lm_list[4][1], y1 - lm_list[4][2])
            
            # Map length to brush size (e.g., 20 to 200)
            brush_thickness = int(np.interp(length, [20, 200], [5, 100]))
            
            # Visual feedback
            cv2.circle(img, (x1, y1), brush_thickness // 2, draw_color, cv2.FILLED)
            cv2.putText(img, f"Size: {brush_thickness}", (x1+50, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
            
            xp, yp = 0, 0 # Don't draw while resizing

        else:
            xp, yp = 0, 0

    # --- Image Blending ---
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Draw UI on top
    draw_ui(img, draw_color, brush_thickness)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, WINDOW_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("AI Virtual Painter Pro", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
