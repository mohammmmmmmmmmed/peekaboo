from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import time

app = Flask(__name__)

cap = None
background = {color: None for color in [
    'blue', 'green', 'red', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'white', 'black'
]}

# Function to create the background
def create_background(cap, num_frames=30):
    print("Capturing background. Please move out of frame.")
    backgrounds = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            backgrounds.append(frame)
        else:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
        time.sleep(0.1)
    if backgrounds:
        return np.median(backgrounds, axis=0).astype(np.uint8)
    else:
        raise ValueError("Could not capture any frames for background")

# Function to create a mask
def create_mask(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return mask

# Function to apply cloak effect
def apply_cloak_effect(frame, mask, background):
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)
    return cv2.add(fg, bg)

# Function to generate frames
def generate_frames(color):
    global cap, background
    color_ranges = {
        'blue': ([90, 50, 50], [130, 255, 255]),
        'green': ([35, 50, 50], [85, 255, 255]),
        'red': ([0, 50, 50], [10, 255, 255]),
        'yellow': ([20, 50, 50], [30, 255, 255]),
        'cyan': ([80, 50, 50], [100, 255, 255]),
        'magenta': ([140, 50, 50], [160, 255, 255]),
        'orange': ([10, 50, 50], [20, 255, 255]),
        'purple': ([130, 50, 50], [160, 255, 255]),
        'brown': ([10, 50, 50], [20, 150, 150]),
        'pink': ([160, 50, 50], [170, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50])  # Black color range
    }
    lower_color, upper_color = color_ranges.get(color, ([0, 0, 0], [0, 0, 0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(1)
            continue

        if background[color] is not None:
            mask = create_mask(frame, np.array(lower_color), np.array(upper_color))
            result = apply_cloak_effect(frame, mask, background[color])
        else:
            result = frame

        ret, buffer = cv2.imencode('.jpg', result)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture')
def start_capture():
    global cap, background
    color = request.args.get('color', 'blue')

    if cap is None:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open camera.", 500

    try:
        background[color] = create_background(cap)
        return "Background captured successfully.", 200
    except ValueError as e:
        cap.release()
        cap = None
        background = {color: None for color in background}
        return f"Error: {e}", 500

@app.route('/stop_capture')
def stop_capture():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        return "Capture stopped and camera released.", 200
    else:
        return "No active capture to stop.", 400

@app.route('/video_feed/<color>')
def video_feed(color):
    if color in background:
        return Response(generate_frames(color),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Color not supported.", 400

if __name__ == "__main__":
    app.run(debug=True)
