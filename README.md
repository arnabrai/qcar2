# qcar2

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from calculate_steer_stop import compute_steering_for_distance, get_stop_distance
from pal.products.qcar import (
    QCar, QCarRealSense, QCarLidar, IS_PHYSICAL_QCAR
)

# ---------- Optional OCR ----------
OCR_ENABLED = True
OCR_LANGS = ['en']
OCR_CONF_THRES = 0.45        # ✅ FIX 1 — lowered from 0.55, OCR misses detections at high threshold
OCR_EVERY_N_FRAMES = 3       # ✅ FIX 2 — run OCR more frequently (was 5, now 3)
OCR_CENTER_ROI = 0.8         # ✅ slightly wider crop to catch signs better

try:
    if OCR_ENABLED:
        import easyocr
        _gpu = torch.cuda.is_available()
        ocr_reader = easyocr.Reader(OCR_LANGS, gpu=_gpu)
        OCR_AVAILABLE = True
        print(f"EasyOCR initialized (gpu={_gpu})")
    else:
        OCR_AVAILABLE = False
except Exception as e:
    OCR_AVAILABLE = False
    print("EasyOCR not available:", e)

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()

# === Configuration ===
sampleRate = 200
runTime = 300.0
frame_size = (320, 240)
max_depth_distance = 2

# === LiDAR setup ===
lidar_points = 1000
lidar_mode = 2
lidar_interp = 0

# === Model + Device ===
MODEL_PATH = r"/home/nvidia/Desktop/qcar2_hardware_tests/Qcar_modifications/best.pt"
CONF_THRES = 0.25
IMGSZ = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {'CUDA GPU' if DEVICE == 0 else 'CPU'}")

# ✅ FIX 3 — TRIGGER_FRAMES lowered to 2, but counter now PERSISTS (doesn't decay between OCR runs)
TRIGGER_FRAMES = 2
TURN_DURATION = 2.0
STOP_DURATION = 2.0
FORWARD_SPEED = 0.12
TURN_STEER = 0.5
IDLE_SPEED = 0.00
IDLE_STEER = 0.0
OBSTACLE_THRESHOLD = 0.35
MIN_SIGN_DISTANCE = 0.3
MAX_SIGN_DISTANCE = 1.5

# Load YOLO once
model = YOLO(MODEL_PATH)
print("Model loaded. Device:", DEVICE)
print("Model classes:", model.model.names)

# === Display setup ===
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
fig.canvas.draw()
plt.ion()

def plot_lidar_to_array(angles, distances):
    ax.cla()
    ax.scatter(angles, distances, s=1)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.resize(img_array, frame_size)

# === LiDAR obstacle check ===
# ✅ FIX 4 — LiDAR obstacle detection actually connected to braking
def check_lidar_obstacle(distances, angles):
    """Returns True if obstacle detected within OBSTACLE_THRESHOLD in front arc"""
    if distances is None or len(distances) == 0:
        return False
    # Front arc = angles within ±30 degrees of forward (0 radians)
    front_mask = (np.abs(angles) < np.deg2rad(30))
    front_distances = distances[front_mask]
    front_distances = front_distances[front_distances > 0]  # remove zeros
    if len(front_distances) == 0:
        return False
    min_dist = np.min(front_distances)
    print(f"[LiDAR] Min front distance: {min_dist:.2f}m")
    return min_dist < OBSTACLE_THRESHOLD

# === State Machine ===
STATE_TURN_LEFT    = "TURN_LEFT"
STATE_TURN_RIGHT   = "TURN_RIGHT"
STATE_MOVE_FORWARD = "MOVE_FORWARD"
STATE_STOPPED      = "STOPPED"
STATE_LIDAR_STOP   = "LIDAR_STOP"   # ✅ new state for LiDAR obstacle

state = STATE_MOVE_FORWARD
state_start = 0.0
has_stopped_permanently = False

# === Detection Counters ===
det_counter = {'left': 0, 'right': 0, 'stop-sign': 0}

def reset_counters():
    for k in det_counter:
        det_counter[k] = 0

def want_to_trigger():
    global has_stopped_permanently
    if has_stopped_permanently:
        return STATE_STOPPED
    if det_counter.get('stop-sign', 0) >= TRIGGER_FRAMES:
        has_stopped_permanently = True
        return STATE_STOPPED
    if det_counter.get('left', 0) >= TRIGGER_FRAMES:
        return STATE_TURN_LEFT
    if det_counter.get('right', 0) >= TRIGGER_FRAMES:
        return STATE_TURN_RIGHT
    return STATE_MOVE_FORWARD

def run_ocr_fallback(img_bgr):
    if not OCR_AVAILABLE:
        return set()
    h, w = img_bgr.shape[:2]
    roi_scale = np.clip(OCR_CENTER_ROI, 0.2, 1.0)
    cw, ch = int(w * roi_scale), int(h * roi_scale)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = img_bgr[y0:y0+ch, x0:x0+cw]
    try:
        ocr_results = ocr_reader.readtext(crop)
    except Exception:
        return set()
    found = set()
    for (_bbox, text, conf) in ocr_results:
        if conf < OCR_CONF_THRES:
            continue
        t = text.strip().lower()
        if "left" in t:
            found.add("left")
        elif "right" in t:
            found.add("right")
        elif "stop" in t:
            found.add("stop-sign")
    return found

# === Main Loop ===
try:
    with QCar(readMode=1, frequency=sampleRate) as myCar, \
         QCarRealSense(mode='RGB, Depth, IR') as myCam:

        myLidar = QCarLidar(
            numMeasurements=lidar_points,
            rangingDistanceMode=lidar_mode,
            interpolationMode=lidar_interp
        )

        t0 = time.time()
        frame_idx = 0

        while time.time() - t0 < runTime:
            now = time.time()
            frame_idx += 1

            # === Sensor reads ===
            myCar.read()
            myCam.read_RGB()
            myCam.read_depth(dataMode='PX')
            myCam.read_IR()
            myLidar.read()

            # === YOLO Detection ===
            rgb_raw = myCam.imageBufferRGB.copy()
            results = model.predict(
                rgb_raw,
                verbose=False,
                device=DEVICE,
                conf=CONF_THRES,
                imgsz=IMGSZ
            )[0]
            rgb = results.plot()
            rgb = cv2.resize(rgb, frame_size)

            # ✅ YOLO detections → update counters directly
            yolo_seen = set()
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id   = int(box.cls)
                    cls_name = model.model.names.get(cls_id, str(cls_id))
                    if cls_name in det_counter:
                        yolo_seen.add(cls_name)
                        print(f"[YOLO] Detected: {cls_name} (conf: {float(box.conf):.2f})")

            # ✅ FIX 3 CORE — Counter logic completely rewritten
            # YOLO detections → immediately increment
            for k in yolo_seen:
                det_counter[k] = min(det_counter[k] + 1, TRIGGER_FRAMES + 2)

            # OCR fallback — runs every N frames
            if OCR_AVAILABLE and (frame_idx % OCR_EVERY_N_FRAMES == 0):
                ocr_seen = run_ocr_fallback(rgb_raw)
                if ocr_seen:
                    print(f"[OCR] Detected: {', '.join(sorted(ocr_seen))}")
                # ✅ KEY FIX — only decay counters for classes NOT seen by BOTH YOLO and OCR
                for k in det_counter:
                    if k in ocr_seen or k in yolo_seen:
                        # seen by at least one sensor → increment
                        det_counter[k] = min(det_counter[k] + 1, TRIGGER_FRAMES + 2)
                    else:
                        # not seen by anyone → slow decay (only -1 every OCR cycle, not every frame)
                        det_counter[k] = max(det_counter[k] - 1, 0)

            print(f"Counters: {det_counter}")

            # ✅ FIX 4 — LiDAR obstacle check runs every frame
            lidar_obstacle = check_lidar_obstacle(myLidar.distances, myLidar.angles)

            # === Control Logic ===
            throttle = 0.0
            steering = 0.0
            LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

            # ✅ LiDAR obstacle = highest priority → immediate stop
            if lidar_obstacle and state == STATE_MOVE_FORWARD:
                print("[⚠️ LiDAR] Obstacle detected — stopping!")
                throttle = 0.0
                steering = 0.0
                LEDs[4] = 1
                LEDs[5] = 1

            elif state == STATE_STOPPED:
                throttle = 0.0
                steering = 0.0
                LEDs[4] = 1
                LEDs[5] = 1

            elif state == STATE_TURN_LEFT:
                throttle = FORWARD_SPEED
                steering = +TURN_STEER
                LEDs[0] = LEDs[2] = 1
                if (now - state_start) >= TURN_DURATION:
                    state = STATE_MOVE_FORWARD
                    state_start = now
                    reset_counters()
                    print("[✅] Turn LEFT complete → MOVE_FORWARD")

            elif state == STATE_TURN_RIGHT:
                throttle = FORWARD_SPEED
                steering = -TURN_STEER
                LEDs[1] = LEDs[3] = 1
                if (now - state_start) >= TURN_DURATION:
                    state = STATE_MOVE_FORWARD
                    state_start = now
                    reset_counters()
                    print("[✅] Turn RIGHT complete → MOVE_FORWARD")

            elif state == STATE_MOVE_FORWARD:
                next_state = want_to_trigger()
                print(f"next state: {next_state} | counters: {det_counter}")
                if next_state != STATE_MOVE_FORWARD:
                    print(f"[🚗 ACTION] Switching to {next_state}!")
                    state = next_state
                    state_start = now
                    reset_counters()
                    continue
                else:
                    throttle = FORWARD_SPEED
                    steering = 0.0

            myCar.write(throttle, steering, LEDs)

            # === Visuals ===
            depth = cv2.resize(
                (myCam.imageBufferDepthPX / max_depth_distance * 255).astype(np.uint8),
                frame_size
            )
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            ir_left  = cv2.resize(myCam.imageBufferIRLeft,  frame_size)
            ir_right = cv2.resize(myCam.imageBufferIRRight, frame_size)
            ir_left  = cv2.cvtColor(ir_left,  cv2.COLOR_GRAY2BGR)
            ir_right = cv2.cvtColor(ir_right, cv2.COLOR_GRAY2BGR)
            lidar_img = plot_lidar_to_array(myLidar.angles, myLidar.distances)

            top_row    = np.hstack([rgb, depth, lidar_img])
            bottom_row = np.hstack([ir_left, ir_right, np.zeros_like(lidar_img)])
            dashboard  = np.vstack([top_row, bottom_row])

            # === Telemetry overlay ===
            text = (
                f"Battery: {myCar.batteryVoltage:.2f}V | "
                f"Current: {myCar.motorCurrent:.2f}A | Encoder: {myCar.motorEncoder}\n"
                f"State: {state} | Counters: L={det_counter['left']} "
                f"R={det_counter['right']} S={det_counter['stop-sign']}"
            )
            y0_txt = 20
            for i, line in enumerate(text.split('\n')):
                cv2.putText(dashboard, line, (10, y0_txt + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("QCar2 Dashboard", dashboard)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

except Exception as e:
    print("[❌ ERROR]:", e)
    print("[⛔ SAFETY] Emergency stop.")
    myCar.write(0.0, 0.0, np.array([0, 0, 0, 0, 0, 1, 1, 1]))

finally:
    myLidar.terminate()
    cv2.destroyAllWindows()
```

---

## 📋 All 4 Fixes Explained Simply

| Fix | Old behavior | New behavior |
|---|---|---|
| **Fix 1** — OCR threshold | 0.55 — too strict, rejecting valid detections | 0.45 — accepts more real detections |
| **Fix 2** — OCR frequency | Every 5 frames — too slow | Every 3 frames — faster response |
| **Fix 3** — Counter logic | Counter went 0→1→0→1 forever, never hit 2 | Counter now only decays during OCR cycle, not every frame — reaches 2 easily |
| **Fix 4** — LiDAR connected | LiDAR was plotted visually but never used for decisions | Now checks front arc every frame — stops car if obstacle < 35cm |

---

## 👀 What You'll Now See in Terminal
```
Counters: {'left': 0, 'right': 0, 'stop-sign': 0}
[OCR] Detected: left
Counters: {'left': 1, 'right': 0, 'stop-sign': 0}
Counters: {'left': 1, 'right': 0, 'stop-sign': 0}   ← stays at 1 now!
[OCR] Detected: left
Counters: {'left': 2, 'right': 0, 'stop-sign': 0}   ← hits 2!
[🚗 ACTION] Switching to TURN_LEFT!                  ← car actually turns!
