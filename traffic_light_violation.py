from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

VIDEO = "helloo.mp4"
DETECT_EVERY_N_FRAMES = 3
STREAM_FPS = 15
YOLO_IMGSZ = 480
JPEG_QUALITY = 75
VEHICLE_CLASSES = [2, 3, 5, 7]
stop_line = {"x1": 0.05, "y1": 0.65, "x2": 0.95, "y2": 0.65}

car_model = YOLO("yolov8n.pt")
tl_model = YOLO("best_traffic_nano_yolo.pt")

cap = cv2.VideoCapture(VIDEO)

paused = False
frame_lock = None
last_capture = 0


def get_stop_line_points(width, height):
    return (
        int(stop_line["x1"] * width),
        int(stop_line["y1"] * height),
        int(stop_line["x2"] * width),
        int(stop_line["y2"] * height),
    )


def vehicle_crossed_stop_line(box, width, height):
    x1, y1, x2, y2 = box
    lx1, ly1, lx2, ly2 = get_stop_line_points(width, height)

    if lx1 == lx2:
        return False

    vehicle_center_x = (x1 + x2) / 2
    min_line_x = min(lx1, lx2)
    max_line_x = max(lx1, lx2)

    if vehicle_center_x < min_line_x or vehicle_center_x > max_line_x:
        return False

    ratio = (vehicle_center_x - lx1) / (lx2 - lx1)
    line_y_at_vehicle = ly1 + ratio * (ly2 - ly1)

    return y2 >= line_y_at_vehicle


def detect_traffic_light(frame):
    tl_res = tl_model(frame, conf=0.3, imgsz=YOLO_IMGSZ, verbose=False)[0]
    light = "none"
    traffic_boxes = []

    for b in tl_res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        light = tl_model.model.names[cls]
        traffic_boxes.append((x1, y1, x2, y2, light))

    return light, traffic_boxes


def detect_vehicles(frame):
    res = car_model(
        frame,
        conf=0.4,
        imgsz=YOLO_IMGSZ,
        classes=VEHICLE_CLASSES,
        verbose=False,
    )[0]
    vehicles = []

    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        name = car_model.model.names[cls]
        vehicles.append((x1, y1, x2, y2, name))

    return vehicles


def generate():
    global paused, frame_lock, last_capture
    frame_count = 0
    light = "none"
    traffic_boxes = []
    vehicles = []
    frame_interval = 1 / STREAM_FPS

    while True:
        start_time = time.time()

        # ===== READ FRAME =====
        if not paused:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_lock = frame.copy()

        if frame_lock is None:
            continue

        raw_frame = frame_lock.copy()
        frame = raw_frame.copy()
        height, width = frame.shape[:2]
        line_x1, line_y1, line_x2, line_y2 = get_stop_line_points(width, height)

        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            light, traffic_boxes = detect_traffic_light(raw_frame)
            vehicles = detect_vehicles(raw_frame)

        frame_count += 1

        cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (255, 255, 255), 3)
        cv2.putText(frame, "STOP LINE", (line_x1, line_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ================= TRAFFIC LIGHT =================
        for x1, y1, x2, y2, detected_light in traffic_boxes:
            tl_color = (0, 255, 0)
            if detected_light == "red":
                tl_color = (0, 0, 255)
            elif detected_light == "yellow":
                tl_color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), tl_color, 2)
            cv2.putText(frame, detected_light, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tl_color, 2)

        cv2.putText(frame, f"Light: {light}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255) if light=="red" else (0,255,0), 2)

        # ================= VEHICLE =================
        has_violation = False

        for x1, y1, x2, y2, name in vehicles:
            crossed_stop_line = vehicle_crossed_stop_line((x1, y1, x2, y2), width, height)

            color = (0,255,0)

            if name == "motorcycle":
                color = (255,255,0)

            if light == "red" and crossed_stop_line:
                color = (0,0,255)
                has_violation = True

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,name,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            # ===== CẢNH BÁO ĐÈN ĐỎ (KHÔNG MẤT) =====
            if light == "red" and crossed_stop_line:
                cv2.putText(frame,
                            "CẢNH BÁO ĐÈN ĐỎ",
                            (x1,y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

        # ===== TỰ ĐỘNG CHỤP VI PHẠM ĐÈN ĐỎ =====
        if has_violation:
            current_time = time.time()
            if current_time - last_capture > 5:
                os.makedirs("vi_pham", exist_ok=True)
                filename = f"vi_pham/auto_violation_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)
                last_capture = current_time

        # ================= STREAM =================
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


# ================= ROUTES =================

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/line', methods=['GET', 'POST'])
def line():
    global stop_line

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}

        try:
            new_line = {
                "x1": float(data["x1"]),
                "y1": float(data["y1"]),
                "x2": float(data["x2"]),
                "y2": float(data["y2"]),
            }
        except (KeyError, TypeError, ValueError):
            return jsonify({"status": "error", "message": "invalid line"}), 400

        for key, value in new_line.items():
            new_line[key] = max(0.0, min(1.0, value))

        stop_line = new_line
        return jsonify({"status": "saved", "line": stop_line})

    return jsonify(stop_line)


@app.route('/pause')
def pause():
    global paused
    paused = True
    return jsonify({"status": "paused"})


@app.route('/resume')
def resume():
    global paused
    paused = False
    return jsonify({"status": "running"})


@app.route('/capture')
def capture():
    if frame_lock is not None:
        os.makedirs("vi_pham", exist_ok=True)
        filename = f"vi_pham/capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame_lock)
        return jsonify({"status": "saved", "file": filename})
    return jsonify({"status": "no frame"})


if __name__ == "__main__":
    os.makedirs("vi_pham", exist_ok=True)
    print("RUN http://127.0.0.1:5000")
    app.run(debug=False, threaded=True)
