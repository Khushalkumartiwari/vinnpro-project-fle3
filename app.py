
import os, uuid, cv2
from flask import Flask, render_template, request, send_from_directory
from boundary_detector import BoundaryDetector
from vehicle_detector import VehicleDetector
from dharam_katta_monitor import evaluate_frame, annotate

app = Flask(__name__)
UPLOAD_DIR = "web_uploads"
RESULT_DIR = "web_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Lazy load - dont load at startup, load on first request
_vehicle_det = None
_boundary_dets = {}

def get_vehicle_det():
    global _vehicle_det
    if _vehicle_det is None:
        _vehicle_det = VehicleDetector(model_path="yolov8n.pt")
    return _vehicle_det

def get_boundary_det(view):
    if view not in _boundary_dets:
        _boundary_dets[view] = BoundaryDetector(view=view)
    return _boundary_dets[view]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("image")
        view = request.form.get("view", "back")
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
            uid = uuid.uuid4().hex[:8]
            upload_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
            file.save(upload_path)
            frame = cv2.imread(upload_path)
            if frame is None:
                result = {"error": "Could not read image. Try JPG or PNG."}
            else:
                polygon, _ = get_boundary_det(view).detect(frame)
                vehicles, foreign = get_vehicle_det().detect(frame, view=view)
                alarm, reasons = evaluate_frame(polygon, vehicles, foreign, margin_px=10)
                annotated = annotate(frame, polygon, vehicles, foreign, alarm)
                result_filename = f"{uid}_result.jpg"
                cv2.imwrite(os.path.join(RESULT_DIR, result_filename), annotated)
                result = {
                    "alarm": alarm,
                    "reasons": reasons,
                    "image_url": f"/result/{result_filename}",
                    "view": view,
                }
        else:
            result = {"error": "No file selected."}
    return render_template("index.html", result=result)

@app.route("/result/<filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
