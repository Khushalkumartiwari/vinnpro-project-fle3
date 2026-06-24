
from ultralytics import YOLO

VEHICLE_CLASS_IDS = {2:"car", 5:"bus", 6:"train", 7:"truck"}
FOREIGN_CLASS_IDS = {
    0: "person",
    56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 24: "backpack",
    26: "handbag", 28: "suitcase", 39: "bottle",
    41: "cup", 67: "cell phone", 73: "book",
    76: "scissors", 77: "teddy bear",
}
MIN_AREA_FRACTION = 0.01

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.20, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr, view="back"):
        h, w = frame_bgr.shape[:2]
        frame_area = h * w
        results = self.model.predict(frame_bgr, conf=self.conf, device=self.device, verbose=False)
        vehicles = []
        foreign = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                confidence = float(box.conf[0])
                cx = int((x1 + x2) / 2)

                if cls_id in VEHICLE_CLASS_IDS:
                    if area / frame_area < 0.05:
                        continue
                    contact_points = [
                        (int(x1), int(y2)),
                        (cx,      int(y2)),
                        (int(x2), int(y2)),
                    ]
                    vehicles.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "area": area,
                        "label": VEHICLE_CLASS_IDS[cls_id],
                        "contact_points": contact_points,
                    })
                elif cls_id in FOREIGN_CLASS_IDS:
                    if area / frame_area < MIN_AREA_FRACTION:
                        continue
                    contact_points = [
                        (int(x1), int(y2)),
                        (cx,      int(y2)),
                        (int(x2), int(y2)),
                    ]
                    foreign.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "area": area,
                        "label": FOREIGN_CLASS_IDS[cls_id],
                        "contact_points": contact_points,
                    })

        # Keep only largest vehicle
        vehicles.sort(key=lambda d: d["area"], reverse=True)
        return vehicles[:1], foreign
