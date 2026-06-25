
from ultralytics import YOLO
import cv2
import numpy as np

VEHICLE_CLASS_IDS = {2:"car", 5:"bus", 6:"train", 7:"truck"}
FOREIGN_CLASS_IDS = {
    0: "person", 56: "chair", 57: "couch", 59: "bed",
    60: "dining table", 24: "backpack", 26: "handbag", 28: "suitcase",
}
MIN_AREA_FRACTION = 0.01

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.20, device=None):
        import torch
        self.model = YOLO(model_path)
        self.model.to("cpu")
        self.conf = conf
        self.device = device

    def _find_tyre_contact_points(self, frame_bgr, bbox):
        """
        Find tyre contact points using dark blob detection.
        Returns multiple points across the width of each detected tyre.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        # Search bottom 45% of bbox for tyres
        search_y1 = int(y1 + bbox_h * 0.55)
        search_y2 = min(y2 + int(bbox_h * 0.05), h)
        search_x1 = max(0, x1)
        search_x2 = min(w, x2)

        region = frame_bgr[search_y1:search_y2, search_x1:search_x2]
        if region.size == 0:
            return [], False

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Tyres are very dark
        _, dark = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((7, 7), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        region_area = (search_y2-search_y1) * (search_x2-search_x1)
        min_area = region_area * 0.03

        tyre_contact_points = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.15:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / float(rh + 1e-5)
            if aspect < 0.3 or aspect > 3.0:
                continue

            # Contact point = bottom of tyre in full image coords
            tyre_bottom_y = search_y1 + ry + rh
            tyre_left_x   = search_x1 + rx
            tyre_right_x  = search_x1 + rx + rw
            tyre_cx        = search_x1 + rx + rw // 2

            # Add 3 points per tyre: left edge, center, right edge
            # This ensures if ANY part of tyre is outside → alarm
            tyre_contact_points.append((tyre_left_x,  tyre_bottom_y))
            tyre_contact_points.append((tyre_cx,       tyre_bottom_y))
            tyre_contact_points.append((tyre_right_x,  tyre_bottom_y))

        return tyre_contact_points, len(tyre_contact_points) > 0

    def detect(self, frame_bgr, view="back"):
        h, w = frame_bgr.shape[:2]
        frame_area = h * w
        results = self.model.predict(
            frame_bgr, conf=self.conf, device=self.device, verbose=False)

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

                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    tyre_points, tyre_found = self._find_tyre_contact_points(
                        frame_bgr, bbox)

                    if tyre_found:
                        contact_points = tyre_points
                    else:
                        # Fallback: 5 points along bottom edge
                        contact_points = [
                            (int(x1),             int(y2)),
                            (int(x1+(x2-x1)*0.25),int(y2)),
                            (cx,                  int(y2)),
                            (int(x1+(x2-x1)*0.75),int(y2)),
                            (int(x2),             int(y2)),
                        ]

                    vehicles.append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "area": area,
                        "label": VEHICLE_CLASS_IDS[cls_id],
                        "contact_points": contact_points,
                        "tyre_detected": tyre_found,
                    })

                elif cls_id in FOREIGN_CLASS_IDS:
                    if area / frame_area < MIN_AREA_FRACTION:
                        continue
                    foreign.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "area": area,
                        "label": FOREIGN_CLASS_IDS[cls_id],
                        "contact_points": [
                            (int(x1), int(y2)),
                            (cx,      int(y2)),
                            (int(x2), int(y2)),
                        ],
                    })

        vehicles.sort(key=lambda d: d["area"], reverse=True)
        return vehicles[:1], foreign
