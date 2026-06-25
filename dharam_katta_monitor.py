
import cv2, numpy as np
from boundary_detector import BoundaryDetector
from vehicle_detector import VehicleDetector

def is_inside(polygon, point, margin_px=0):
    polygon = polygon.astype(np.float32)
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True)
    return result >= -margin_px

def evaluate_frame(platform_polygon, vehicles, foreign, margin_px=10):
    reasons = []
    alarm = False

    if platform_polygon is None:
        reasons.append("WARNING: boundary not configured")
        return alarm, reasons

    for obj in foreign:
        inside_points = [p for p in obj["contact_points"]
                         if is_inside(platform_polygon, p, margin_px)]
        if inside_points:
            alarm = True
            reasons.append(
                f"ALARM: {obj['label']} detected ON platform")

    if not vehicles:
        if not alarm:
            reasons.append("OK: no vehicle on platform")
    else:
        for v in vehicles:
            outside = [p for p in v["contact_points"]
                       if not is_inside(platform_polygon, p, margin_px)]
            tyre_info = " (tyre position)" if v.get("tyre_detected") else " (bbox bottom)"
            if outside:
                alarm = True
                reasons.append(
                    f"ALARM: {v['label']} wheel crosses boundary{tyre_info}")
            else:
                reasons.append(
                    f"OK: {v['label']} wheels within boundary{tyre_info}")

    if not reasons:
        reasons.append("OK: platform clear")
    return alarm, reasons

def annotate(frame, platform_polygon, vehicles, foreign, alarm):
    out = frame.copy()
    if platform_polygon is not None:
        color = (0, 0, 200) if alarm else (0, 200, 0)
        overlay = out.copy()
        cv2.fillPoly(overlay, [platform_polygon], color)
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
        cv2.polylines(out, [platform_polygon], True, color, 2)

    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (255,128,0), 2)
        cv2.putText(out, f"{v['label']} {v['confidence']:.2f}",
                    (x1, max(0,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,128,0), 2)
        for p in v["contact_points"]:
            # Green dot = tyre detected, Yellow = bbox fallback
            dot_color = (0, 255, 0) if v.get("tyre_detected") else (0, 255, 255)
            cv2.circle(out, p, 8, dot_color, -1)
            cv2.circle(out, p, 8, (0,0,0), 1)  # outline

    for obj in foreign:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,50,255), 2)
        cv2.putText(out, f"{obj['label']} {obj['confidence']:.2f}",
                    (x1, max(0,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,100,255), 2)

    label = "ALARM" if alarm else "OK"
    color = (0,0,255) if alarm else (0,200,0)
    cv2.rectangle(out, (10,10), (220,65), (0,0,0), -1)
    cv2.putText(out, label, (20,52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return out
