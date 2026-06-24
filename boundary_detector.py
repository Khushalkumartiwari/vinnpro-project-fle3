
import cv2, numpy as np, json, os

CONFIG_FILE = "boundary_config.json"
DEFAULT_CONFIG = {
    "front": {"p1": [0, 422], "p2": [639, 422]},
    "back":  {"p1": [0, 422], "p2": [639, 422]},
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

class BoundaryDetector:
    def __init__(self, view="front"):
        assert view in ("front", "back")
        self.view = view
        self.config = load_config()

    def _get_line(self, h, w):
        cfg = self.config.get(self.view, DEFAULT_CONFIG[self.view])
        if "p1" in cfg and "p2" in cfg:
            p1 = tuple(cfg["p1"])
            p2 = tuple(cfg["p2"])
            if p2[0] != p1[0]:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                y0 = int(p1[1] - slope * p1[0])
                y1 = int(y0 + slope * (w - 1))
                return (0, y0), (w - 1, y1)
            return p1, p2
        y = int(cfg.get("y_fraction", 0.88) * h)
        return (0, y), (w - 1, y)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        p1, p2 = self._get_line(h, w)
        polygon = np.array([
            [0,      0],
            [w - 1,  0],
            [p2[0],  p2[1]],
            [p1[0],  p1[1]],
        ], dtype=np.int32)
        debug = frame_bgr.copy()
        danger = np.array([[p1[0],p1[1]],[p2[0],p2[1]],[w-1,h-1],[0,h-1]], dtype=np.int32)
        overlay = debug.copy()
        cv2.fillPoly(overlay, [danger], (0, 0, 160))
        cv2.addWeighted(overlay, 0.3, debug, 0.7, 0, debug)
        cv2.line(debug, p1, p2, (0, 100, 255), 3)
        cv2.putText(debug, f"{self.view.upper()} STOP LINE",
                    (p1[0]+5, max(p1[1]-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return polygon, debug
