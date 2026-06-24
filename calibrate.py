
import cv2, json, os, argparse
import numpy as np

CONFIG_FILE = "boundary_config.json"
points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"  Point {len(points)}: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--view", required=True, choices=["front","back"])
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Cannot read: {args.image}"); return

    h, w = frame.shape[:2]
    print(f"Image: {w}x{h}  |  View: {args.view.upper()}")
    print("CLICK 2 POINTS on the stop line (left side first, then right side)")
    print("R=reset clicks | S=save | Q=quit")

    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    # Load existing line if any
    saved = config.get(args.view, {})
    global points
    if "p1" in saved and "p2" in saved:
        points = [tuple(saved["p1"]), tuple(saved["p2"])]
        print(f"  Loaded existing: {points[0]} -> {points[1]}")

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate", min(w*2, 1200), min(h*2, 900))
    cv2.setMouseCallback("Calibrate", mouse_callback)

    while True:
        display = frame.copy()

        if len(points) == 2:
            p1, p2 = points[0], points[1]
            # Shade danger zone below the line
            danger_poly = np.array([
                p1, p2,
                (p2[0], h), (p1[0], h)
            ], dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [danger_poly], (0, 0, 180))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            # Draw the line
            cv2.line(display, p1, p2, (0, 100, 255), 3)
            cv2.circle(display, p1, 8, (0, 255, 0), -1)
            cv2.circle(display, p2, 8, (0, 255, 0), -1)
            cv2.putText(display, f"{args.view.upper()} STOP LINE",
                        (p1[0], p1[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        elif len(points) == 1:
            cv2.circle(display, points[0], 8, (0, 255, 0), -1)
            cv2.putText(display, "Now click the RIGHT end of the line",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            cv2.putText(display, "Click LEFT end of the stop line",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(display, "R=reset | S=save | Q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow("Calibrate", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            points = []
            print("  Reset — click 2 points again")
        elif key in (ord("s"), 13):
            if len(points) == 2:
                config[args.view] = {
                    "p1": list(points[0]),
                    "p2": list(points[1]),
                    "y_fraction": (points[0][1] + points[1][1]) / 2 / h
                }
                with open(CONFIG_FILE, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"Saved! {args.view}: {points[0]} -> {points[1]}")
                break
            else:
                print("  Need 2 points first!")
        elif key == ord("q"):
            print("Quit without saving."); break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
