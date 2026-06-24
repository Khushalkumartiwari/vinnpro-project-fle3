"""
batch_evaluate.py
-------------------
Walks labeled subfolders and scores detection accuracy automatically.

IMPORTANT: view (front/back) is determined PER FILE from the filename
(must contain "front" or "back"), not from the folder name - a folder like
VehicleFrontOutSideWB legitimately contains both _Back.jpg and _Front.jpg
files (two camera angles of the same event), and each needs its own
matching detector.

Folder name only determines the EXPECTED ground-truth label:
  - contains "inside" (and not "outside") -> expect OK
  - everything else (person/object/outside) -> expect ALARM

Usage:
    python3 batch_evaluate.py --root "/Users/kushaltiwari/Desktop/Incorrect Weighment Images"
"""

import argparse
import csv
import os
import cv2

from boundary_detector import BoundaryDetector
from vehicle_detector import VehicleDetector
from dharam_katta_monitor import evaluate_frame, annotate

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".webp"}


def infer_view_from_filename(filename):
    name = filename.lower()
    if "front" in name:
        return "front"
    if "back" in name:
        return "back"
    return None  # ambiguous - caller should skip or warn


def infer_expected_alarm(folder_name):
    name = folder_name.lower()
    if "inside" in name and "outside" not in name:
        return False
    return True


def find_image_subfolders(root):
    out = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and not entry.startswith("."):
            out.append((full, entry))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--vehicle-model", default="yolov8n.pt")
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--out-dir", default="batch_annotated")
    parser.add_argument("--csv", default="batch_results.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vehicle_det = VehicleDetector(model_path=args.vehicle_model)
    boundary_dets = {"front": BoundaryDetector(view="front"), "back": BoundaryDetector(view="back")}

    subfolders = find_image_subfolders(args.root)
    if not subfolders:
        print(f"No subfolders found in {args.root}")
        return

    rows = []
    correct = 0
    total = 0

    for folder_path, folder_name in subfolders:
        expected_alarm = infer_expected_alarm(folder_name)
        out_subdir = os.path.join(args.out_dir, folder_name)
        os.makedirs(out_subdir, exist_ok=True)

        files = sorted(
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not files:
            print(f"[{folder_name}] no images found, skipping")
            continue

        print(f"\n[{folder_name}] expected_alarm={expected_alarm} ({len(files)} images)")

        for fname in files:
            view = infer_view_from_filename(fname)
            if view is None:
                print(f"  [SKIP] {fname}: filename doesn't indicate front/back")
                continue

            path = os.path.join(folder_path, fname)
            frame = cv2.imread(path)
            if frame is None:
                print(f"  [SKIP] could not read {fname}")
                continue

            polygon, _ = boundary_dets[view].detect(frame)
            vehicles, foreign = vehicle_det.detect(frame)
            alarm, reasons = evaluate_frame(polygon, vehicles, foreign, args.margin)

            annotated = annotate(frame, polygon, vehicles, foreign, alarm)
            cv2.imwrite(os.path.join(out_subdir, fname), annotated)

            is_correct = (alarm == expected_alarm)
            correct += int(is_correct)
            total += 1

            rows.append({
                "folder": folder_name,
                "filename": fname,
                "view": view,
                "expected": "ALARM" if expected_alarm else "OK",
                "predicted": "ALARM" if alarm else "OK",
                "match": "YES" if is_correct else "NO",
                "reasons": " | ".join(reasons),
            })
            print(f"  {fname} (view={view}): predicted={'ALARM' if alarm else 'OK'} expected={'ALARM' if expected_alarm else 'OK'} {'OK' if is_correct else '<-- MISMATCH'}")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "filename", "view", "expected", "predicted", "match", "reasons"])
        writer.writeheader()
        writer.writerows(rows)

    accuracy = (correct / total * 100) if total else 0
    print(f"\n=== DONE: {correct}/{total} correct ({accuracy:.1f}%) ===")
    print(f"CSV -> {args.csv}")
    print(f"Annotated images -> {args.out_dir}/")


if __name__ == "__main__":
    main()
