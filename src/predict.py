"""Detect and recognise traffic signs in a road photo.

Stage 1: YOLOv8s (trained on GTSDB) finds the signs.
Stage 2: the CNN (trained on GTSRB) classifies each detected sign into 43 classes.

Usage:
    python src/predict.py --image road.jpg
    python src/predict.py --image road.jpg --output road_pred.jpg --conf 0.5
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing veh over 3.5 tons", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End speed + passing limits", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing", "End no passing veh > 3.5 tons",
]
IMG_SIZE = 32  # input size of the CNN


def detect_and_classify(img, detector, classifier, conf=0.5, imgsz=1280):
    """Return a list of (box, detection score, class id, class probability)."""
    result = detector.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    if len(boxes) == 0:
        return []
    crops = np.stack([
        np.asarray(img.crop(tuple(int(round(v)) for v in b)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        for b in boxes
    ])
    probs = classifier.predict(crops, verbose=0)
    return [(b, s, int(p.argmax()), float(p.max())) for b, s, p in zip(boxes, scores, probs)]


def draw(img, detections):
    out = img.copy()
    pen = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for (x1, y1, x2, y2), score, cls, _ in detections:
        label = f"{CLASS_NAMES[cls]} ({score:.2f})"
        pen.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        tx, ty = x1, max(0, y1 - 14)
        pen.rectangle(pen.textbbox((tx, ty), label, font=font), fill=(0, 255, 0))
        pen.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return out


def main():
    parser = argparse.ArgumentParser(description="Traffic sign detection (YOLOv8) and recognition (CNN).")
    parser.add_argument("--image", required=True, help="path to a road photo")
    parser.add_argument("--output", help="annotated image path (default: <image>_pred.jpg)")
    parser.add_argument("--detector", default="models/gtsdb_yolov8s_detector.pt")
    parser.add_argument("--classifier", default="models/gtsrb_cnn_classifier.keras")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    args = parser.parse_args()

    import tensorflow as tf
    from ultralytics import YOLO

    detector = YOLO(args.detector)
    classifier = tf.keras.models.load_model(args.classifier)

    img = Image.open(args.image).convert("RGB")
    detections = detect_and_classify(img, detector, classifier, conf=args.conf)

    print(f"{len(detections)} sign(s) detected")
    for (x1, y1, x2, y2), score, cls, prob in detections:
        print(f"  {CLASS_NAMES[cls]:<32} detection {score:.2f} | class {prob:.2f} "
              f"| box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    image_path = Path(args.image)
    output = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_pred.jpg")
    draw(img, detections).save(output, quality=95)
    print(f"Annotated image saved to {output}")


if __name__ == "__main__":
    main()
