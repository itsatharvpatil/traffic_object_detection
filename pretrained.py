from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
import numpy as np

# Load model
model = YOLO("yolov8n.pt")

# Paths
dataset_root = r"D:\datasets\VOC\images"
output_root = r"C:\Users\Atharv\Downloads\traffic_detection\clean_detections"  # New folder for clean results

os.makedirs(output_root, exist_ok=True)

# Traffic classes only (from COCO IDs – YOLO uses these)
traffic_classes = [0, 1, 2, 3, 5, 7]  # person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7

# Splits
splits = ["test2007"]  # Start with test set – change to all if happy: ["train2012", "val2012", "train2007", "val2007", "test2007"]

total_processed = 0
total_detected = 0

print("Starting CPU detection (no CUDA errors)...")

for split in splits:
    split_path = os.path.join(dataset_root, split)
    if not os.path.exists(split_path):
        print(f"Skipping {split} - not found")
        continue
    
    output_split = os.path.join(output_root, split)
    os.makedirs(output_split, exist_ok=True)
    
    images = [f for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {split}: {len(images)} images")
    
    for img_name in tqdm(images, desc=f"{split}"):
        img_path = os.path.join(split_path, img_name)
        output_path = os.path.join(output_split, img_name)
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # CPU inference (safe, no CUDA crashes)
        results = model(frame, conf=0.25, iou=0.45, verbose=False, device="cpu", classes=traffic_classes)
        
        # Get result
        result = results[0]
        
        # If detections, draw boxes (clean, no text on errors)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            if np.any(np.isnan(boxes)):
                # Rare NaN – save original clean
                cv2.imwrite(output_path, frame)
            else:
                annotated = result.plot()
                cv2.imwrite(output_path, annotated)
                total_detected += 1
        else:
            # No detections – save clean original (no text)
            cv2.imwrite(output_path, frame)
        
        total_processed += 1

print(f"\nDONE! Processed {total_processed} images")
print(f"Images with traffic detections: {total_detected} (clean boxes)")
print(f"Other images saved clean (no boxes or text)")
print(f"Results in: {output_root}")
