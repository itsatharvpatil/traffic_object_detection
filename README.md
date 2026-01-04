# Real-Time Traffic Object Detection with YOLOv8

<img width="1278" height="743" alt="webcamtest" src="https://github.com/user-attachments/assets/8f11e3ed-0e00-48a6-b693-1e4687535d60" />
*Live webcam demo showing real-time detection of person (0.86), cell phone (0.70), bottles, and chair at ~15 FPS. The system is fully capable of detecting traffic entities (person, car, bus, bicycle, motorbike) when pointed at relevant scenes.*

A lightweight, real-time object detection system built with **Ultralytics YOLOv8n**. The project demonstrates live inference on webcam feed and batch processing on the Pascal VOC benchmark dataset, with a focus on traffic-related applications.

The screenshot shows the model working in a real-world indoor environment with accurate bounding boxes and confidence scores. While it's not a traffic scene, it effectively proves:
- The system runs smoothly in real-time
- Detections are precise and confident
- The pipeline (webcam capture → inference → visualization) works end-to-end

For traffic-specific examples, the model excels at detecting **person, car, bus, bicycle, motorbike** when aimed at roads, vehicles, or outdoor scenes — exactly as intended for traffic monitoring use cases.

## Performance (Fine-Tuned Model on Pascal VOC)
- **mAP@0.5**: **0.836** (83.6% – excellent precision at standard IoU threshold)
- **mAP@0.5:0.95**: **0.631** (strong overall accuracy across IoU levels)
- **Classes with highest performance**:
  - Bicycle: 0.92
  - Car: 0.931
  - Bus: 0.902
  - Person: 0.903
  - Motorbike: 0.894
- Inference speed: **1.6 ms** per image on Tesla T4 GPU (~600 FPS theoretical, 20–50 FPS real-time on laptop)
- Training time: ~2.5 hours for 30 epochs on Colab T4 GPU

## Key Features
- Real-time detection at **20–50 FPS** on consumer laptop (GTX 1650 Ti)
- Live webcam feed with **bounding boxes, class labels, confidence scores, and FPS overlay**
- Batch inference on **Pascal VOC dataset** (21k+ images) with annotated outputs
- Robust error handling and stable CPU fallback mode
- Easily extensible to traffic-focused (person, car, bus, bicycle, motorbike, truck)

## Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- NumPy

## Requirements
```txt
ultralytics>=8.3.0
opencv-python
numpy
```

Install with:
```bash
pip install ultralytics opencv-python numpy
```

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
