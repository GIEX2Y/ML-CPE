# LAB-10 Object Detection Using YOLO

## Objective

This project applies YOLO (You Only Look Once) for object detection and classification. The model is trained to detect and classify objects in images by identifying their locations using bounding boxes.

The objectives are:

- Apply YOLO for object detection using a selected dataset.
- Train and evaluate YOLO models with different training configurations.
- Measure model performance using Precision, Recall, and mAP metrics.
- Test object detection results with bounding boxes, class labels, and confidence scores.

---

## Dataset

Dataset: Cat and Dog Object Detection Dataset

Classes:

- 0: Cat
- 1: Dog

Dataset preparation:

- Total images: 500 images
- Training set: 400 images
- Validation set: 50 images
- Testing set: 50 images

Dataset format:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

YOLO annotation format:

```
class_id x_center y_center width height
```

---

## Environment

Hardware:

- MacBook Air M1

Software:

- Python 3.12
- PyTorch
- Ultralytics YOLOv8

Install dependencies:

```bash
pip3 install -r requirements.txt
```

---

## Model Configuration

Model:

```
YOLOv8n
```

Training configuration:

| Parameter | Value |
|---|---|
| Image Size | 416 x 416 |
| Batch Size | 8 |
| Epochs | 50 |
| Model | YOLOv8 Nano |

---

## Training

Run training:

```bash
python3 train.py
```

Training output:

```
runs/detect/YOLO_50_epoch/
```

Generated files:

```
weights/
├── best.pt
└── last.pt
```

---

## Evaluation

Run evaluation:

```bash
python3 evaluate.py
```

Evaluation metrics:

- Precision
- Recall
- mAP50
- mAP50-95

---

## Evaluation Results

Model:

```
YOLO_50_epoch
```

Results:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.937 | 0.990 | 0.995 | 0.995 |
| Cat | 1.000 | 0.980 | 0.995 | 0.995 |
| Dog | 0.874 | 1.000 | 0.995 | 0.995 |

---

## Training Performance

Training completed:

```
50 epochs completed in 0.716 hours
```

The model achieved high detection performance with:

- Precision: 93.7%
- Recall: 99.0%
- mAP50: 99.5%
- mAP50-95: 99.5%

---

## Object Detection Prediction

Run prediction:

```bash
python3 predict.py
```

The model produces:

- Bounding boxes
- Class labels
- Confidence scores

Prediction results are saved in:

```
runs/detect/predict/
```

Example output:

```
Cat 0.95
Dog 0.91
```

---

## Project Structure

```
LAB-10/

├── dataset/
├── data.yaml
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
└── runs/
```

---

## Conclusion

The YOLOv8 object detection model was successfully trained to detect cats and dogs. The model achieved high performance with a mAP50 score of 99.5%, demonstrating that YOLO can effectively detect and classify objects with accurate bounding boxes and confidence scores.