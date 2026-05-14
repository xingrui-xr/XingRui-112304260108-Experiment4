# Traffic Sign Detection Challenge

## Task
Train an object detection model with the provided YOLO dataset and predict objects on the hidden-label test set.

## Classes
Green Light, Red Light, Speed Limit 10, Speed Limit 100, Speed Limit 110, Speed Limit 120, Speed Limit 20, Speed Limit 30, Speed Limit 40, Speed Limit 50, Speed Limit 60, Speed Limit 70, Speed Limit 80, Speed Limit 90, Stop

## Directory
- `train/`: training images and labels
- `val/`: validation images and labels
- `test/images/`: test images only
- `data.yaml`: Ultralytics training config
- `sample_submission.csv`: submission schema
- `baseline_infer.py`: example inference-to-CSV script

## Submission
Submit one `submission.csv` file with these columns:
- `image_id`
- `class_id`
- `x_center`
- `y_center`
- `width`
- `height`
- `confidence`

All coordinates must be YOLO-style normalized values in `[0, 1]`.

## Metric
Ranking metric: `mAP@0.5`

## Example training
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416
```

## Example submission generation
```bash
python baseline_infer.py --model runs/detect/train/weights/best.pt --test-dir test/images --output submission.csv
```
