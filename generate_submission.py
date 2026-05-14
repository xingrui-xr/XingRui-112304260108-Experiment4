import csv
from pathlib import Path
from ultralytics import YOLO

def generate_submission():
    # 训练好的模型路径
    model_path = 'runs/detect/runs/detect/train/weights/best.pt'
    
    # 加载模型
    model = YOLO(model_path)
    
    # 测试图片目录
    test_dir = Path('test/images')
    
    # 获取所有测试图片（支持 jpg 和 png）
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.png')])
    
    print(f"Found {len(image_paths)} test images")
    print(f"First few images: {[p.name for p in image_paths[:5]]}")
    
    # 输出文件路径
    output_path = 'submission.csv'
    
    print(f"Generating submission to {output_path}...")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        # 遍历每张图片进行推理
        for img_path in image_paths:
            # 获取真实的图片文件名
            image_id = img_path.name
            
            # 进行推理
            results = model.predict(
                source=str(img_path),
                conf=0.001,
                iou=0.6,
                imgsz=416,
                save=False,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                max_det=300,
            )
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        
                        # 只保留置信度大于阈值的检测（过滤低置信度噪声）
                        if confidence >= 0.05:
                            writer.writerow({
                                "image_id": image_id,
                                "class_id": class_id,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": width,
                                "height": height,
                                "confidence": confidence,
                            })
    
    print(f"Submission file generated successfully!")

if __name__ == "__main__":
    import torch
    generate_submission()
