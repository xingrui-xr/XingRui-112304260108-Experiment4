import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch

def train_model():
    """训练YOLOv8模型【GPU强力加速版】"""
    # 强制使用GPU，没有GPU会直接报错提醒
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到NVIDIA GPU，请检查CUDA环境！")
    
    # 使用更大的模型以获得更好的性能
    model = YOLO("yolov8m.pt") # 使用x版本模型
    
    # 训练参数配置（GPU优化版）
    results = model.train(
        data='data.yaml',
        epochs=50,           
        imgsz=416,            
        batch=8,             # 显存足够保持16，不足可改为8
        patience=50,          
        device=1,             # 强制使用第一张GPU（核心修改）
        workers=0,            # GPU训练建议调高
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
    )
    
    return model

def generate_submission(model_path, test_dir, output_path):
    """生成提交文件【GPU加速版】"""
    # 强制GPU推理
    model = YOLO(model_path)
    
    test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    
    submission_data = []
    
    for img_path in test_images:
        image_id = img_path.name
        
        # GPU推理（核心修改）
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.001,  
            iou=0.6,
            device=0,        # 强制使用GPU推理
            max_det=300,
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取YOLO格式的归一化坐标
                    x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    submission_data.append({
                        'image_id': image_id,
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(submission_data)
    df = df[['image_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence']]
    df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

if __name__ == "__main__":
    # 训练模型
    print("Starting model training...【GPU版本】")
    model = train_model()
    
    # 生成提交文件
    print("Generating submission...【GPU加速】")
    best_model_path = 'runs/detect/train/weights/best.pt'
    generate_submission(best_model_path, 'test/images', 'submission.csv')
    
    print("All done!")