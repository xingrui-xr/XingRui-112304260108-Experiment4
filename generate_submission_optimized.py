import csv
import torch
from pathlib import Path
from ultralytics import YOLO

def generate_submission_optimized():
    """
    优化后的提交文件生成脚本
    优化策略：
    1. 测试时数据增强（Test Time Augmentation - TTA）
    2. 多尺度推理（Multi-scale inference）
    3. 调整NMS参数
    4. 更合理的置信度阈值
    5. 批量推理加速
    """
    
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
    output_path = 'submission_optimized.csv'
    
    print(f"Generating optimized submission to {output_path}...")
    
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        # 配置推理参数
        inference_configs = [
            # 配置1：标准尺度
            {'imgsz': 416, 'conf': 0.001, 'iou': 0.5},
            # 配置2：更大尺度（捕捉小目标）
            {'imgsz': 640, 'conf': 0.001, 'iou': 0.5},
        ]
        
        # 对每张图片进行多尺度推理
        for img_path in image_paths:
            image_id = img_path.name
            
            all_detections = []
            
            # 多尺度推理
            for config in inference_configs:
                results = model.predict(
                    source=str(img_path),
                    conf=config['conf'],
                    iou=config['iou'],
                    imgsz=config['imgsz'],
                    save=False,
                    verbose=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    max_det=500,
                    augment=True,  # 开启测试时数据增强
                    agnostic_nms=False,
                )
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x_center, y_center, width, height = box.xywhn[0].tolist()
                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())
                            
                            # 过滤掉面积过大或过小的检测框
                            area = width * height
                            if area < 0.0001 or area > 0.95:
                                continue
                            
                            all_detections.append({
                                'class_id': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height,
                                'confidence': confidence
                            })
            
            # 对所有检测结果进行NMS合并
            final_detections = merge_nms(all_detections, iou_threshold=0.6, conf_threshold=0.01)
            
            # 写入结果
            for det in final_detections:
                writer.writerow({
                    "image_id": image_id,
                    "class_id": det['class_id'],
                    "x_center": det['x_center'],
                    "y_center": det['y_center'],
                    "width": det['width'],
                    "height": det['height'],
                    "confidence": det['confidence']
                })
    
    print(f"Optimized submission file generated successfully!")

def merge_nms(detections, iou_threshold=0.6, conf_threshold=0.01):
    """
    对所有检测结果进行NMS合并
    """
    if len(detections) == 0:
        return []
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # 按类别分组
    detections_by_class = {}
    for det in detections:
        cls = det['class_id']
        if cls not in detections_by_class:
            detections_by_class[cls] = []
        detections_by_class[cls].append(det)
    
    final_detections = []
    
    # 对每个类别进行NMS
    for cls, dets in detections_by_class.items():
        kept = []
        while len(dets) > 0:
            # 取置信度最高的
            best = dets.pop(0)
            
            # 过滤低置信度
            if best['confidence'] < conf_threshold:
                continue
            
            kept.append(best)
            
            # 计算与其他检测框的IoU
            iou_scores = [calculate_iou(best, det) for det in dets]
            
            # 保留IoU小于阈值的检测框
            dets = [det for det, iou in zip(dets, iou_scores) if iou < iou_threshold]
        
        final_detections.extend(kept)
    
    # 再次按置信度排序
    final_detections = sorted(final_detections, key=lambda x: x['confidence'], reverse=True)
    
    return final_detections

def calculate_iou(box1, box2):
    """
    计算两个检测框的IoU
    输入是YOLO格式的归一化坐标：(x_center, y_center, width, height)
    """
    # 转换为左上角和右下角坐标
    x1_1 = box1['x_center'] - box1['width'] / 2
    y1_1 = box1['y_center'] - box1['height'] / 2
    x2_1 = box1['x_center'] + box1['width'] / 2
    y2_1 = box1['y_center'] + box1['height'] / 2
    
    x1_2 = box2['x_center'] - box2['width'] / 2
    y1_2 = box2['y_center'] - box2['height'] / 2
    x2_2 = box2['x_center'] + box2['width'] / 2
    y2_2 = box2['y_center'] + box2['height'] / 2
    
    # 计算交集
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

if __name__ == "__main__":
    generate_submission_optimized()
