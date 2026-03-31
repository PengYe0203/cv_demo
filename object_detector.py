"""
YOLOv8物体识别模块
"""
from typing import List, Optional
import numpy as np
import cv2
from ultralytics import YOLO

from interfaces import Detector, DetectionResult, FrameData


class YOLOv8Detector(Detector):
    """YOLOv8物体检测器"""
    
    # YOLOv8 COCO数据集中的类别ID
    # 识别瓶子和容器
    ALLOWED_CLASS_IDS = {39, 46}  # 39=bottle(瓶子), 46=cup(杯/罐)
    
    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.5,
                 device: str = '0'):  # '0'表示GPU，'cpu'表示CPU
        """
        初始化YOLOv8检测器
        
        Args:
            model_name: 模型名称 ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            confidence_threshold: 置信度阈值
            device: 运算设备 ('0' for GPU, 'cpu')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model: Optional[YOLO] = None
        self.class_names = {}
        
    def initialize(self) -> bool:
        """初始化检测器，加载YOLOv8模型"""
        try:
            print(f"[YOLOv8] Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            print(f"[YOLOv8] Model initialized successfully")
            return True
        except Exception as e:
            print(f"[YOLOv8] Error initializing model: {e}")
            return False
    
    def detect(self, frame: FrameData) -> List[DetectionResult]:
        """
        对单帧进行物体检测
        
        Args:
            frame: FrameData对象
            
        Return: DetectionResult列表
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        try:
            # 运行检测
            results = self.model(frame.image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            # 处理检测结果
            if results and len(results) > 0:
                result = results[0]
                
                # 提取边界框和类别信息
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        # 获取类别
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 只保留允许的类别
                        if class_id not in self.ALLOWED_CLASS_IDS:
                            continue
                        
                        # 获取边界框坐标
                        bbox_coords = box.xyxy[0].cpu().numpy()
                        x_min, y_min, x_max, y_max = bbox_coords.astype(int)
                        
                        # 获取置信度
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        class_name = self.class_names.get(class_id, f"Class_{class_id}")
                        
                        # 创建检测结果
                        detection = DetectionResult(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(int(x_min), int(y_min), int(x_max), int(y_max))
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[YOLOv8] Error during detection: {e}")
            return []
    
    def release(self) -> None:
        """释放资源"""
        if self.model is not None:
            # YOLO模型不需要特殊的释放操作，但这里保留接口
            self.model = None
            print("[YOLOv8] Resources released")
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        }
