"""
深度估计模块 - 使用DPT模型
"""
from typing import Optional, Tuple
import numpy as np
import cv2
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

class DepthEstimator:
    """DPT深度估计器"""
    
    def __init__(self, model_name: str = "intel/dpt-large"):
        """
        初始化深度估计器
        
        Args:
            model_name: 模型名称 ('intel/dpt-large', 'intel/dpt-hybrid-midas', 等)
        """
        self.model_name = model_name
        self.processor: Optional[DPTImageProcessor] = None
        self.model: Optional[DPTForDepthEstimation] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize(self) -> bool:
        """初始化深度估计模型"""
        try:
            print(f"[DepthEstimator] Loading {self.model_name}...")
            self.processor = DPTImageProcessor.from_pretrained(self.model_name)
            self.model = DPTForDepthEstimation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"[DepthEstimator] Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"[DepthEstimator] Error loading model: {e}")
            return False
    
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        估计图像的深度图
        
        Args:
            image: BGR图像数组
            
        Return: 深度图 (HxW)，值在0-1之间（相对深度）
        """
        try:
            if self.model is None or self.processor is None:
                print("[DepthEstimator] Model not initialized")
                return None
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 预处理
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
                predicted_depth = outputs.predicted_depth
            
            # 转换为numpy并缩放到原始分辨率
            depth_map = predicted_depth.squeeze().cpu().numpy()
            h, w = image.shape[:2]
            depth_map = cv2.resize(depth_map, (w, h))
            
            # 归一化到0-1
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)
            
            return depth_map
            
        except Exception as e:
            print(f"[DepthEstimator] Error estimating depth: {e}")
            return None
    
    def get_depth_at_bbox(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        获取边界框内的平均深度
        
        Args:
            depth_map: 深度图
            bbox: (x_min, y_min, x_max, y_max)
            
        Return: 平均深度值 (0-1，值越大表示离摄像头越近)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 确保坐标在范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(depth_map.shape[1], x_max)
        y_max = min(depth_map.shape[0], y_max)
        
        roi = depth_map[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return 0.5
        
        return float(roi.mean())
    
    def is_hand_in_front(self, depth_map: np.ndarray, 
                        hand_bbox: Tuple[int, int, int, int],
                        object_bbox: Tuple[int, int, int, int],
                        threshold: float = 0.05) -> bool:
        """
        判断手是否在物体前面
        
        Args:
            depth_map: 深度图
            hand_bbox: 手的边界框
            object_bbox: 物体的边界框
            threshold: 深度差异阈值
            
        Return: True表示手在前面，False表示手在后面或不确定
        """
        hand_depth = self.get_depth_at_bbox(depth_map, hand_bbox)
        object_depth = self.get_depth_at_bbox(depth_map, object_bbox)
        
        depth_diff = hand_depth - object_depth
        
        # 如果手的深度值明显大于物体，说明手更靠近摄像头
        return depth_diff > threshold
    
    def release(self):
        """释放资源"""
        self.model = None
        self.processor = None
        print("[DepthEstimator] Resources released")
