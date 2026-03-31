"""
视觉识别模块的接口定义
为物体识别和手部识别提供统一的接口规范
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class DetectionResult:
    """检测结果数据类"""
    class_name: str  # 类名（物体或手）
    confidence: float  # 置信度 (0-1)
    bbox: Tuple[int, int, int, int]  # 边界框 (x_min, y_min, x_max, y_max)
    landmarks: Optional[np.ndarray] = None  # 关键点坐标，用于手部检测
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'landmarks': self.landmarks.tolist() if self.landmarks is not None else None
        }


@dataclass
class FrameData:
    """帧数据结构"""
    image: np.ndarray  # 图像数据 (HxWx3, BGR格式)
    frame_id: int  # 帧ID
    timestamp: float  # 时间戳
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """获取图像形状"""
        return self.image.shape


class Detector(ABC):
    """检测器基类 - 定义检测器的统一接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化检测器（加载模型等）
        Return: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def detect(self, frame: FrameData) -> List[DetectionResult]:
        """
        对单帧进行检测
        Args:
            frame: FrameData对象
        Return: 检测结果列表
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """释放资源"""
        pass
    
    def __enter__(self):
        """上下文管理器支持"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.release()


class VisionProcessor(ABC):
    """视觉处理管道基类"""
    
    @abstractmethod
    def process_frame(self, frame: FrameData) -> dict:
        """
        处理单帧，整合多个检测器的结果
        Args:
            frame: FrameData对象
        Return: 包含所有检测结果的字典
        """
        pass
    
    @abstractmethod
    def make_decision(self, detections: dict) -> Optional[str]:
        """
        根据检测结果做出决策
        Args:
            detections: 检测结果字典
        Return: 决策文本（如路径建议），暂时保留
        """
        pass
