"""
视觉模块包
包含物体检测、手部检测、深度估计等视觉相关功能
"""
from .interfaces import DetectionResult, FrameData, Detector, VisionProcessor
from .object_detector import YOLOv8Detector
from .hand_detector import MediaPipeHandsDetector
from .depth_estimator import DepthEstimator
from .vision_processor import ObjectGraspingVisionProcessor

__all__ = [
    'DetectionResult',
    'FrameData',
    'Detector',
    'VisionProcessor',
    'YOLOv8Detector',
    'MediaPipeHandsDetector',
    'DepthEstimator',
    'ObjectGraspingVisionProcessor',
]
