"""
视觉处理管道 - 整合物体识别、手部识别和深度估计
"""
from typing import Optional, List, Dict
import numpy as np
import cv2
from .interfaces import VisionProcessor, FrameData, DetectionResult, Detector
from .object_detector import YOLOv8Detector
from .hand_detector import MediaPipeHandsDetector

try:
    from .depth_estimator import DepthEstimator
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False
    print("[VisionProcessor] Warning: DepthEstimator not available (transformers not installed)")


class ObjectGraspingVisionProcessor(VisionProcessor):
    """辅助抓取物体的视觉处理管道"""
    
    def __init__(self, 
                 enable_object_detection: bool = True,
                 enable_hand_detection: bool = True,
                 enable_depth_estimation: bool = False,
                 yolo_model: str = 'vision_modules/models/yolov8n.pt',
                 yolo_confidence: float = 0.5,
                 device: str = '0'):
        """
        初始化视觉处理管道
        
        Args:
            enable_object_detection: 是否启用物体检测
            enable_hand_detection: 是否启用手部检测
            enable_depth_estimation: 是否启用深度估计
            yolo_model: YOLOv8模型名称
            yolo_confidence: YOLOv8置信度阈值
            device: 运算设备
        """
        self.enable_object_detection = enable_object_detection
        self.enable_hand_detection = enable_hand_detection
        self.enable_depth_estimation = enable_depth_estimation and DEPTH_AVAILABLE
        
        # 初始化检测器
        self.object_detector: Optional[YOLOv8Detector] = None
        self.hand_detector: Optional[MediaPipeHandsDetector] = None
        self.depth_estimator: Optional['DepthEstimator'] = None
        
        if enable_object_detection:
            self.object_detector = YOLOv8Detector(
                model_name=yolo_model,
                confidence_threshold=yolo_confidence,
                device=device
            )
        
        if enable_hand_detection:
            self.hand_detector = MediaPipeHandsDetector(
                static_image_mode=False,  # 处理视频流
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        
        if self.enable_depth_estimation:
            self.depth_estimator = DepthEstimator()
    
    def initialize(self) -> bool:
        """初始化所有检测器"""
        success = True
        
        if self.object_detector:
            print("[VisionProcessor] Initializing object detector...")
            if not self.object_detector.initialize():
                print("[VisionProcessor] Warning: Object detector initialization failed")
                success = False
            else:
                print("[VisionProcessor] Object detector initialized")
        
        if self.hand_detector:
            print("[VisionProcessor] Initializing hand detector...")
            if not self.hand_detector.initialize():
                print("[VisionProcessor] Warning: Hand detector initialization failed")
                success = False
            else:
                print("[VisionProcessor] Hand detector initialized")
        
        if self.enable_depth_estimation and self.depth_estimator:
            print("[VisionProcessor] Initializing depth estimator...")
            if not self.depth_estimator.initialize():
                print("[VisionProcessor] Warning: Depth estimator initialization failed")
                self.enable_depth_estimation = False
            else:
                print("[VisionProcessor] Depth estimator initialized")
        
        if success:
            print("[VisionProcessor] All detectors initialized successfully")
        return success
    
    def process_frame(self, frame: FrameData) -> dict:
        """
        处理单帧，获取物体和手部检测结果及深度信息
        
        Args:
            frame: FrameData对象
            
        Return: 包含检测结果的字典
        """
        results = {
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp,
            'objects': [],
            'hands': [],
            'raw_detections': [],
            'depth_map': None,
            'spatial_relations': []
        }
        
        try:
            # 物体检测
            if self.object_detector:
                objects = self.object_detector.detect(frame)
                results['objects'] = objects
                results['raw_detections'].extend([
                    {'type': 'object', 'detection': obj} for obj in objects
                ])
            
            # 手部检测
            if self.hand_detector:
                hands = self.hand_detector.detect(frame)
                results['hands'] = hands
                results['raw_detections'].extend([
                    {'type': 'hand', 'detection': hand} for hand in hands
                ])
            
            # 深度估计及空间关系分析
            if self.enable_depth_estimation and self.depth_estimator:
                depth_map = self.depth_estimator.estimate_depth(frame.image)
                results['depth_map'] = depth_map
                
                # 分析手与物体的前后关系
                if depth_map is not None and results['objects'] and results['hands']:
                    for hand in results['hands']:
                        for obj in results['objects']:
                            hand_in_front = self.depth_estimator.is_hand_in_front(
                                depth_map,
                                hand.bbox,
                                obj.bbox
                            )
                            
                            relation = {
                                'hand': hand.class_name,
                                'object': obj.class_name,
                                'hand_in_front': hand_in_front,
                                'relation_text': f"{hand.class_name} 在 {obj.class_name} {'前面' if hand_in_front else '后面'}"
                            }
                            results['spatial_relations'].append(relation)
            
            return results
            
        except Exception as e:
            print(f"[VisionProcessor] Error processing frame: {e}")
            return results
    
    def make_decision(self, detections: dict) -> Optional[str]:
        """
        根据检测结果做出决策
        当前暂时保留此接口，返回检测信息的文本描述
        后续可扩展为语音指导等功能
        
        Args:
            detections: 检测结果字典
            
        Return: 决策文本描述
        """
        try:
            decision_text = []
            
            # 分析物体位置
            if detections.get('objects'):
                objects = detections['objects']
                closest_object = min(objects, key=lambda x: x.bbox[2] - x.bbox[0])
                
                obj_x_min, obj_y_min, obj_x_max, obj_y_max = closest_object.bbox
                obj_center_x = (obj_x_min + obj_x_max) // 2
                obj_center_y = (obj_y_min + obj_y_max) // 2
                
                decision_text.append(
                    f"物体: {closest_object.class_name} (置信度: {closest_object.confidence:.2f}) "
                    f"位置: ({obj_center_x}, {obj_center_y})"
                )
            
            # 分析手部位置
            if detections.get('hands'):
                hands = detections['hands']
                for i, hand in enumerate(hands):
                    hand_x_min, hand_y_min, hand_x_max, hand_y_max = hand.bbox
                    hand_center_x = (hand_x_min + hand_x_max) // 2
                    hand_center_y = (hand_y_min + hand_y_max) // 2
                    
                    decision_text.append(
                        f"手部{i+1}: {hand.class_name} "
                        f"位置: ({hand_center_x}, {hand_center_y})"
                    )
            
            return " | ".join(decision_text) if decision_text else "未检测到物体或手部"
            
        except Exception as e:
            print(f"[VisionProcessor] Error making decision: {e}")
            return None
    
    def release(self) -> None:
        """释放所有资源"""
        if self.object_detector:
            self.object_detector.release()
        
        if self.hand_detector:
            self.hand_detector.release()
        
        if self.depth_estimator:
            self.depth_estimator.release()
        
        print("[VisionProcessor] All resources released")
    
    def get_system_info(self) -> dict:
        """获取系统配置信息"""
        info = {
            'object_detection_enabled': self.enable_object_detection,
            'hand_detection_enabled': self.enable_hand_detection,
        }
        
        if self.object_detector:
            info['object_detector'] = self.object_detector.get_model_info()
        
        return info
    
    def __enter__(self):
        """上下文管理器支持"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.release()
