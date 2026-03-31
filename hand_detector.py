"""
MediaPipe Hands手部识别模块 - 使用新版API (v0.10.33+)
"""
from typing import List, Optional
import numpy as np
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe import Image as mp_image
try:
    from mediapipe.framework.formats.image import ImageFormat
except ImportError:
    # 尝试其他位置
    try:
        from mediapipe.tasks.python.vision.image_format import ImageFormat
    except ImportError:
        # 如果都找不到，定义替代品
        class ImageFormat:
            SRGB = 1

from interfaces import Detector, DetectionResult, FrameData


class MediaPipeHandsDetector(Detector):
    """MediaPipe Hands检测器 - 使用HandLandmarker"""
    
    # 手部关键点名称
    HAND_LANDMARKS = [
        'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
        'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
        'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
    ]
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        初始化MediaPipe Hands检测器
        
        Args:
            static_image_mode: 是否为静态图像模式
            max_num_hands: 最多检测的手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.hand_landmarker = None
        
    def initialize(self) -> bool:
        """初始化MediaPipe Hands检测器"""
        try:
            print("[MediaPipe Hands] Initializing...")
            import os
            import urllib.request
            
            # 本地模型路径
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 如果本地模型不存在，下载
            if not os.path.exists(model_path):
                print("[MediaPipe Hands] Model not found. Downloading...")
                # 使用正确的官方模型URL
                model_url = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print(f"[MediaPipe Hands] Model downloaded successfully to {model_path}")
                    print(f"[MediaPipe Hands] Model size: {os.path.getsize(model_path)} bytes")
                except Exception as download_error:
                    print(f"[MediaPipe Hands] Download failed: {download_error}")
                    return False
            else:
                print(f"[MediaPipe Hands] Using existing model: {model_path}")
                print(f"[MediaPipe Hands] Model size: {os.path.getsize(model_path)} bytes")
            
            # 创建HandLandmarker选项（必须提供真实的model_asset_path）
            print(f"[MediaPipe Hands] Creating HandLandmarkerOptions...")
            base_opt = base_options.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_opt,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE
            )
            
            # 创建检测器
            print("[MediaPipe Hands] Creating HandLandmarker...")
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            
            print("[MediaPipe Hands] Initialized successfully!")
            return True
        except Exception as e:
            print(f"[MediaPipe Hands] Error initializing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect(self, frame: FrameData) -> List[DetectionResult]:
        """
        检测手部
        
        Args:
            frame: FrameData对象
            
        Return: DetectionResult列表，每个手为一个检测结果
        """
        if self.hand_landmarker is None:
            raise RuntimeError("Detector not initialized. Call initialize() first.")
        
        try:
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.image.shape
            
            # 创建MediaPipe Image
            mp_image_obj = mp_image(image_format=ImageFormat.SRGB, 
                                   data=image_rgb)
            
            # 运行检测
            result = self.hand_landmarker.detect(mp_image_obj)
            
            detections = []
            
            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    # 提取手部关键点坐标
                    landmarks = []
                    x_coords = []
                    y_coords = []
                    
                    for landmark in hand_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        z = landmark.z
                        landmarks.append([x, y, z])
                        x_coords.append(x)
                        y_coords.append(y)
                    
                    # 计算边界框
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    
                    # 获取手的标签（Left或Right）
                    hand_label = result.handedness[i][0].category_name if result.handedness else "Unknown"
                    confidence = result.handedness[i][0].score if result.handedness else 0.0
                    
                    # 创建检测结果
                    detection = DetectionResult(
                        class_name=f"Hand_{hand_label}",
                        confidence=confidence,
                        bbox=(x_min, y_min, x_max, y_max),
                        landmarks=np.array(landmarks)
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[MediaPipe Hands] Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def release(self) -> None:
        """释放资源"""
        if self.hand_landmarker is not None:
            self.hand_landmarker = None
            print("[MediaPipe Hands] Resources released")
    
    def draw_landmarks(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        绘制手部关键点和骨骼连接
        
        Args:
            frame: 图像数据
            detections: 检测结果列表
            
        Return: 绘制后的图像
        """
        image_copy = frame.copy()
        
        # 定义手部骨骼连接
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 指根连接
        ]
        
        for detection in detections:
            if detection.landmarks is not None:
                landmarks = detection.landmarks.astype(np.int32)
                
                # 绘制骨骼连接
                for connection in HAND_CONNECTIONS:
                    start = landmarks[connection[0]]
                    end = landmarks[connection[1]]
                    cv2.line(image_copy, tuple(start[:2]), tuple(end[:2]), (0, 255, 0), 2)
                
                # 绘制关键点
                for i, landmark in enumerate(landmarks):
                    cv2.circle(image_copy, tuple(landmark[:2]), 3, (255, 165, 0), -1)
        
        return image_copy
    
    def get_hand_gesture_info(self, landmarks: np.ndarray) -> dict:
        """
        根据关键点计算手部姿态信息
        
        Args:
            landmarks: 关键点坐标 (21, 3)
            
        Return: 手部姿态信息字典
        """
        if landmarks.shape[0] != 21:
            return {}
        
        try:
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # 计算一些基本的手部方向信息
            hand_x = index_tip[0] - wrist[0]
            hand_y = index_tip[1] - wrist[1]
            
            return {
                'wrist_pos': tuple(wrist[:2].astype(int)),
                'finger_tips': {
                    'thumb': tuple(thumb_tip[:2].astype(int)),
                    'index': tuple(index_tip[:2].astype(int)),
                    'middle': tuple(middle_tip[:2].astype(int)),
                    'ring': tuple(ring_tip[:2].astype(int)),
                    'pinky': tuple(pinky_tip[:2].astype(int))
                },
                'hand_direction': (hand_x, hand_y)
            }
        except Exception as e:
            print(f"[MediaPipe Hands] Error computing gesture info: {e}")
            return {}
