"""
主程序 - 测试物体抓取辅助系统
支持测试本地图片和视频
"""
import cv2
import numpy as np
import os
import time
from pathlib import Path
from typing import Optional
from vision_modules import FrameData, ObjectGraspingVisionProcessor


class VisionTestApp:
    """视觉系统测试应用"""
    
    def __init__(self, 
                 yolo_model: str = 'yolov8n.pt',
                 yolo_confidence: float = 0.5,
                 device: str = '0'):
        """
        初始化测试应用
        
        Args:
            yolo_model: YOLOv8模型名称
            yolo_confidence: YOLOv8置信度阈值
            device: 运算设备（'0'表示GPU，'cpu'表示CPU）
        """
        self.processor = ObjectGraspingVisionProcessor(
            enable_object_detection=True,
            enable_hand_detection=True,
            yolo_model=os.path.join('vision_modules', 'models', yolo_model),
            yolo_confidence=yolo_confidence,
            device=device
        )
        self.frame_count = 0
    
    def initialize(self) -> bool:
        """初始化应用"""
        print("[App] Initializing vision processor...")
        print("[App] Note: First run may take longer due to model downloads")
        if not self.processor.initialize():
            print("\n[App] ⚠️  Vision processor initialization had issues")
            print("[App] This may be due to:")
            print("      • First-time model downloads")
            print("      • GPU not available (using --device cpu)")
            print("      • Network connectivity issues")
            print("\n[App] Continuing with available components...")
            # 不返回False，继续运行
        
        print("[App] System initialized")
        return True
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """
        处理单张图片
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径（可选）
            
        Return: 处理是否成功
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 读取图片
            if not os.path.exists(image_path):
                print(f"[App] Image not found: {image_path}")
                return False
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"[App] Failed to read image: {image_path}")
                return False
            
            print(f"\n[App] Processing image: {image_path}")
            print(f"[App] Image shape: {image.shape}")
            
            # 创建FrameData
            frame = FrameData(
                image=image,
                frame_id=0,
                timestamp=0.0
            )
            
            # 记录检测开始时间
            detection_start = time.time()
            
            # 处理帧
            detections = self.processor.process_frame(frame)
            
            # 计算检测时间
            detection_time = time.time() - detection_start
            detections['detection_time'] = detection_time
            
            # 输出结果
            self._print_results(detections)
            
            # 绘制结果
            annotated_image = self._draw_detections(image, detections)
            
            # 保存或显示结果
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                print(f"[App] Result saved to: {output_path}")
            else:
                cv2.imshow('Detection Result', annotated_image)
                print("[App] Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # 计算总时间
            total_time = time.time() - start_time
            print(f"\n[App] ⏱️  总耗时: {total_time:.3f}s (检测: {detection_time:.3f}s)")
            
            return True
            
        except Exception as e:
            print(f"[App] Error processing image: {e}")
            return False
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     skip_frames: int = 1) -> bool:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            skip_frames: 跳帧数（可选，用于加快处理）
            
        Return: 处理是否成功
        """
        try:
            # 打开视频
            if not os.path.exists(video_path):
                print(f"[App] Video not found: {video_path}")
                return False
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[App] Failed to open video: {video_path}")
                return False
            
            print(f"\n[App] Processing video: {video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[App] Video info - FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
            
            # 初始化视频写入（如果指定了输出路径）
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                print(f"[App] Output video will be saved to: {output_path}")
            
            frame_id = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳帧处理
                if frame_id % skip_frames != 0:
                    frame_id += 1
                    continue
                
                # 创建FrameData
                frame_data = FrameData(
                    image=frame,
                    frame_id=frame_id,
                    timestamp=frame_id / fps
                )
                
                # 处理帧
                detections = self.processor.process_frame(frame_data)
                
                # 绘制结果
                annotated_frame = self._draw_detections(frame, detections)
                
                # 添加决策信息
                decision_text = self.processor.make_decision(detections)
                if decision_text:
                    cv2.putText(annotated_frame, decision_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 保存或显示
                if output_path and out:
                    out.write(annotated_frame)
                
                # 每处理100帧输出一次进度
                processed_frames += 1
                if processed_frames % 100 == 0:
                    print(f"[App] Processed {processed_frames} frames...")
                
                frame_id += 1
            
            # 清理资源
            cap.release()
            if out:
                out.release()
            
            print(f"[App] Video processing completed. Total processed frames: {processed_frames}")
            return True
            
        except Exception as e:
            print(f"[App] Error processing video: {e}")
            return False
    
    def _draw_detections(self, image: np.ndarray, detections: dict) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 图像数据
            detections: 检测结果字典
            
        Return: 绘制后的图像
        """
        annotated = image.copy()
        
        # 绘制物体边界框 (YOLOv8)
        for obj in detections.get('objects', []):
            x_min, y_min, x_max, y_max = obj.bbox
            
            # 绘制边界框
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 绘制标签 - 添加模型标识
            label = f"[YOLOv8] {obj.class_name}: {obj.confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x_min, y_min - label_size[1] - baseline),
                        (x_min + label_size[0], y_min), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x_min, y_min - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 绘制手部边界框 (MediaPipe)
        for hand in detections.get('hands', []):
            x_min, y_min, x_max, y_max = hand.bbox
            
            # 绘制边界框
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # 绘制标签 - 添加模型标识
            label = f"[MediaPipe] {hand.class_name}: {hand.confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x_min, y_min - label_size[1] - baseline),
                        (x_min + label_size[0], y_min), (255, 0, 0), -1)
            cv2.putText(annotated, label, (x_min, y_min - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制手部关键点（如果有）
            if hand.landmarks is not None:
                landmarks = hand.landmarks.astype(np.int32)
                for i, (x, y, z) in enumerate(landmarks):
                    cv2.circle(annotated, (x, y), 3, (255, 165, 0), -1)
        
        return annotated
    
    def _print_results(self, detections: dict) -> None:
        """
        打印检测结果
        
        Args:
            detections: 检测结果字典
        """
        print("\n" + "="*60)
        print(f"Frame ID: {detections['frame_id']}, Timestamp: {detections['timestamp']:.4f}")
        
        # 物体检测结果 (YOLOv8)
        objects = detections.get('objects', [])
        print(f"\n[YOLOv8] Detected Objects: {len(objects)}")
        for i, obj in enumerate(objects):
            print(f"  [{i+1}] {obj.class_name} - Confidence: {obj.confidence:.4f}, "
                  f"BBox: {obj.bbox}")
        
        # 手部检测结果 (MediaPipe)
        hands = detections.get('hands', [])
        print(f"\n[MediaPipe] Detected Hands: {len(hands)}")
        for i, hand in enumerate(hands):
            print(f"  [{i+1}] {hand.class_name} - Confidence: {hand.confidence:.4f}, "
                  f"BBox: {hand.bbox}")
            if hand.landmarks is not None:
                print(f"       Landmarks shape: {hand.landmarks.shape}")
        
        # 决策输出
        decision = self.processor.make_decision(detections)
        print(f"\nDecision: {decision}")
        print("="*60)
    
    def release(self) -> None:
        """释放资源"""
        self.processor.release()
        print("[App] Application released")
    
    def __enter__(self):
        """上下文管理器支持"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.release()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vision-based Object Grasping Assistance System"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input image or video file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (for images/videos with annotations)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLOv8 model name'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='YOLOv8 confidence threshold'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='0',
        choices=['0', 'cpu'],
        help='Device to use (0 for GPU, cpu for CPU)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=1,
        help='Skip frames in video processing (for speed)'
    )
    
    args = parser.parse_args()
    
    # 创建应用
    with VisionTestApp(
        yolo_model=args.model,
        yolo_confidence=args.confidence,
        device=args.device
    ) as app:
        if args.input:
            # 处理指定的文件
            file_ext = Path(args.input).suffix.lower()
            
            # 如果没有指定输出路径，自动生成输出路径到test_data/test_result
            if args.output is None:
                input_filename = Path(args.input).stem
                result_dir = Path('test_data/test_result')
                result_dir.mkdir(parents=True, exist_ok=True)
                args.output = str(result_dir / f"{input_filename}_result{file_ext}")
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 处理图片
                app.process_image(args.input, args.output)
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 处理视频
                app.process_video(args.input, args.output, args.skip_frames)
            else:
                print(f"Unsupported file format: {file_ext}")
        else:
            print("\n" + "="*70)
            print("No input file provided. Please use --input to specify a file.")
            print("="*70)
            print("\nUsage Examples:")
            print("  Image:  python main.py --input sample.jpg --output result.jpg")
            print("  Video:  python main.py --input video.mp4 --output output.mp4")
            print("  CPU:    python main.py --input sample.jpg --device cpu")
            print("  Model:  python main.py --input sample.jpg --model yolov8s.pt")
            print("\nAvailable options:")
            print("  --input, -i:        Input file path (required)")
            print("  --output, -o:       Output file path (optional)")
            print("  --model, -m:        YOLOv8 model (default: yolov8n.pt)")
            print("  --confidence, -c:   Confidence threshold (default: 0.5)")
            print("  --device, -d:       Device: 0 (GPU) or cpu (default: 0)")
            print("  --skip-frames:      Skip N frames in video (default: 1)")
            print("="*70)


if __name__ == '__main__':
    main()
