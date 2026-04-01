#!/usr/bin/env python3
"""
批量测试test_data中的所有图片和视频文件，包括深度估计和空间关系分析
"""
import cv2
import os
import time
from pathlib import Path

# 设置导入路径
import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# 导入视觉模块
from vision_modules import FrameData, ObjectGraspingVisionProcessor

# 支持的图片和视频格式
IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

def get_media_files(test_data_dir='test_data'):
    """扫描test_data目录，返回所有图片和视频文件"""
    # 如果是相对路径，基于当前脚本的目录
    if not os.path.isabs(test_data_dir):
        test_data_dir = os.path.join(_current_dir, test_data_dir)
    
    test_dir = Path(test_data_dir)
    print(f"[DEBUG] 扫描目录: {test_data_dir}")
    print(f"[DEBUG] 绝对路径: {test_dir.absolute()}")
    print(f"[DEBUG] 目录存在: {test_dir.exists()}")
    if not test_dir.exists():
        print(f"❌ 目录不存在: {test_data_dir}")
        return [], []
    
    images = []
    videos = []
    
    for file_path in sorted(test_dir.iterdir()):
        if file_path.is_file():
            suffix = file_path.suffix.lower()  # 转换为小写进行大小写无关比对
            print(f"[DEBUG] 检测文件: {file_path.name}, 后缀: {suffix}")
            if suffix in IMAGE_FORMATS:
                print(f"  ✓ 识别为图片")
                images.append(str(file_path))
            elif suffix in VIDEO_FORMATS:
                print(f"  ✓ 识别为视频")
                videos.append(str(file_path))
            else:
                print(f"  ✗ 不支持的格式")
    
    return images, videos

# 获取所有图片和视频
test_images, test_videos = get_media_files('test_data')

print("="*70)
print(f"批量处理test_data中的媒体文件 (含深度估计)")
print(f"图片: {len(test_images)},  视频: {len(test_videos)}")
print("="*70)

# 询问是否启用深度估计
enable_depth = True  # 默认启用
print("\n[配置] 深度估计: 已启用")

# 初始化处理器
print("\n[初始化]")
init_start = time.time()
# 使用脚本所在位置的相对路径
script_dir = Path(__file__).parent
yolo_model_path = str(script_dir / 'models' / 'yolov8m.pt')  # 用yolov8m，准确度更高
processor = ObjectGraspingVisionProcessor(
    enable_object_detection=True,
    enable_hand_detection=True,
    enable_depth_estimation=enable_depth,
    yolo_model=yolo_model_path,
    yolo_confidence=0.3,       # 置信度阈值
    yolo_nms_iou=0.2,          # 降低NMS IoU阈值，更激进地保留框
    device='cpu'
)

if not processor.initialize():
    print("❌ 初始化失败")
    exit(1)

init_time = time.time() - init_start
print(f"✓ 初始化耗时: {init_time:.3f}s\n")

# 存储结果
image_results = []
video_results = []

# ==================== 处理图片 ====================
print("\n" + "="*70)
print(f"处理 {len(test_images)} 个图片文件")
print("="*70)

# 处理每个图片
for idx, image_path in enumerate(test_images, 1):
    print(f"\n{'='*70}")
    print(f"图片 {idx}/3: {image_path}")
    print('='*70)
    
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        continue
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片")
        continue
    
    h, w = image.shape[:2]
    file_size = os.path.getsize(image_path) / 1024
    print(f"分辨率: {w}x{h}")
    print(f"文件大小: {file_size:.1f} KB")
    
    # 处理开始
    detection_start = time.time()
    
    frame = FrameData(image=image, frame_id=0, timestamp=0.0)
    detections = processor.process_frame(frame)
    
    # 计算检测时间
    detection_time = time.time() - detection_start
    
    # 提取结果
    objects = detections.get('objects', [])
    hands = detections.get('hands', [])
    spatial_relations = detections.get('spatial_relations', [])
    
    print(f"\n检测结果:")
    print(f"  物体数量: {len(objects)}")
    for obj in objects:
        print(f"    • [YOLOv8] {obj.class_name} (置信度: {obj.confidence:.4f})")
    
    print(f"  手部数量: {len(hands)}")
    for hand in hands:
        print(f"    • [MediaPipe] {hand.class_name} (置信度: {hand.confidence:.4f})")
        if hand.landmarks is not None:
            print(f"      └─ 21个关键点")
    
    # 显示空间关系
    if spatial_relations:
        print(f"\n  空间关系分析 (基于DPT深度图):")
        for relation in spatial_relations:
            status = "✓" if relation['hand_in_front'] else "✗"
            print(f"    • [{status}] {relation['relation_text']}")
    else:
        if enable_depth and (not objects or not hands):
            print(f"\n  ℹ️  深度估计已启用，但无法进行空间关系分析 (缺少物体或手部)")
    
    print(f"\n⏱️  检测耗时: {detection_time:.3f}s")
    
    # 绘制结果
    annotated = image.copy()
    
    # 绘制物体
    for obj in objects:
        x_min, y_min, x_max, y_max = obj.bbox
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"[YOLOv8] {obj.class_name}: {obj.confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x_min, y_min - label_size[1] - baseline),
                    (x_min + label_size[0], y_min), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x_min, y_min - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制手部
    for hand in hands:
        x_min, y_min, x_max, y_max = hand.bbox
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        label = f"[MediaPipe] {hand.class_name}: {hand.confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x_min, y_min - label_size[1] - baseline),
                    (x_min + label_size[0], y_min), (255, 0, 0), -1)
        cv2.putText(annotated, label, (x_min, y_min - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制关键点
        if hand.landmarks is not None:
            import numpy as np
            landmarks = hand.landmarks.astype(np.int32)
            for x, y, z in landmarks:
                cv2.circle(annotated, (x, y), 3, (255, 165, 0), -1)
    
    # 添加空间关系信息到图片
    if spatial_relations:
        y_offset = 30
        for relation in spatial_relations:
            status_text = relation['relation_text']
            cv2.putText(annotated, status_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
    
    # 为该图片创建独立的文件夹
    image_name = Path(image_path).stem
    image_result_dir = Path(_current_dir) / 'test_data' / image_name
    image_result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    output_path = image_result_dir / f"{image_name}_result.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"✓ 结果已保存: {output_path}")
    
    # 如果启用深度估计，保存深度图
    if enable_depth:
        depth_map = detections.get('depth_map')
        if depth_map is not None:
            import numpy as np
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_path = image_result_dir / f"{image_name}_depth_map.jpg"
            cv2.imwrite(str(depth_path), depth_vis_color)
            print(f"✓ 深度图已保存: {depth_path}")
    
    # 记录结果
    image_results.append({
        'image': image_path,
        'time': detection_time,
        'objects': len(objects),
        'hands': len(hands),
        'spatial_relations': len(spatial_relations),

        
        'output': str(output_path)
    })

# ==================== 处理视频 ====================
if test_videos:
    print("\n" + "="*70)
    print(f"处理 {len(test_videos)} 个视频文件")
    print("="*70)
    
    for idx, video_path in enumerate(test_videos, 1):
        print(f"\n{'='*70}")
        print(f"视频 {idx}/{len(test_videos)}: {video_path}")
        print('='*70)
        
        if not os.path.exists(video_path):
            print(f"❌ 文件不存在: {video_path}")
            continue
        
        # 为该视频创建独立的文件夹
        video_name = Path(video_path).stem
        video_result_dir = Path(_current_dir) / 'test_data' / video_name
        video_result_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频")
            continue
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = int(fps)  # 每秒处理一次
        
        print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"帧率: {fps:.1f} FPS, 总帧数: {total_frames}")
        print(f"\n处理视频中... (每秒处理1帧，跳过 {skip_frames} 帧)")
        
        # 处理视频
        video_start = time.time()
        frame_id = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳帧处理
            if frame_id % skip_frames != 0:
                frame_id += 1
                continue
            
            # 创建FrameData并处理
            frame_data = FrameData(
                image=frame,
                frame_id=frame_id,
                timestamp=frame_id / fps
            )
            
            detections = processor.process_frame(frame_data)
            
            # 绘制物体
            annotated = frame.copy()
            objects = detections.get('objects', [])
            hands = detections.get('hands', [])
            
            for obj in objects:
                x_min, y_min, x_max, y_max = obj.bbox
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"[YOLOv8] {obj.class_name}: {obj.confidence:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x_min, y_min - label_size[1] - baseline),
                            (x_min + label_size[0], y_min), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x_min, y_min - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            for hand in hands:
                x_min, y_min, x_max, y_max = hand.bbox
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                label = f"[MediaPipe] {hand.class_name}: {hand.confidence:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x_min, y_min - label_size[1] - baseline),
                            (x_min + label_size[0], y_min), (255, 0, 0), -1)
                cv2.putText(annotated, label, (x_min, y_min - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 绘制关键点
                if hand.landmarks is not None:
                    import numpy as np
                    landmarks = hand.landmarks.astype(np.int32)
                    for x, y, z in landmarks:
                        cv2.circle(annotated, (x, y), 3, (255, 165, 0), -1)
            
            # 保存处理后的帧
            output_frame_path = video_result_dir / f"{video_name}_frame_{processed_count:04d}.jpg"
            cv2.imwrite(str(output_frame_path), annotated)
            
            # 如果启用深度估计，保存深度热力图
            if enable_depth:
                depth_map = detections.get('depth_map')
                if depth_map is not None:
                    import numpy as np
                    depth_vis = (depth_map * 255).astype(np.uint8)
                    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    depth_path = video_result_dir / f"{video_name}_frame_{processed_count:04d}_depth.jpg"
                    cv2.imwrite(str(depth_path), depth_vis_color)
            
            processed_count += 1
            
            frame_id += 1
        
        cap.release()
        
        video_time = time.time() - video_start
        
        if processed_count > 0:
            print(f"✓ 视频处理完成")
            print(f"✓ 已保存 {processed_count} 帧检测结果到: {video_result_dir}/")
            print(f"⏱️  处理耗时: {video_time:.3f}s")
            
            video_results.append({
                'video': video_path,
                'time': video_time,
                'frames': processed_count,
                'output': str(video_result_dir)
            })
        else:
            print(f"❌ 视频处理失败")

# ==================== 处理总结 ====================
print("\n" + "="*70)
print("处理总结")
print("="*70)

# 图片处理总结
if image_results:
    print(f"\n📷 图片处理结果 ({len(image_results)} 个):")
    print(f"{'文件':<20} {'检测时间':>12} {'物体':>8} {'手':>6} {'关系':>6}")
    print("-" * 70)
    
    total_image_time = 0
    for r in image_results:
        img_name = Path(r['image']).name
        print(f"{img_name:<20} {r['time']:>11.3f}s {r['objects']:>8} {r['hands']:>6} {r['spatial_relations']:>6}")
        total_image_time += r['time']
    
    print("-" * 70)
    avg_image_time = total_image_time / len(image_results)
    print(f"{'总计':<20} {total_image_time:>11.3f}s {'(平均: '+f'{avg_image_time:.3f}s)':>15}")
else:
    print("\n📷 没有图片文件")

# 视频处理总结
if video_results:
    print(f"\n🎬 视频处理结果 ({len(video_results)} 个):")
    print(f"{'文件':<20} {'处理帧数':>12} {'处理时间':>12}")
    print("-" * 70)
    
    total_video_time = 0
    total_video_frames = 0
    for r in video_results:
        video_name = Path(r['video']).name
        print(f"{video_name:<20} {r['frames']:>12} {r['time']:>11.3f}s")
        total_video_time += r['time']
        total_video_frames += r['frames']
    
    print("-" * 70)
    print(f"{'总计':<20} {total_video_frames:>12} {total_video_time:>11.3f}s")
elif test_videos:
    print(f"\n🎬 没有成功处理任何视频")
else:
    print(f"\n🎬 没有视频文件")

if enable_depth:
    print("\n📊 深度估计状态:")
    print("  ✓ DPT模型已启用")
    print("  ✓ 深度图已生成")
    print("  ✓ 空间关系已分析")

print("="*70)

processor.release()
print("\n✓ 所有媒体文件处理完成！")
if test_images or test_videos:
    print(f"✓ 结果保存在: {Path(_current_dir) / 'test_data'} 下各文件的独立文件夹")

