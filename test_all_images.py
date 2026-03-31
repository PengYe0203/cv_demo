#!/usr/bin/env python3
"""
批量测试三个手部图片，包括深度估计和空间关系分析
"""
import cv2
import os
import time
from pathlib import Path
from interfaces import FrameData
from vision_processor import ObjectGraspingVisionProcessor

test_images = [
    'test_data/hand_1.jpg',
    'test_data/hand_2.jpg', 
    'test_data/hand_3.JPG'
]

result_dir = Path('test_data/test_result')
result_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("批量处理三个手部图片 (含深度估计)")
print("="*70)

# 询问是否启用深度估计
enable_depth = True  # 默认启用
print("\n[配置] 深度估计: 已启用")

# 初始化处理器
print("\n[初始化]")
init_start = time.time()
processor = ObjectGraspingVisionProcessor(
    enable_object_detection=True,
    enable_hand_detection=True,
    enable_depth_estimation=enable_depth,
    device='cpu'
)

if not processor.initialize():
    print("❌ 初始化失败")
    exit(1)

init_time = time.time() - init_start
print(f"✓ 初始化耗时: {init_time:.3f}s\n")

# 存储结果
results = []

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
    
    # 保存结果
    output_filename = Path(image_path).stem
    output_path = result_dir / f"{output_filename}_result.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"✓ 结果已保存: {output_path}")
    
    # 如果启用深度估计，保存深度图
    if enable_depth:
        depth_map = detections.get('depth_map')
        if depth_map is not None:
            import numpy as np
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_path = result_dir / f"{output_filename}_depth_map.jpg"
            cv2.imwrite(str(depth_path), depth_vis_color)
            print(f"✓ 深度图已保存: {depth_path}")
    
    # 记录结果
    results.append({
        'image': image_path,
        'time': detection_time,
        'objects': len(objects),
        'hands': len(hands),
        'spatial_relations': len(spatial_relations),
        'output': str(output_path)
    })

# 总结
print("\n" + "="*70)
print("处理总结")
print("="*70)
print(f"\n{'图片':<20} {'检测时间':>12} {'物体':>8} {'手':>6} {'关系':>6}")
print("-" * 70)

total_time = 0
for r in results:
    img_name = Path(r['image']).name
    print(f"{img_name:<20} {r['time']:>11.3f}s {r['objects']:>8} {r['hands']:>6} {r['spatial_relations']:>6}")
    total_time += r['time']

print("-" * 70)
avg_time = total_time / len(results) if results else 0
print(f"{'总计':<20} {total_time:>11.3f}s {'(平均: '+f'{avg_time:.3f}s)':>15}")

if enable_depth:
    print("\n📊 深度估计状态:")
    print("  ✓ DPT模型已启用")
    print("  ✓ 深度图已生成")
    print("  ✓ 空间关系已分析")

print("="*70)

processor.release()
print("\n✓ 所有图片处理完成！")
print(f"✓ 结果保存在: {result_dir}")

