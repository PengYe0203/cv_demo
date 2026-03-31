#!/usr/bin/env python3
"""
视觉模块的主入口
当作为包执行时调用此文件
"""
import sys
import os

# 执行测试脚本
if __name__ == '__main__':
    # 获取 vision_modules 的目录
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 将导入改为直接路径相关的导入
    import importlib.util
    
    test_script = os.path.join(module_dir, 'test_all_images.py')
    spec = importlib.util.spec_from_file_location("test_all_images", test_script)
    test_module = importlib.util.module_from_spec(spec)
    
    # 执行测试脚本
    spec.loader.exec_module(test_module)
