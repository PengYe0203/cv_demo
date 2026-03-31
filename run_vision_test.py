#!/usr/bin/env python3
"""
调用视觉模块的测试脚本

使用方法:
    python run_vision_test.py
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 执行 vision_modules 中的测试脚本
if __name__ == '__main__':
    # 改变工作目录到 vision_modules
    os.chdir(os.path.join(project_root, 'vision_modules'))
    
    # 读取测试脚本
    test_script = os.path.join(project_root, 'vision_modules', 'test_all_images.py')
    
    # 准备全局命名空间
    global_namespace = {
        '__file__': test_script,
        '__name__': '__main__',
    }
    
    # 执行脚本
    with open(test_script, 'r', encoding='utf-8') as f:
        code = f.read()
    
    exec(compile(code, test_script, 'exec'), global_namespace)
