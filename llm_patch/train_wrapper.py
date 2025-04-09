"""
这个模块包装了projects/fine/train.py中的函数，
确保可以安全地从其他模块导入这些函数
"""

import os
import sys
import importlib.util
import traceback

# 获取当前目录的父目录
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 首先尝试导入专门的测试评估函数
try:
    # 尝试从专门的模块导入
    test_script_path = os.path.join(script_dir, "run_test_evaluation_script.py")
    if os.path.exists(test_script_path):
        print(f"[测试回调] 从专门模块导入测试函数: {test_script_path}")
        spec = importlib.util.spec_from_file_location("test_module", test_script_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        run_test_evaluation_script = test_module.run_test_evaluation_script
        print("[测试回调] 成功从专门模块导入run_test_evaluation_script函数")
        
    # 如果专门模块不存在，再尝试从train.py导入
    else:
        # 使用importlib动态导入train.py
        train_file = os.path.join(script_dir, "train.py")
        if not os.path.exists(train_file):
            print(f"[测试回调] 错误: 训练文件不存在: {train_file}")
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
            
        print(f"[测试回调] 加载训练模块: {train_file}")
        spec = importlib.util.spec_from_file_location("train_module", train_file)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        # 导出需要的函数
        run_test_evaluation_script = train_module.run_test_evaluation_script
        print("[测试回调] 成功从train.py导入run_test_evaluation_script函数")
        
        # 导出其他可能需要的函数
        calculate_metrics = getattr(train_module, "calculate_metrics", None)
        if calculate_metrics:
            print("[测试回调] 成功导入calculate_metrics函数")
    
except Exception as e:
    print(f"[测试回调] 导入训练模块函数时出错: {e}")
    traceback.print_exc()
    
    # 创建一个备用函数，避免导入错误
    def run_test_evaluation_script(*args, **kwargs):
        print("[测试回调] 警告: 使用了备用的run_test_evaluation_script函数")
        print(f"[测试回调] 原始参数: {args}, {kwargs}")
        return False 