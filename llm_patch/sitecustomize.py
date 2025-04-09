# 在Python启动时自动加载的模块
print("[测试回调] sitecustomize.py被加载，将注入测试回调")

import os
import sys
import importlib.util
import traceback

# 显示当前Python路径，帮助调试
try:
    print(f"[测试回调] 当前工作目录: {os.getcwd()}")
    print(f"[测试回调] Python路径: {sys.path}")
    
    # 打印环境变量
    print(f"[测试回调] PYTHONPATH 环境变量: {os.environ.get('PYTHONPATH', '未设置')}")
    
    # 添加当前目录的父目录到Python路径，确保能找到测试回调模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"[测试回调] 添加父目录到Python路径: {parent_dir}")
    
    # 显示所有可能相关的环境变量
    test_env_vars = [var for var in os.environ if var.startswith("TEST_") or "CALLBACK" in var or "PYTHON" in var]
    for var in test_env_vars:
        print(f"[测试回调] 环境变量 {var}: {os.environ[var]}")
    
    # 导入__init__.py中的补丁
    try:
        # 加载__init__.py
        init_path = os.path.join(current_dir, "__init__.py")
        if os.path.exists(init_path):
            print(f"[测试回调] 正在导入测试回调补丁: {init_path}")
            spec = importlib.util.spec_from_file_location("llm_patch", init_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("[测试回调] 成功导入测试回调补丁")
        else:
            print(f"[测试回调] 警告: {init_path} 不存在")
    except Exception as e:
        print(f"[测试回调] 导入测试回调补丁时出错: {e}")
        traceback.print_exc()
except Exception as e:
    print(f"[测试回调] sitecustomize.py初始化时出错: {e}")
    traceback.print_exc() 