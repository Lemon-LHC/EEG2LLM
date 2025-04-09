# 用于向LLaMA-Factory添加测试评估回调
import os
import sys
import importlib.util

# 添加自定义目录到Python路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    print(f"[测试回调] 添加目录到Python路径: {script_dir}")

# 导入测试回调
try:
    # 先导入测试回调类
    print(f"[测试回调] 尝试导入测试回调类...")
    from projects.fine.test_callback import TestEvaluationCallback
    print(f"[测试回调] 成功导入测试回调类")
    
    # 导入包装模块中的函数
    print(f"[测试回调] 尝试导入训练函数...")
    from .train_wrapper import run_test_evaluation_script
    print(f"[测试回调] 成功导入训练函数")
except ImportError as e:
    print(f"[测试回调] 导入错误: {e}")
    
    try:
        # 备选导入方式
        print(f"[测试回调] 尝试备选导入方式...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 导入测试回调
        callback_path = os.path.join(script_dir, "test_callback.py")
        if os.path.exists(callback_path):
            spec = importlib.util.spec_from_file_location("test_callback_module", callback_path)
            callback_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(callback_module)
            TestEvaluationCallback = callback_module.TestEvaluationCallback
            print(f"[测试回调] 通过备选方式导入TestEvaluationCallback")
        
        # 导入train_wrapper模块
        wrapper_path = os.path.join(current_dir, "train_wrapper.py")
        if os.path.exists(wrapper_path):
            spec = importlib.util.spec_from_file_location("train_wrapper_module", wrapper_path)
            wrapper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wrapper_module)
            run_test_evaluation_script = wrapper_module.run_test_evaluation_script
            print(f"[测试回调] 通过备选方式导入run_test_evaluation_script")
    except Exception as e:
        print(f"[测试回调] 备选导入方式也失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 创建一个空的回调类和函数，以避免导入错误
        class TestEvaluationCallback:
            def __init__(self, *args, **kwargs):
                print("[测试回调] 警告: 使用了空的TestEvaluationCallback类")
            def on_evaluate(self, *args, **kwargs):
                pass
                
        def run_test_evaluation_script(*args, **kwargs):
            print("[测试回调] 警告: 使用了空的run_test_evaluation_script函数")
            return False

def load_and_print_env():
    """加载和打印环境变量"""
    # 检查是否启用测试回调
    enabled = os.environ.get("ENABLE_TEST_CALLBACK", "").lower() == "true"
    
    # 获取环境变量
    test_interval = int(os.environ.get("TEST_INTERVAL", "20"))
    test_dataset_path = os.environ.get("TEST_DATASET_PATH", "")
    tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "")
    test_batch_size = int(os.environ.get("TEST_BATCH_SIZE", "1"))
    eval_precision = os.environ.get("EVAL_PRECISION", "fp16")
    
    print("[测试回调] 环境变量:")
    print(f"  • 启用测试回调: {enabled}")
    print(f"  • 测试间隔: {test_interval}步")
    print(f"  • 测试数据集: {test_dataset_path}")
    print(f"  • TensorBoard目录: {tensorboard_dir}")
    print(f"  • 测试批处理大小: {test_batch_size}")
    print(f"  • 评估精度: {eval_precision}")
    
    return {
        "enabled": enabled,
        "test_interval": test_interval,
        "test_dataset_path": test_dataset_path,
        "tensorboard_dir": tensorboard_dir,
        "test_batch_size": test_batch_size,
        "eval_precision": eval_precision
    }

# 打补丁到LLaMA-Factory的sft训练流程
def patch_sft_workflow():
    """给LLaMA-Factory的SFT训练流程添加测试回调"""
    
    # 加载和打印环境变量
    env_vars = load_and_print_env()
    
    # 检查是否启用测试回调
    if not env_vars["enabled"]:
        print("测试回调未启用，跳过")
        return
    
    try:
        # 验证测试数据集路径
        test_dataset_path = env_vars["test_dataset_path"]
        if not test_dataset_path or not os.path.exists(test_dataset_path):
            print(f"测试数据集路径无效: {test_dataset_path}")
            return
            
        print(f"开始给LLaMA-Factory添加测试回调...")
        
        # 找到LLaMA-Factory的sft workflow模块
        try:
            from llamafactory.train.sft.workflow import run_sft
            original_run_sft = run_sft
            
            # 创建测试参数对象
            class Args:
                def __init__(self):
                    self.test_interval = env_vars["test_interval"]
                    self.train_batch_size = env_vars["test_batch_size"]
                    # 移除self.eval_precision，使用fp16和bf16标志
                    self.fp16 = env_vars["eval_precision"] == "fp16"
                    self.bf16 = env_vars["eval_precision"] == "bf16"
            
            # 定义包装函数
            def wrapped_run_sft(*args, **kwargs):
                print("[测试回调] 正在注入测试回调到训练流程...")
                
                # 创建测试回调
                test_args = Args()
                test_callback = TestEvaluationCallback(
                    args=test_args,
                    test_data_path=test_dataset_path,
                    tensorboard_dir=env_vars["tensorboard_dir"],
                    run_test_evaluation_fn=run_test_evaluation_script
                )
                
                # 添加到回调列表
                if "callbacks" in kwargs and kwargs["callbacks"] is not None:
                    kwargs["callbacks"].append(test_callback)
                else:
                    kwargs["callbacks"] = [test_callback]
                
                # 调用原始函数
                return original_run_sft(*args, **kwargs)
            
            # 替换原始函数
            from llamafactory.train.sft import workflow
            workflow.run_sft = wrapped_run_sft
            
            print("[测试回调] 成功注入测试回调到训练流程")
            
        except ImportError:
            print("[测试回调] 未找到LLaMA-Factory的SFT模块，无法添加测试回调")
            
    except Exception as e:
        print(f"[测试回调] 添加测试回调时出错: {e}")
        import traceback
        traceback.print_exc()

# 在导入时自动执行补丁
patch_sft_workflow() 