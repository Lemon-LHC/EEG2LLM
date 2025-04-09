#!/usr/bin/env python
"""
直接运行测试评估脚本 - 手动评估最新检查点
"""
import os
import sys
import argparse
import glob
import importlib.util

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点目录
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        str: 最新检查点目录路径，如果未找到则返回None
    """
    # 查找所有checkpoint目录
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    
    if not checkpoint_dirs:
        print(f"在{checkpoint_dir}中没有找到检查点")
        return None
        
    # 按照步数排序
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[1]))
    
    # 返回最新的检查点
    latest = checkpoint_dirs[-1]
    print(f"找到最新检查点: {latest}")
    return latest

def import_test_function():
    """导入测试评估函数"""
    # 尝试从专门的模块导入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_script_path = os.path.join(script_dir, "run_test_evaluation_script.py")
    
    if os.path.exists(test_script_path):
        print(f"从专门模块导入测试函数: {test_script_path}")
        spec = importlib.util.spec_from_file_location("test_module", test_script_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        return test_module.run_test_evaluation_script
    
    # 如果专门模块不存在，尝试从train.py导入
    train_path = os.path.join(script_dir, "train.py")
    if os.path.exists(train_path):
        print(f"尝试从train.py导入测试函数: {train_path}")
        spec = importlib.util.spec_from_file_location("train_module", train_path)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        return getattr(train_module, "run_test_evaluation_script", None)
    
    return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="手动运行测试评估")
    
    # 必需参数
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="/data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf5_100hz_10000ms_tok8363_train",
        help="检查点目录"
    )
    parser.add_argument(
        "--test_data_path", 
        type=str, 
        default="/data/lhc/datasets_new/sleep/test/edf5_100hz_10000ms_tok8363_test.json",
        help="测试数据集路径"
    )
    parser.add_argument(
        "--tensorboard_dir", 
        type=str, 
        default="/data/lhc/saves/runs/Llama-3.2-1B-Instruct/edf5_100hz_10000ms_tok8363",
        help="TensorBoard输出目录"
    )
    parser.add_argument(
        "--step", 
        type=int, 
        default=None,
        help="指定步数，默认自动计算"
    )
    
    # 可选参数
    parser.add_argument("--device", type=str, default="cuda", help="设备:cuda或cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--detail_metrics", action="store_true", help="是否计算详细指标")
    parser.add_argument("--half_precision", action="store_true", help="是否使用半精度")
    
    args = parser.parse_args()
    
    # 查找最新检查点
    if "checkpoint-" not in args.checkpoint_dir:
        checkpoint_dir = find_latest_checkpoint(args.checkpoint_dir)
        if not checkpoint_dir:
            print("错误: 未找到有效的检查点目录")
            return 1
    else:
        checkpoint_dir = args.checkpoint_dir
    
    # 自动计算步数
    if args.step is None:
        try:
            step = int(os.path.basename(checkpoint_dir).split("-")[1])
            print(f"自动计算步数: {step}")
        except:
            step = 999999
            print(f"无法计算步数，使用默认值: {step}")
    else:
        step = args.step
    
    # 确保TensorBoard目录存在
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # 导入测试函数
    run_test_evaluation_script = import_test_function()
    if not run_test_evaluation_script:
        print("错误: 无法导入测试评估函数")
        return 1
    
    # 运行测试评估
    print("\n" + "="*50)
    print(f"运行测试评估 (步数: {step})")
    print(f"检查点: {checkpoint_dir}")
    print(f"测试数据: {args.test_data_path}")
    print(f"TensorBoard: {args.tensorboard_dir}")
    print("="*50 + "\n")
    
    success = run_test_evaluation_script(
        checkpoint_dir=checkpoint_dir,
        test_data_path=args.test_data_path,
        tensorboard_dir=args.tensorboard_dir,
        global_step=step,
        device=args.device,
        detail_metrics=args.detail_metrics,
        batch_size=args.batch_size,
        half_precision=args.half_precision
    )
    
    if success:
        print("\n✓ 测试评估成功完成")
        print(f"结果已写入TensorBoard: {args.tensorboard_dir}")
        return 0
    else:
        print("\n✗ 测试评估失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 