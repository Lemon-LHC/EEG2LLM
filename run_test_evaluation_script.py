"""
单独的测试评估函数模块，确保可以从任何地方导入
"""

import os
import glob
import logging
import subprocess
import importlib.util
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_metrics_from_api_test():
    """
    从test_api_llm.py导入计算指标的函数
    
    Returns:
        函数: calculate_metrics函数或None
    """
    try:
        # 获取当前目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        api_test_path = os.path.join(script_dir, "test_api_llm.py")
        
        if os.path.exists(api_test_path):
            logger.info(f"发现test_api_llm.py，尝试导入calculate_metrics函数")
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("api_test", api_test_path)
            api_test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_test)
            
            # 获取calculate_metrics函数
            if hasattr(api_test, "calculate_metrics"):
                logger.info("成功导入calculate_metrics函数")
                return api_test.calculate_metrics
            else:
                logger.warning("test_api_llm.py中未找到calculate_metrics函数")
        else:
            logger.warning(f"未找到test_api_llm.py: {api_test_path}")
    except Exception as e:
        logger.error(f"导入calculate_metrics函数时出错: {e}")
        traceback.print_exc()
    
    return None

def get_sleep_stage_labels():
    """获取睡眠阶段标签
    
    Returns:
        list: 睡眠阶段标签列表
    """
    # 首先尝试从test_api_llm.py导入
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        api_test_path = os.path.join(script_dir, "test_api_llm.py")
        
        if os.path.exists(api_test_path):
            spec = importlib.util.spec_from_file_location("api_test", api_test_path)
            api_test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_test)
            
            if hasattr(api_test, "SLEEP_STAGE_LABELS"):
                return api_test.SLEEP_STAGE_LABELS
    except Exception:
        pass
    
    # 如果导入失败，使用默认值
    return [
        'Wake (W)', 
        'NREM Stage 1 (N1)', 
        'NREM Stage 2 (N2)', 
        'NREM Stage 3 (N3)', 
        'NREM Stage 4 (N4)', 
        'REM Sleep (R)'
    ]

def run_test_evaluation_script(checkpoint_dir, test_data_path, tensorboard_dir, global_step, device="cuda", detail_metrics=True, batch_size=8, half_precision=True):
    """运行测试评估脚本
    
    Args:
        checkpoint_dir: 检查点目录
        test_data_path: 测试数据路径
        tensorboard_dir: TensorBoard日志目录
        global_step: 当前步数
        device: 使用的设备
        detail_metrics: 是否记录详细的分类指标
        batch_size: 评估时的批处理大小
        half_precision: 是否使用半精度加速评估
        
    Returns:
        bool: 评估是否成功
    """
    logger.info(f"\n[检测到检查点] 开始测试集评估 (步数: {global_step})")
    
    # 获取测试脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(script_dir, "test_during_training.py")
    
    if not os.path.exists(test_script):
        # 尝试查找其他可能的脚本
        alternative_script = os.path.join(script_dir, "test_api_llm.py")
        if os.path.exists(alternative_script):
            logger.info(f"未找到test_during_training.py，使用替代脚本: {alternative_script}")
            test_script = alternative_script
        else:
            logger.error(f"错误: 测试脚本不存在: {test_script}")
            return False
    
    # 构建命令
    cmd = [
        "python", test_script,
        "--checkpoint_dir", checkpoint_dir,
        "--test_data_path", test_data_path,
        "--tensorboard_dir", tensorboard_dir,
        "--global_step", str(global_step),
        "--device", device,
        "--detail_metrics", "true" if detail_metrics else "false",
        "--batch_size", str(batch_size),
        "--half_precision", "true" if half_precision else "false"
    ]
    
    # 运行命令
    try:
        logger.info(f"执行测试命令: {' '.join(cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"  # 设置MKL线程层为GNU以避免与libgomp冲突
        env["PYTHONPATH"] = f"{script_dir}:{env.get('PYTHONPATH', '')}"  # 确保能找到相关模块
        
        # 添加MODEL_NAME环境变量（基于检查点路径）
        if "MODEL_NAME" not in env:
            model_name = os.path.basename(os.path.dirname(checkpoint_dir))
            env["MODEL_NAME"] = f"{model_name}_{os.path.basename(checkpoint_dir)}"
            logger.info(f"设置MODEL_NAME环境变量: {env['MODEL_NAME']}")
        
        # 运行进程并实时显示输出
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(line, end='')  # 保留这个直接输出，因为它来自子进程
        
        # 等待进程完成
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"成功评估检查点: {checkpoint_dir}")
            # 验证TensorBoard日志是否已写入
            tensorboard_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents*"))
            if tensorboard_files:
                logger.info(f"发现TensorBoard事件文件: {[os.path.basename(f) for f in tensorboard_files]}")
                return True
            else:
                logger.warning(f"警告: 未找到TensorBoard事件文件，指标可能未被记录")
                return False
        else:
            logger.error(f"评估检查点失败，返回码: {process.returncode}")
            return False
    except Exception as e:
        logger.error(f"运行评估脚本时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

# 如果作为主程序运行，提供命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行测试评估")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="检查点目录")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--tensorboard_dir", type=str, required=True, help="TensorBoard输出目录")
    parser.add_argument("--global_step", type=int, required=True, help="当前步数")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    parser.add_argument("--detail_metrics", type=str, default="true", help="是否记录详细指标")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--half_precision", type=str, default="true", help="是否使用半精度")
    
    args = parser.parse_args()
    
    # 将字符串参数转换为布尔值
    detail_metrics = args.detail_metrics.lower() == "true"
    half_precision = args.half_precision.lower() == "true"
    
    # 执行测试评估
    success = run_test_evaluation_script(
        checkpoint_dir=args.checkpoint_dir,
        test_data_path=args.test_data_path,
        tensorboard_dir=args.tensorboard_dir,
        global_step=args.global_step,
        device=args.device,
        detail_metrics=detail_metrics,
        batch_size=args.batch_size,
        half_precision=half_precision
    )
    
    # 设置退出代码
    import sys
    sys.exit(0 if success else 1) 