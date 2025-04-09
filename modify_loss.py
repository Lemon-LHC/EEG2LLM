#!/usr/bin/env python3
"""
修改LLaMA-Factory的损失函数计算
此脚本用于为LLaMA-Factory添加类别权重支持
"""

import os
import sys
import argparse
import glob
import re
from typing import List, Optional

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="为LLaMA-Factory添加类别权重支持")
    
    parser.add_argument("--llama_factory_path", type=str, 
                        default="/data/lhc/projects/LLaMA-Factory", 
                        help="LLaMA-Factory项目路径")
    parser.add_argument("--apply_patch", action="store_true",
                        help="应用补丁到LLaMA-Factory代码")
    parser.add_argument("--revert_patch", action="store_true",
                        help="还原LLaMA-Factory代码")
    
    args = parser.parse_args()
    return args

def find_trainer_files(llama_factory_path: str) -> List[str]:
    """查找LLaMA-Factory中的训练器文件"""
    trainer_paths = []
    
    # 常见位置
    common_paths = [
        os.path.join(llama_factory_path, "src/llamafactory/train/sft/trainer.py"),
        os.path.join(llama_factory_path, "src/llamafactory/train/trainer_utils.py"),
        os.path.join(llama_factory_path, "src/llamafactory/train/base.py"),
    ]
    
    # 检查常见位置
    for path in common_paths:
        if os.path.exists(path):
            trainer_paths.append(path)
    
    # 如果没找到，使用glob搜索
    if not trainer_paths:
        pattern = os.path.join(llama_factory_path, "**/*trainer*.py")
        trainer_paths = glob.glob(pattern, recursive=True)
    
    return trainer_paths

def add_weighted_loss_support(file_path: str) -> bool:
    """为训练器文件添加加权损失函数支持"""
    # 首先备份原始文件
    backup_path = file_path + ".bak"
    if not os.path.exists(backup_path):
        with open(file_path, "r", encoding="utf-8") as src:
            with open(backup_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        print(f"已备份原始文件: {backup_path}")
    
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 导入和定义部分修改
    import_patch = """
from typing import Optional, List
import torch
import torch.nn.functional as F

def parse_class_weights(weights_str: Optional[str] = None) -> Optional[torch.Tensor]:
    \"\"\"解析类别权重字符串\"\"\"
    if not weights_str:
        return None
    
    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        print(f"使用类别权重: {weights}")
        return torch.tensor(weights, dtype=torch.float)
    except Exception as e:
        print(f"解析类别权重出错: {e}")
        return None

"""
    
    # 检查是否包含导入torch的语句
    if "import torch" not in content:
        # 找到最后一个import语句
        import_match = re.search(r'^import.*$', content, re.MULTILINE)
        if import_match:
            pos = import_match.end()
            content = content[:pos] + "\nimport torch" + content[pos:]
    
    # 检查是否已经应用了补丁
    if "parse_class_weights" in content:
        print(f"文件 {file_path} 已经包含加权损失函数支持")
        return False
    
    # 寻找适合插入的位置
    # 通常是在导入语句之后，类定义之前
    insert_point = 0
    import_blocks = list(re.finditer(r'^import .*$|^from .* import .*$', content, re.MULTILINE))
    if import_blocks:
        # 找到最后一个导入语句后的位置
        last_import = import_blocks[-1]
        insert_point = last_import.end()
    
    # 插入导入和定义代码
    if insert_point > 0:
        content = content[:insert_point] + import_patch + content[insert_point:]
    
    # 寻找计算损失的代码块
    # 典型模式: loss = F.cross_entropy(logits, labels) 或 loss = cross_entropy(...)
    loss_patterns = [
        r'(loss\s*=\s*F\.cross_entropy\([^)]*\))',
        r'(loss\s*=\s*cross_entropy\([^)]*\))',
        r'(loss\s*=\s*criterion\([^)]*\))',
        r'(loss\s*=\s*self\.criterion\([^)]*\))',
    ]
    
    found_loss = False
    for pattern in loss_patterns:
        for match in re.finditer(pattern, content):
            found_loss = True
            orig_loss_calc = match.group(1)
            
            # 构建新的损失计算代码
            if "F.cross_entropy" in orig_loss_calc:
                # 提取参数
                params_match = re.search(r'F\.cross_entropy\(([^)]*)\)', orig_loss_calc)
                if params_match:
                    params = params_match.group(1)
                    # 插入权重参数
                    new_loss_calc = f"""
            # 使用加权损失函数（如果指定了类别权重）
            if hasattr(self.args, 'use_weighted_loss') and self.args.use_weighted_loss and hasattr(self.args, 'class_weights'):
                class_weights = parse_class_weights(self.args.class_weights)
                if class_weights is not None:
                    class_weights = class_weights.to(logits.device)
                    loss = F.cross_entropy({params}, weight=class_weights)
                else:
                    loss = F.cross_entropy({params})
            else:
                {orig_loss_calc}
                """
                    content = content.replace(orig_loss_calc, new_loss_calc)
            else:
                # 其他模式，添加注释说明需要手动修改
                comment = f"""
            # TODO: 需要手动修改以支持加权损失函数
            # 示例:
            # if hasattr(self.args, 'use_weighted_loss') and self.args.use_weighted_loss and hasattr(self.args, 'class_weights'):
            #     class_weights = parse_class_weights(self.args.class_weights)
            #     if class_weights is not None:
            #         class_weights = class_weights.to(logits.device)
            #         # 修改为合适的损失计算
            #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), weight=class_weights)
            #     else:
            #         {orig_loss_calc}
            # else:
            """
                content = content.replace(orig_loss_calc, comment + orig_loss_calc)
    
    if not found_loss:
        print(f"警告: 在文件 {file_path} 中未找到损失函数计算代码")
        return False
    
    # 更新参数解析部分，添加类别权重参数
    args_pattern = r'(parser\.add_argument\([^)]*\))'
    args_matches = list(re.finditer(args_pattern, content))
    if args_matches:
        # 找到最后一个参数添加位置
        last_arg = args_matches[-1]
        last_arg_pos = last_arg.end()
        
        # 添加类别权重参数
        class_weight_args = """
    parser.add_argument("--use_weighted_loss", action="store_true", help="是否使用加权损失函数")
    parser.add_argument("--class_weights", type=str, default=None, help="类别权重，格式为逗号分隔的浮点数列表")
"""
        # 确保不会插入到其他代码块中
        next_line_match = re.search(r'\n\s*\S', content[last_arg_pos:])
        if next_line_match:
            insert_pos = last_arg_pos + next_line_match.start()
            content = content[:insert_pos] + class_weight_args + content[insert_pos:]
        else:
            content = content[:last_arg_pos] + class_weight_args + content[last_arg_pos:]
    
    # 写入修改后的内容
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"已成功修改文件: {file_path}")
    return True

def restore_original_file(file_path: str) -> bool:
    """还原原始文件"""
    backup_path = file_path + ".bak"
    if os.path.exists(backup_path):
        with open(backup_path, "r", encoding="utf-8") as src:
            with open(file_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        print(f"已还原原始文件: {file_path}")
        return True
    else:
        print(f"未找到备份文件: {backup_path}")
        return False

def main():
    """主函数"""
    args = parse_arguments()
    
    # 查找训练器文件
    trainer_files = find_trainer_files(args.llama_factory_path)
    
    if not trainer_files:
        print(f"错误: 未在 {args.llama_factory_path} 中找到训练器文件")
        return 1
    
    print(f"找到以下训练器文件:")
    for i, file_path in enumerate(trainer_files):
        print(f"  [{i+1}] {file_path}")
    
    # 应用或还原补丁
    if args.apply_patch:
        success_count = 0
        for file_path in trainer_files:
            if add_weighted_loss_support(file_path):
                success_count += 1
        print(f"成功修改 {success_count}/{len(trainer_files)} 个文件")
    elif args.revert_patch:
        success_count = 0
        for file_path in trainer_files:
            if restore_original_file(file_path):
                success_count += 1
        print(f"成功还原 {success_count}/{len(trainer_files)} 个文件")
    else:
        print("请指定 --apply_patch 或 --revert_patch 参数")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 