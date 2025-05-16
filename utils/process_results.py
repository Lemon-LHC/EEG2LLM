import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties
import glob # 用于查找文件
import seaborn as sns

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签. 由于图表已改为英文，此行不再必要，且可能引起字体未找到的警告。
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --- 文件夹路径 --- 
model1_dir_path = "/data/lhc/results/llama3.2-1b-instruct_ST_edf5_100hz_15000ms_tok12672_balanced_0.1/intermediate_1100"
model2_dir_path = "/data/lhc/results/emotion_st44_ST_edf44_100hz_15000ms_raw_clean_tok12588_20250429_222055"
model3_dir_path = "/data/lhc/results/emotion_st44_sleep_st_44_100hz_eeg15s-step15s_emo2_20250429_182630"

# --- 辅助函数：在目录中查找唯一的Excel文件 ---
def find_unique_excel_file(dir_path):
    if not os.path.isdir(dir_path):
        print(f"错误：提供的路径并非一个有效目录: {dir_path}")
        return None
    
    excel_files = glob.glob(os.path.join(dir_path, "*.xlsx"))
    
    if not excel_files:
        print(f"错误：在目录 {dir_path} 中未找到任何 .xlsx 文件。")
        return None
    if len(excel_files) > 1:
        print(f"警告：在目录 {dir_path} 中找到多个 .xlsx 文件: {excel_files}。将使用第一个找到的文件: {excel_files[0]}")
        # 或者可以按某种规则选择，例如最新修改时间，但目前只取第一个
    return excel_files[0]

# --- 自动查找Excel文件路径 ---
model1_full_path = find_unique_excel_file(model1_dir_path)
model2_full_path = find_unique_excel_file(model2_dir_path)
model3_full_path = find_unique_excel_file(model3_dir_path)

# 检查文件是否都已找到
if not all([model1_full_path, model2_full_path, model3_full_path]):
    print("错误：未能找到所有必需的Excel文件。请检查上面列出的错误/警告信息。脚本将终止。")
    exit()

print(f"找到的模型1 Excel文件: {model1_full_path}")
print(f"找到的模型2 Excel文件: {model2_full_path}")
print(f"找到的模型3 Excel文件: {model3_full_path}")

# 提取文件名用于图表标注 (去除前缀)
path_prefix_to_remove = "/data/lhc/results/"
# 使用 os.path.relpath 来更安全地获取相对路径，如果前缀不完全匹配也能工作
model1_display_name = os.path.relpath(model1_full_path, path_prefix_to_remove) if model1_full_path.startswith(path_prefix_to_remove) else os.path.basename(model1_full_path)
model2_display_name = os.path.relpath(model2_full_path, path_prefix_to_remove) if model2_full_path.startswith(path_prefix_to_remove) else os.path.basename(model2_full_path)
model3_display_name = os.path.relpath(model3_full_path, path_prefix_to_remove) if model3_full_path.startswith(path_prefix_to_remove) else os.path.basename(model3_full_path)

# --- 创建输出目录 --- 
# 基于第三个模型的文件名创建子文件夹
output_parent_dir = "/data/lhc/projects/EEG2LLM/utils/output"
# 使用 model3_dir_path 的 basename 作为子目录名会更稳定，因为它代表了实验的配置
output_subdir_name = os.path.basename(model3_dir_path) 
os.makedirs(output_parent_dir, exist_ok=True)
output_dir_for_this_run = os.path.join(output_parent_dir, output_subdir_name)
os.makedirs(output_dir_for_this_run, exist_ok=True)

# 图例标签
legend_model1 = "llama3.2-1b-instruct"
legend_model2 = "EEG Fine-tuned"
legend_model3 = "Emotion+EEG Fine-tuned"

# 指标名称英文映射
metrics_translation = {
    '总体准确率': 'Overall Accuracy',
    '宏平均精确率': 'Macro Avg Precision',
    '宏平均召回率': 'Macro Avg Recall',
    '宏平均F1分数': 'Macro Avg F1 Score',
    '准确率': 'Accuracy',
    '精确率': 'Precision',
    '召回率': 'Recall',
    'F1分数': 'F1 Score'
}

# 读取数据
def read_metrics(file_path):
    try:
    # 读取总体性能指标（第二个子表的15-18行B列）
        overall_metrics_df = pd.read_excel(file_path, sheet_name="总体指标", header=None, usecols="B", skiprows=14, nrows=4)
        if overall_metrics_df.empty or overall_metrics_df.isnull().all().all():
            print(f"警告: 从 {file_path} 的 '总体指标' 表读取到的 overall_metrics 为空或全为NaN。将使用0填充。")
            overall_metrics = np.zeros(4)
        else:
            overall_metrics = overall_metrics_df.iloc[:, 0].values.flatten()
    
    # 读取详细类别指标（第三个子表的2-7行）
        class_metrics_df = pd.read_excel(file_path, sheet_name="详细类别指标", header=0, skiprows=0, nrows=6) # header=0 表示第一行为表头
        if class_metrics_df.empty:
            print(f"警告: 从 {file_path} 的 '详细类别指标' 表读取到的 class_metrics 为空。将创建空DataFrame。")
            class_metrics_df = pd.DataFrame(columns=['睡眠阶段', '准确率', '精确率', '召回率', 'F1分数'])
        else:
            expected_cols = ['睡眠阶段', '准确率', '精确率', '召回率', 'F1分数']
            if list(class_metrics_df.columns) != expected_cols:
                print(f"警告: {file_path} '详细类别指标'表的列名与预期不符。预期: {expected_cols}, 实际: {list(class_metrics_df.columns)}. 请检查Excel文件。")
            class_metrics_df = class_metrics_df[expected_cols]

        # --- 读取混淆矩阵 (B3:G8 in sheet "混淆矩阵") ---
        confusion_matrix_data = np.zeros((6, 6), dtype=int) # Default to zero matrix
        cm_source_info = "默认零矩阵 (读取失败)"

        try:
            # skiprows=2 (0-indexed) to start reading from the 3rd row.
            # usecols="B:G" to read columns B through G.
            # header=None, index_col=None as we are reading a raw block.
            cm_df = pd.read_excel(file_path, sheet_name="混淆矩阵", 
                                  header=None, index_col=None, 
                                  skiprows=2, usecols="B:G", nrows=6)
            
            if cm_df.shape == (6, 6) and not cm_df.isnull().values.any() and cm_df.map(lambda x: isinstance(x, (int, float, np.number))).all().all():
                confusion_matrix_data = cm_df.fillna(0).values.astype(int)
                cm_source_info = f"单元格 B3:G8 from sheet '混淆矩阵' in {file_path}"
                print(f"成功从 {cm_source_info} 加载混淆矩阵。")
            else:
                print(f"警告: 从 {file_path} 的 '混淆矩阵' 表的 B3:G8 区域读取到数据，但格式非预期6x6纯数字 (shape: {cm_df.shape}, contains_null: {cm_df.isnull().values.any()}, data_example: {cm_df.iloc[0,0] if not cm_df.empty else 'N/A'})。将使用6x6零矩阵。")
                # confusion_matrix_data remains zeros

        except Exception as e_cm:
            print(f"警告: 读取 {file_path} 的 '混淆矩阵' 子表 (区域 B3:G8) 时出错: {e_cm}。将使用6x6零矩阵。")
            # confusion_matrix_data remains zeros
        
        # Log if still using default after trying the specific method
        if cm_source_info.startswith("默认零矩阵"):
             print(f"最终警告: 未能从 {file_path} 的 '混淆矩阵' 表的 B3:G8 区域加载有效的6x6混淆矩阵。使用零矩阵。")

    except Exception as e:
        print(f"读取Excel文件 {file_path} 整体出错: {e}. 请确保文件路径和子表名称正确，并且文件未损坏。")
        overall_metrics = np.zeros(4)
        class_metrics_df = pd.DataFrame(columns=['睡眠阶段', '准确率', '精确率', '召回率', 'F1分数'])
        confusion_matrix_data = np.zeros((6, 6), dtype=int) # Ensure default CM data on major read error
    
    return {
        'overall': {
            '总体准确率': overall_metrics[0] if len(overall_metrics) > 0 else 0,
            '宏平均精确率': overall_metrics[1] if len(overall_metrics) > 1 else 0,
            '宏平均召回率': overall_metrics[2] if len(overall_metrics) > 2 else 0,
            '宏平均F1分数': overall_metrics[3] if len(overall_metrics) > 3 else 0
        },
        'class_metrics': class_metrics_df,
        'confusion_matrix': confusion_matrix_data
    }

# 添加辅助函数用于生成图表顶部的路径标注
def create_source_annotation(paths_display_names):
    return "Data Sources:\\n" + "\\n".join(paths_display_names)

# 读取三个实验的数据
try:
    data_model1 = read_metrics(model1_full_path)
    data_model2 = read_metrics(model2_full_path)
    data_model3 = read_metrics(model3_full_path)
    
    print("数据读取成功！")
    source_annotation_text = create_source_annotation([model1_display_name, model2_display_name, model3_display_name])
    
    # 1. 可视化总体性能指标对比
    metrics_overall_chinese = ['总体准确率', '宏平均精确率', '宏平均召回率', '宏平均F1分数']
    metrics_overall_english = [metrics_translation[m] for m in metrics_overall_chinese]
    
    values_model1 = [data_model1['overall'][m] for m in metrics_overall_chinese]
    values_model2 = [data_model2['overall'][m] for m in metrics_overall_chinese]
    values_model3 = [data_model3['overall'][m] for m in metrics_overall_chinese]
    
    # 计算提升百分比 (模型2 vs 模型1, 模型3 vs 模型1)
    improvement_m2_vs_m1 = [(m2 - m1) / m1 * 100 if m1 != 0 else (float('inf') if m2 > m1 else 0) for m1, m2 in zip(values_model1, values_model2)]
    improvement_m3_vs_m1 = [(m3 - m1) / m1 * 100 if m1 != 0 else (float('inf') if m3 > m1 else 0) for m1, m3 in zip(values_model1, values_model3)]

    fig_overall, ax_overall = plt.subplots(figsize=(15, 7))
    x_overall = np.arange(len(metrics_overall_english))
    width_overall = 0.25 # 调整宽度以容纳三个柱子
    
    bars_m1 = ax_overall.bar(x_overall - width_overall, values_model1, width_overall, label=legend_model1, color='#5A9BD5')
    bars_m2 = ax_overall.bar(x_overall, values_model2, width_overall, label=legend_model2, color='#ED7D31')
    bars_m3 = ax_overall.bar(x_overall + width_overall, values_model3, width_overall, label=legend_model3, color='#70AD47')
    
    ax_overall.set_ylabel('Metric Value')
    ax_overall.set_title('Overall Performance Metrics Comparison for Sleep Stage Classification')
    ax_overall.set_xticks(x_overall)
    ax_overall.set_xticklabels(metrics_overall_english)
    ax_overall.legend(loc='upper right', framealpha=0.7)
    fig_overall.text(0.01, 0.01, source_annotation_text, transform=fig_overall.transFigure, size=7, color='gray', ha='left', va='bottom')

    def add_labels_overall(bars, values, improvement_vs_m1=None, model_index=0):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax_overall.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{values[i]:.4f}', ha='center', va='bottom', fontsize=8)
            if improvement_vs_m1 and model_index > 0: # model_index 0 is baseline (m1)
                imp_val = improvement_vs_m1[i]
                if imp_val != 0 and not np.isinf(imp_val):
                     color = 'green' if imp_val > 0 else 'red'
                     sign = '+' if imp_val > 0 else ''
                     ax_overall.text(bar.get_x() + bar.get_width()/2., height/1.5, f'{sign}{imp_val:.1f}% (vs M1)', 
                                     ha='center', va='center', color='white', fontweight='bold', fontsize=7)
                elif np.isinf(imp_val):
                     display_text = 'Large Impr.' # 使用更安全的文本替代 '+Inf%'
                     ax_overall.text(bar.get_x() + bar.get_width()/2., height/1.5, f'{display_text} (vs M1)', 
                                     ha='center', va='center', color='white', fontweight='bold', fontsize=7)
    
    add_labels_overall(bars_m1, values_model1, model_index=0)
    add_labels_overall(bars_m2, values_model2, improvement_m2_vs_m1, model_index=1)
    add_labels_overall(bars_m3, values_model3, improvement_m3_vs_m1, model_index=2)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95]) # Adjust layout to make space for annotation
    # plt.savefig(os.path.join(output_dir_for_this_run, 'overall_performance_comparison_3models.png'), dpi=300, transparent=True)
    base_filename_overall = 'overall_performance_comparison_3models.png'
    plt.savefig(os.path.join(output_dir_for_this_run, base_filename_overall), dpi=300, transparent=False) # Opaque background
    plt.savefig(os.path.join(output_dir_for_this_run, base_filename_overall.replace('.png', '_transparent.png')), dpi=300, transparent=True) # Transparent background
    plt.close(fig_overall)
    
    # 2. 可视化各睡眠阶段的详细指标
    stages = data_model1['class_metrics']['睡眠阶段'].unique().tolist() # Get unique stages
    # Fallback if stages are not read correctly or are missing in one of the files
    if not stages:
        # Try to get stages from other models if model1 failed
        if not data_model2['class_metrics'].empty:
            stages = data_model2['class_metrics']['睡眠阶段'].unique().tolist()
        elif not data_model3['class_metrics'].empty:
            stages = data_model3['class_metrics']['睡眠阶段'].unique().tolist()
        else: # Default stages if all fail
            stages = ['W (Wakefulness)', 'N1 (Light Sleep)', 'N2 (Intermediate Sleep)', 'N3 (Deep Sleep)', 'N4 (Very Deep Sleep)', 'REM (Rapid Eye Movement)']
            print(f"警告: 无法从任何数据源确定睡眠阶段，使用默认阶段: {stages}")

    print(f"用于绘图的睡眠阶段: {stages}")

    metrics_class_chinese = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_class_english = [metrics_translation[m] for m in metrics_class_chinese]
    fig_class, axs_class = plt.subplots(2, 2, figsize=(18, 14))
    axs_class = axs_class.flatten()
    
    for i_metric, metric_name_chinese in enumerate(metrics_class_chinese):
        metric_name_english = metrics_class_english[i_metric]
        ax = axs_class[i_metric]
        
        m1_class_vals = [data_model1['class_metrics'][data_model1['class_metrics']['睡眠阶段'] == stage][metric_name_chinese].values[0] if stage in data_model1['class_metrics']['睡眠阶段'].values else 0 for stage in stages]
        m2_class_vals = [data_model2['class_metrics'][data_model2['class_metrics']['睡眠阶段'] == stage][metric_name_chinese].values[0] if stage in data_model2['class_metrics']['睡眠阶段'].values else 0 for stage in stages]
        m3_class_vals = [data_model3['class_metrics'][data_model3['class_metrics']['睡眠阶段'] == stage][metric_name_chinese].values[0] if stage in data_model3['class_metrics']['睡眠阶段'].values else 0 for stage in stages]

        imp_m2_vs_m1_class = [(m2 - m1) / m1 * 100 if m1 != 0 else (float('inf') if m2 > m1 else 0) for m1, m2 in zip(m1_class_vals, m2_class_vals)]
        imp_m3_vs_m1_class = [(m3 - m1) / m1 * 100 if m1 != 0 else (float('inf') if m3 > m1 else 0) for m1, m3 in zip(m1_class_vals, m3_class_vals)]

        x_class = np.arange(len(stages))
        bars_c_m1 = ax.bar(x_class - width_overall, m1_class_vals, width_overall, label=legend_model1, color='#5A9BD5')
        bars_c_m2 = ax.bar(x_class, m2_class_vals, width_overall, label=legend_model2, color='#ED7D31')
        bars_c_m3 = ax.bar(x_class + width_overall, m3_class_vals, width_overall, label=legend_model3, color='#70AD47')
        
        ax.set_ylabel(f'{metric_name_english} Value')
        ax.set_title(f'{metric_name_english} Comparison by Sleep Stage')
        ax.set_xticks(x_class)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.legend(framealpha=0.7)
        
        add_labels_overall(bars_c_m1, m1_class_vals, model_index=0)
        add_labels_overall(bars_c_m2, m2_class_vals, imp_m2_vs_m1_class, model_index=1)
        add_labels_overall(bars_c_m3, m3_class_vals, imp_m3_vs_m1_class, model_index=2)

    fig_class.suptitle('Detailed Metrics Comparison by Sleep Stage (Three Models)', fontsize=16)
    fig_class.text(0.01, 0.01, source_annotation_text, transform=fig_class.transFigure, size=7, color='gray', ha='left', va='bottom')
    
    # --- 尝试改进布局处理 ---
    try:
        # Preferred method for Matplotlib 3.6+
        fig_class.set_layout_engine("tight")
        print("Info: Using fig.set_layout_engine('tight') for detailed metrics plot.")
    except AttributeError:
        print("Info: fig.set_layout_engine('tight') not available. Using fig_class.tight_layout() with explicit DPI.")
        # Fallback for older Matplotlib
        fig_class.set_dpi(300) # Match savefig DPI before calculating layout
        fig_class.tight_layout(rect=[0.05, 0.05, 0.95, 0.93]) # Call on the figure object
    
    # --- 恢复 detailed_metrics_by_stage_3models 的 PNG 保存方式 ---
    base_filename_detailed = 'detailed_metrics_by_stage_3models.png'
    # DPI for savefig is still important, even if fig_class.set_dpi was called earlier for layout.
    plt.savefig(os.path.join(output_dir_for_this_run, base_filename_detailed), dpi=300, transparent=False) # Opaque background
    plt.savefig(os.path.join(output_dir_for_this_run, base_filename_detailed.replace('.png', '_transparent.png')), dpi=300, transparent=True) # Transparent background
    plt.close(fig_class)

    # 3. 创建热力图展示改进效果 (模型2 vs 模型1, 和 模型3 vs 模型1)
    def create_improvement_heatmap(data_new, data_baseline, baseline_model_name_short, new_model_name_short, title_suffix, filename_suffix, stages_list, metrics_list_chinese, metrics_list_english):
        improvement_matrix = np.zeros((len(stages_list), len(metrics_list_chinese)))
        for i_s, stage_val in enumerate(stages_list):
            for j_m, metric_val_chinese in enumerate(metrics_list_chinese):
                baseline_s_data = data_baseline['class_metrics'][data_baseline['class_metrics']['睡眠阶段'] == stage_val]
                baseline_metric_val = baseline_s_data[metric_val_chinese].values[0] if not baseline_s_data.empty else 0
            
                new_s_data = data_new['class_metrics'][data_new['class_metrics']['睡眠阶段'] == stage_val]
                new_metric_val = new_s_data[metric_val_chinese].values[0] if not new_s_data.empty else 0
            
                if baseline_metric_val != 0:
                    improvement_matrix[i_s, j_m] = (new_metric_val - baseline_metric_val) / baseline_metric_val * 100
                elif new_metric_val > baseline_metric_val: # baseline is 0, new is positive
                    improvement_matrix[i_s, j_m] = float('inf') # Represent as Inf
                else: # baseline is 0, new is 0 or negative (though metrics are non-negative)
                    improvement_matrix[i_s, j_m] = 0
        
        fig_h, ax_h = plt.subplots(figsize=(10, 8))
        im = ax_h.imshow(improvement_matrix, cmap='RdYlGn', vmin=-100, vmax=100) # Cap improvement at +/-100% for better color scale for typical cases
        
        ax_h.set_xticks(np.arange(len(metrics_list_chinese)))
        ax_h.set_yticks(np.arange(len(stages_list)))
        ax_h.set_xticklabels(metrics_list_english)
        ax_h.set_yticklabels(stages_list)
        plt.setp(ax_h.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
        cbar = ax_h.figure.colorbar(im, ax=ax_h)
        cbar.ax.set_ylabel(f"Improvement (%) (vs {baseline_model_name_short})", rotation=-90, va="bottom")
    
        for r_idx in range(len(stages_list)):
            for c_idx in range(len(metrics_list_chinese)):
                val = improvement_matrix[r_idx, c_idx]
                text_val = f"{val:.1f}%" if not np.isinf(val) else "+Inf%"
                color = "black"
                if not np.isinf(val):
                     color = "black" if abs(val) < 50 else "white" # Heuristic for text color on heatmap
                ax_h.text(c_idx, r_idx, text_val, ha="center", va="center", color=color)
        
        ax_h.set_title(f"Heatmap of Metric Improvement Pct. by Sleep Stage\\n({new_model_name_short} vs {baseline_model_name_short}{title_suffix})")
        fig_h.text(0.01, 0.01, source_annotation_text, transform=fig_h.transFigure, size=7, color='gray', ha='left', va='bottom')
        fig_h.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])
        # plt.savefig(os.path.join(output_dir_for_this_run, f'improvement_heatmap{filename_suffix}.png'), dpi=300, transparent=True)
        base_filename_heatmap = f'improvement_heatmap{filename_suffix}.png'
        plt.savefig(os.path.join(output_dir_for_this_run, base_filename_heatmap), dpi=300, transparent=False) # Opaque background
        plt.savefig(os.path.join(output_dir_for_this_run, base_filename_heatmap.replace('.png', '_transparent.png')), dpi=300, transparent=True) # Transparent background
        plt.close(fig_h)

    # 热力图1: 模型2 (脑电微调) vs 模型1 (原始)
    create_improvement_heatmap(data_model2, data_model1, legend_model1, legend_model2, "", "_m2_vs_m1", stages, metrics_class_chinese, metrics_class_english)
    # 热力图2: 模型3 (情绪+脑电微调) vs 模型1 (原始)
    create_improvement_heatmap(data_model3, data_model1, legend_model1, legend_model3, "", "_m3_vs_m1", stages, metrics_class_chinese, metrics_class_english)
    # (可选) 热力图3: 模型3 (情绪+脑电微调) vs 模型2 (脑电微调)
    create_improvement_heatmap(data_model3, data_model2, legend_model2, legend_model3, "", "_m3_vs_m2", stages, metrics_class_chinese, metrics_class_english)

    print(f"可视化结果已保存到 {output_dir_for_this_run} 目录")

    # 4. 绘制三个模型的混淆矩阵对比图
    short_sleep_stage_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'R'] # 用于混淆矩阵的标签

    def plot_combined_confusion_matrices(cm_list, model_legends, stage_labels, output_dir, fig_filename, annotation_text):
        if len(cm_list) != 3 or len(model_legends) != 3:
            print("错误: 绘制组合混淆矩阵需要三个模型的数据和图例。")
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 6.5)) # 调整figsize以使单元格更接近正方形并减少总宽度
        
        for i, ax in enumerate(axes):
            cm_data = cm_list[i]
            model_label = model_legends[i]
            
            if cm_data is None or cm_data.shape != (len(stage_labels), len(stage_labels)):
                print(f"警告: 模型 {model_label} 的混淆矩阵数据无效或形状不正确，将跳过绘制。期望 shape: ({len(stage_labels)}, {len(stage_labels)}), 得到: {cm_data.shape if cm_data is not None else 'None'}")
                ax.text(0.5, 0.5, 'Data N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Confusion Matrix: {model_label}\\n(Data Not Available)')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # --- 计算百分比并创建自定义标注 ---
            cm_row_sums = cm_data.sum(axis=1)
            cm_perc = np.zeros_like(cm_data, dtype=float)
            for r_idx in range(cm_data.shape[0]):
                if cm_row_sums[r_idx] > 0:
                    cm_perc[r_idx, :] = (cm_data[r_idx, :] / cm_row_sums[r_idx]) * 100
            
            annot_labels = (np.asarray([f"{count}\n({perc:.1f}%)" 
                                        for count, perc in zip(cm_data.flatten(), cm_perc.flatten())])
                           ).reshape(cm_data.shape)
            # ---

            sns.heatmap(cm_perc, annot=annot_labels, fmt='', cmap='Blues', ax=ax, 
                        xticklabels=stage_labels, yticklabels=stage_labels,
                        cbar= (i == len(axes) -1) , #只在最后一个子图显示colorbar
                        cbar_kws={'label': 'Percentage (%)'} if (i == len(axes) -1) else {},
                        annot_kws={"size": 7}, # 调整标注字体大小
                        vmin=0, vmax=100 # 设置颜色标尺范围为0-100
                        )
            
            ax.set_title(f'Confusion Matrix: {model_label}', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=10)
            if i == 0: # 只在第一个子图显示Y轴标签
                 ax.set_ylabel('True Label', fontsize=10)
            else:
                 ax.set_ylabel('') # 其他子图不显示Y轴标签，避免重复
            
            ax.tick_params(axis='x', labelsize=9, rotation=45)
            ax.tick_params(axis='y', labelsize=9, rotation=0)

        fig.suptitle('Comparison of Confusion Matrices for Sleep Stage Classification', fontsize=16, y=0.98) # 调整y以向下移动总标题
        fig.text(0.5, 0.01, annotation_text, transform=fig.transFigure, size=7, color='gray', ha='center', va='bottom')
        
        plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95]) # 确保rect的top值(0.95)能容纳下移后的标题
        
        output_path_opaque = os.path.join(output_dir, fig_filename)
        output_path_transparent = os.path.join(output_dir, fig_filename.replace('.png', '_transparent.png'))
        
        plt.savefig(output_path_opaque, dpi=300, transparent=False)
        plt.savefig(output_path_transparent, dpi=300, transparent=True)
        plt.close(fig)
        print(f"组合混淆矩阵图已保存到: {output_path_opaque} 和 {output_path_transparent}")

    # 调用新的绘图函数
    cms_to_plot = [
        data_model1.get('confusion_matrix'),
        data_model2.get('confusion_matrix'),
        data_model3.get('confusion_matrix')
    ]
    model_legends_for_cm = [legend_model1, legend_model2, legend_model3]
    
    plot_combined_confusion_matrices(
        cms_to_plot,
        model_legends_for_cm,
        short_sleep_stage_labels,
        output_dir_for_this_run,
        'combined_confusion_matrices_3models.png',
        source_annotation_text
    )

except FileNotFoundError as fnf_error:
    print(f"文件未找到错误: {fnf_error}. 请检查Excel文件路径是否正确。")
except KeyError as ke_error:
    print(f"键错误: {ke_error}. 这通常意味着Excel的子表名称或列名与代码中的预期不符。请检查 '总体指标' 和 '详细类别指标' 表是否存在且表头正确。")
except ValueError as ve_error:
    print(f"值错误: {ve_error}. 这可能发生在尝试从空的或格式不正确的数据单元格转换数据时。")
except Exception as e:
    print(f"处理数据或绘图时发生未知错误: {e}")
    import traceback
    print(traceback.format_exc())