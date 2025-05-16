import json
import re
from collections import defaultdict
import os # 添加 os 模块用于路径操作
import argparse # 新增：用于命令行参数解析
from tqdm import tqdm # 新增 tqdm 导入

# --- 添加绘图库导入 ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    visualization_available = True
except ImportError:
    visualization_available = False
    print("Warning: matplotlib, seaborn or pandas not available. Visualization charts will not be generated.")
    print("Please run 'pip install matplotlib seaborn pandas' to install these libraries.")
# --- 结束绘图库导入 ---

def get_all_16_emotion_codes():
    """生成所有16个四位二进制情绪代码的列表 (从 '0000' 到 '1111')"""
    return [format(i, '04b') for i in range(16)]

# --- 新增：定义重点情绪代码 ---
HIGHLIGHTED_EMOTION_CODES = ["0001", "0010", "0100", "1000","0000"]
# --- 结束新增 ---

def parse_emotion_ratio_string(input_str):
    """
    从input字符串中解析 'Emotion ratio:' 部分。
    例如: "Emotion ratio: 0000:47%, 1000:11%, ..., 1111:0%."
    返回一个字典，如 {'0000': 47.0, '1000': 11.0, ...}
    """
    ratios = {}
    # 匹配 "Emotion ratio: " 和其后的第一个 "." 之间的内容
    ratio_match = re.search(r"Emotion ratio: (.*?)\.", input_str)
    if not ratio_match:
        return ratios # 如果没有找到 "Emotion ratio: ... ." 模式，返回空字典

    ratio_part_str = ratio_match.group(1) # 获取 "0000:47%, ..., 1111:0%"
    
    items = ratio_part_str.split(', ')
    for item in items:
        parts = item.split(':')
        if len(parts) == 2:
            code = parts[0].strip()
            percentage_str = parts[1].strip().rstrip('%')
            try:
                percentage = float(percentage_str)
                # 验证情绪代码是否为4位二进制
                if re.fullmatch(r"[01]{4}", code):
                    ratios[code] = percentage
                else:
                    print(f"警告: 在情绪比例中发现无效的情绪代码格式 '{code}'。已跳过。")
            except ValueError:
                print(f"警告: 无法从 '{item}' 解析情绪百分比。已跳过。")
                continue
    return ratios

# --- 新增：情绪序列解析函数 ---
def parse_emotion_sequence_string(input_str):
    """
    Parses the 'Emotion sequence:' part from the input string.
    Example: "Emotion sequence: 0010>0100>0000"
    Returns a list of emotion codes in the sequence, or None if not found.
    """
    # Regex to find "Emotion sequence: " followed by a > separated list of 4-bit binary codes
    sequence_match = re.search(r"Emotion sequence: ([01]{4}(?:>[01]{4})*)", input_str)
    if sequence_match:
        sequence_part_str = sequence_match.group(1)
        return sequence_part_str.split('>')
    return None
# --- 结束新增 ---

def calculate_emotion_ratios_per_stage(file_path, base_output_dir, input_filename_without_ext):
    """
    主函数，用于计算并打印每个睡眠阶段的情绪占比，并生成可视化图表。
    """
    # file_path, base_output_dir, input_filename_without_ext are now passed as arguments
    
    output_plot_filename = os.path.join(base_output_dir, f"emotion_stage_ratios_{input_filename_without_ext}.png")
    
    # parent_folder_name = os.path.basename(os.path.dirname(file_path)) # 保留以备后用

    sleep_stage_map = {
        0: "W (Wakefulness)",
        1: "N1 (Light Sleep)",
        2: "N2 (Intermediate Sleep)",
        3: "N3 (Deep Sleep)",
        4: "N4 (Very Deep Sleep)",
        5: "REM (Rapid Eye Movement)"
    }
    all_emotion_codes = get_all_16_emotion_codes()
    # other_emotion_codes_for_subplot = [code for code in all_emotion_codes if code not in HIGHLIGHTED_EMOTION_CODES]


    # 初始化用于存储每个阶段各情绪百分比总和的结构
    emotion_totals_by_stage = {
        stage_code: {emotion_code: 0.0 for emotion_code in all_emotion_codes} 
        for stage_code in range(6) # 对应睡眠阶段 0 到 5
    }
    # 初始化用于存储每个阶段样本数量的结构
    sample_counts_by_stage = {stage_code: 0 for stage_code in range(6)}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到于 '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法从 '{file_path}' 解码JSON。请检查文件格式。")
        return
    except Exception as e:
        print(f"读取文件时发生预料之外的错误: {e}")
        return

    for i, sample in enumerate(tqdm(data, desc="正在计算情绪比率", ncols=100)):
        try:
            sleep_stage_str = sample.get("output")
            if sleep_stage_str is None:
                print(f"警告: 样本 {i} 缺少 'output' 字段。已跳过。")
                continue
            
            try:
                sleep_stage = int(sleep_stage_str)
            except ValueError:
                print(f"警告: 样本 {i} 的睡眠阶段 'output' ('{sleep_stage_str}') 不是有效整数。已跳过。")
                continue

            if not (0 <= sleep_stage <= 5):
                print(f"警告: 样本 {i} 的睡眠阶段 '{sleep_stage}' 超出有效范围 [0-5]。已跳过。")
                continue
                
            input_str = sample.get("input", "")
            if not input_str:
                sample_counts_by_stage[sleep_stage] += 1
                continue

            parsed_ratios = parse_emotion_ratio_string(input_str)
            sample_counts_by_stage[sleep_stage] += 1

            for code, percentage in parsed_ratios.items():
                if code in emotion_totals_by_stage[sleep_stage]:
                    emotion_totals_by_stage[sleep_stage][code] += percentage
                else:
                    # This case should ideally not happen if all_emotion_codes covers all possible codes
                    # and parse_emotion_ratio_string validates codes.
                    print(f"警告: 在样本 {i} 中发现未知或格式错误的情绪代码 '{code}'。已跳过此情绪代码。")

        except Exception as e:
            print(f"处理样本 {i} 时发生错误: {e}。样本内容: {str(sample)[:200]}...") # Keeping sample content for debug
            continue
            
    print("\n--- 每个睡眠阶段的平均情绪占比 (16种详细情绪) ---")
    
    average_emotion_ratios_by_stage = {
        s: {code: 0.0 for code in all_emotion_codes} for s in range(6)
    }
    aggregated_ratios_for_subplot = {
        s_code: {emotion_code: 0.0 for emotion_code in HIGHLIGHTED_EMOTION_CODES + ["Other Codes"]}
        for s_code in range(6)
    }

    for s_code in range(6):
        stage_name_for_plot = sleep_stage_map.get(s_code, f"Unknown Stage {s_code}") # Plot titles use this English name
        # For print, decide if Chinese stage name is needed or use the English one for consistency.
        # Let's use the English name from map for print to align with plot if plot labels are fixed English.
        print(f"\n睡眠阶段: {stage_name_for_plot} (代码: {s_code})")
        
        num_samples_for_stage = sample_counts_by_stage[s_code]
        
        if num_samples_for_stage > 0:
            print(f"  (基于 {num_samples_for_stage} 个样本)")
            sum_other_codes_avg_for_subplot = 0.0
            for e_code in all_emotion_codes:
                total_percentage_for_emotion = emotion_totals_by_stage[s_code][e_code]
                average = total_percentage_for_emotion / num_samples_for_stage
                average_emotion_ratios_by_stage[s_code][e_code] = average
                print(f"  情绪代码 {e_code}: {average:.2f}%")
                
                if e_code in HIGHLIGHTED_EMOTION_CODES:
                    aggregated_ratios_for_subplot[s_code][e_code] = average
                else:
                    sum_other_codes_avg_for_subplot += average
            aggregated_ratios_for_subplot[s_code]["Other Codes"] = sum_other_codes_avg_for_subplot # "Other Codes" is a key
        else:
            print(f"  此阶段没有样本。")
            for e_code in all_emotion_codes:
                 average_emotion_ratios_by_stage[s_code][e_code] = 0.0
            for h_code in HIGHLIGHTED_EMOTION_CODES:
                aggregated_ratios_for_subplot[s_code][h_code] = 0.0
            aggregated_ratios_for_subplot[s_code]["Other Codes"] = 0.0
            
    count_bits = lambda s: s.count('1')
    conflict_codes = [code for code in all_emotion_codes if count_bits(code) > 1]
    none_code = "0000" # This is now also in HIGHLIGHTED_EMOTION_CODES

    total_sum_for_categories = defaultdict(float)
    total_samples_overall = sum(sample_counts_by_stage.values())

    if total_samples_overall > 0:
        for s_code in range(6):
            if sample_counts_by_stage[s_code] > 0:
                for e_code in all_emotion_codes:
                    avg_percentage_in_stage = average_emotion_ratios_by_stage[s_code].get(e_code, 0.0)
                    # Weight by number of samples in this stage for overall average
                    total_sum_for_categories[e_code] += avg_percentage_in_stage * sample_counts_by_stage[s_code]

        avg_overall_highlighted_sum = 0.0
        avg_overall_conflict_sum = 0.0
        avg_overall_other_single_sum = 0.0 # Single '1' codes not in HIGHLIGHTED_EMOTION_CODES

        for e_code in all_emotion_codes:
            overall_avg_for_ecode = total_sum_for_categories[e_code] / total_samples_overall
            if e_code in HIGHLIGHTED_EMOTION_CODES:
                avg_overall_highlighted_sum += overall_avg_for_ecode
            elif e_code in conflict_codes: # Multiple '1's
                avg_overall_conflict_sum += overall_avg_for_ecode
            elif count_bits(e_code) == 1: # Single '1' but not in HIGHLIGHTED
                avg_overall_other_single_sum += overall_avg_for_ecode
            # '0000' is handled if it's in HIGHLIGHTED_EMOTION_CODES. If not, it would be missed here.
            # Given "0000" is in HIGHLIGHTED_EMOTION_CODES, this logic is fine.

        print(f"\n--- 总体情绪类别平均占比 (基于 {total_samples_overall} 个样本) ---")
        print(f"  重点情绪编码 ({', '.join(HIGHLIGHTED_EMOTION_CODES)}) 总和平均占比: {avg_overall_highlighted_sum:.2f}%")
        print(f"  混合/冲突情绪编码 (多个\'1\', 不在重点中) 总和平均占比: {avg_overall_conflict_sum:.2f}%")
        print(f"  其他单位一情绪编码 (单个\'1\', 不在重点中) 总和平均占比: {avg_overall_other_single_sum:.2f}%")


        if avg_overall_conflict_sum < avg_overall_highlighted_sum: # Comparing conflict vs explicitly highlighted
             print("  结论: 数据表明，混合/冲突情绪编码的总体平均占比较重点情绪编码的总和要低。")
        elif avg_overall_conflict_sum > avg_overall_highlighted_sum:
            print("  提示: 数据表明，混合/冲突情绪编码的总体平均占比较重点情绪编码的总和要高。")
        else:
            print("  提示: 数据表明，混合/冲突情绪编码的总体平均占比与重点情绪编码的总和相当。")
    else:
        print("\n--- 无有效样本进行总体情绪类别分析 ---")

    print("\n--- 每个睡眠阶段的重点情绪类别平均占比 (聚合) ---")
    for s_code in range(6):
        stage_name_for_plot = sleep_stage_map.get(s_code, f"Unknown Stage {s_code}") # Using English name for print
        print(f"\n睡眠阶段: {stage_name_for_plot} (代码: {s_code})")
        if sample_counts_by_stage[s_code] > 0:
            print(f"  (基于 {sample_counts_by_stage[s_code]} 个样本)")
            # cat_code_or_label will be "0000", "0001", ..., "Other Codes"
            for cat_code_or_label in HIGHLIGHTED_EMOTION_CODES + ["Other Codes"]: # "Other Codes" is a key
                percentage = aggregated_ratios_for_subplot[s_code][cat_code_or_label]
                print(f"  类别 '{cat_code_or_label}': {percentage:.2f}%")
        else:
            print(f"  此阶段没有样本。")
    
    print("\n--- 情绪比率阶段分析完成 ---")

    if visualization_available:
        print(f"\n--- 正在生成情绪比率图表 ---") 
        source_file_basename = os.path.basename(file_path) # For titles

        # --- 图1: 详细情绪热力图 (独立文件) ---
        try:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 10)) # Optimal size for heatmap
            
            heatmap_data_list = []
            ordered_stage_labels_for_plot = []
            for s_code_loop in sorted(average_emotion_ratios_by_stage.keys()): 
                ordered_stage_labels_for_plot.append(sleep_stage_map.get(s_code_loop, f"Unknown {s_code_loop}"))
                row_data = [average_emotion_ratios_by_stage[s_code_loop][e_code] for e_code in all_emotion_codes]
                heatmap_data_list.append(row_data)

            df_heatmap = pd.DataFrame(heatmap_data_list, index=ordered_stage_labels_for_plot, columns=all_emotion_codes)

            sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, 
                        cbar_kws={'label': 'Average Emotion Ratio (%)'}, ax=ax_heatmap) # Keep English
            
            ax_heatmap.set_title(f'Detailed Emotion Ratios per Sleep Stage (16 Emotions)\nSource: {source_file_basename}', fontsize=14) # Keep English
            ax_heatmap.set_xlabel("Emotion Code (4-bit binary)", fontsize=12) # Keep English
            ax_heatmap.set_ylabel("Sleep Stage", fontsize=12) # Keep English
            ax_heatmap.tick_params(axis='x', rotation=45, labelsize=10)
            ax_heatmap.tick_params(axis='y', rotation=0, labelsize=10)

            fig_heatmap.tight_layout()
            
            heatmap_plot_filename = os.path.join(base_output_dir, f"emotion_stage_ratios_heatmap_{input_filename_without_ext}.png")
            plt.savefig(heatmap_plot_filename, dpi=300)
            print(f"情绪阶段比率热力图已保存至: {heatmap_plot_filename}")
            plt.close(fig_heatmap) 

        except Exception as e:
            print(f"生成或保存情绪阶段比率热力图时出错: {e}")
            import traceback
            print(traceback.format_exc())

        # --- 图2: 重点情绪类别条形图 (独立文件) ---
        try:
            fig_barchart, ax_barchart = plt.subplots(figsize=(12, 8)) # Optimal size for bar chart

            subplot_categories = HIGHLIGHTED_EMOTION_CODES + ["Other Codes"] 
            subplot_plot_data = {cat: [] for cat in subplot_categories}
            subplot_stage_labels = []

            for s_code_loop in sorted(aggregated_ratios_for_subplot.keys()): 
                stage_label = sleep_stage_map.get(s_code_loop, f"Unknown {s_code_loop}")
                subplot_stage_labels.append(stage_label)
                for cat in subplot_categories:
                    subplot_plot_data[cat].append(aggregated_ratios_for_subplot[s_code_loop][cat])
            
            df_subplot = pd.DataFrame(subplot_plot_data, index=subplot_stage_labels)
            
            emotion_explanations = { 
                "0000": "0000 (No Emotion)",
                "0001": "0001 (HVHA)",
                "0010": "0010 (HVLA)",
                "0100": "0100 (LVHA)",
                "1000": "1000 (LVLA)",
                "Other Codes": "Other Codes" 
            }
            df_subplot.columns = [emotion_explanations.get(col, col) for col in df_subplot.columns]

            df_subplot.plot(kind='bar', ax=ax_barchart, width=0.8)
            
            ax_barchart.set_title(f'Aggregated Highlighted Emotion Category Ratios per Sleep Stage\nSource: {source_file_basename}', fontsize=14) # Keep English
            ax_barchart.set_xlabel("Sleep Stage", fontsize=12) # Keep English
            ax_barchart.set_ylabel("Average Ratio (%)", fontsize=12) # Keep English
            ax_barchart.tick_params(axis='x', rotation=30, labelsize=10)
            ax_barchart.tick_params(axis='y', labelsize=10)
            ax_barchart.legend(title="Emotion Category", fontsize=9, title_fontsize=10) # Keep English
            ax_barchart.grid(axis='y', linestyle='--', alpha=0.7)

            for p in ax_barchart.patches:
                height = p.get_height()
                if height > 0: 
                    ax_barchart.text(p.get_x() + p.get_width() / 2.,
                             height + 0.5, 
                             f'{height:.1f}%', 
                             ha='center', va='bottom', fontsize=7)

            fig_barchart.tight_layout()
            
            barchart_plot_filename = os.path.join(base_output_dir, f"emotion_stage_ratios_barchart_{input_filename_without_ext}.png")
            plt.savefig(barchart_plot_filename, dpi=300)
            print(f"情绪阶段重点类别条形图已保存至: {barchart_plot_filename}")
            plt.close(fig_barchart) 

        except Exception as e:
            print(f"生成或保存情绪阶段重点类别条形图时出错: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("\n提示: 可视化库 (matplotlib, seaborn, pandas) 未完全加载或存在问题。跳过情绪比率图表生成。")

# --- 新增：马尔科夫链转换分析函数 ---
def analyze_emotion_transitions_markov(file_path, base_output_dir, input_filename_without_ext):
    """
    Analyzes emotion transitions using a Markov chain approach and saves a heatmap.
    """
    if not visualization_available:
        print("警告: 可视化库 (matplotlib, seaborn, pandas) 不可用。跳过马尔科夫链转换分析。")
        return

    all_codes = get_all_16_emotion_codes()
    # --- 修改：按睡眠阶段初始化转换计数 ---
    transition_counts_by_stage = {
        stage: pd.DataFrame(0, index=all_codes, columns=all_codes, dtype=float) 
        for stage in range(6) # 0-5 for sleep stages
    }
    total_transitions_processed_by_stage = defaultdict(int)
    # --- 结束修改 ---

    # Define sleep stage map locally for convenience in this function
    sleep_stage_map = {
        0: "W (Wakefulness)", 
        1: "N1 (Light Sleep)", 
        2: "N2 (Intermediate Sleep)",
        3: "N3 (Deep Sleep)", 
        4: "N4 (Very Deep Sleep)",
        5: "REM (Rapid Eye Movement)"
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误 (马尔科夫): 文件未找到于 '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误 (马尔科夫): 无法从 '{file_path}' 解码JSON。请检查文件格式。")
        return
    except Exception as e:
        print(f"错误 (马尔科夫): 读取文件时发生意外错误: {e}")
        return

    total_transitions_processed = 0
    for i, sample in enumerate(tqdm(data, desc="正在分析一阶马尔可夫转换", ncols=100)):
        input_str = sample.get("input", "")
        sequence = parse_emotion_sequence_string(input_str)
        
        sleep_stage_str = sample.get("output")
        current_stage_code = -1
        if sleep_stage_str is not None:
            try:
                current_stage_code = int(sleep_stage_str)
                if not (0 <= current_stage_code <= 5):
                    print(f"警告 (马尔科夫 样本 {i}): 睡眠阶段 '{current_stage_code}' 超出范围 [0-5]。已跳过此样本的马尔科夫分析。")
                    current_stage_code = -1
            except ValueError:
                print(f"警告 (马尔科夫 样本 {i}): 无效的睡眠阶段 '{sleep_stage_str}'。已跳过此样本的马尔科夫分析。")
                current_stage_code = -1
        else:
            print(f"警告 (马尔科夫 样本 {i}): 缺少 'output' (睡眠阶段)字段。已跳过此样本的马尔科夫分析。")
            current_stage_code = -1
        
        if current_stage_code == -1:
            continue

        if sequence and len(sequence) > 1:
            for j in range(len(sequence) - 1):
                from_emotion = sequence[j]
                to_emotion = sequence[j+1]
                
                if re.fullmatch(r"[01]{4}", from_emotion) and re.fullmatch(r"[01]{4}", to_emotion):
                    if from_emotion in transition_counts_by_stage[current_stage_code].index and \
                       to_emotion in transition_counts_by_stage[current_stage_code].columns:
                        transition_counts_by_stage[current_stage_code].loc[from_emotion, to_emotion] += 1
                        total_transitions_processed_by_stage[current_stage_code] += 1
                        total_transitions_processed +=1
                    else:
                        print(f"警告 (马尔科夫 阶段 {current_stage_code} 样本 {i}): 情绪代码 '{from_emotion}' -> '{to_emotion}' 不在矩阵索引/列中。已跳过。")
                else:
                    print(f"警告 (马尔科夫 阶段 {current_stage_code} 样本 {i}): 序列中情绪代码格式无效。 '{from_emotion}' -> '{to_emotion}'。已跳过。")
        elif sequence and len(sequence) <= 1:
            pass 

    if total_transitions_processed == 0:
        print("信息 (马尔科夫): 未找到有效的情绪序列或在所有阶段均未处理任何转换以进行马尔科夫链分析。")
        return

    print("\n--- 按睡眠阶段进行的马尔科夫链转换分析 (完整16x16热力图) ---")
    
    all_stage_full_transition_probabilities = {} 

    for stage_code_loop in range(6):
        stage_name_for_plot = sleep_stage_map.get(stage_code_loop, f"Unknown Stage {stage_code_loop}") # Keep English for plot title
        print(f"\n--- 阶段: {stage_name_for_plot} (代码: {stage_code_loop}) ---")

        current_stage_counts = transition_counts_by_stage[stage_code_loop]
        num_transitions_in_stage = total_transitions_processed_by_stage[stage_code_loop]

        if num_transitions_in_stage == 0:
            print(f"  信息: 阶段 {stage_name_for_plot} 没有记录到转换。")
            # Store an empty DataFrame or a DataFrame of zeros for consistency if needed by subplot logic
            all_stage_full_transition_probabilities[stage_code_loop] = pd.DataFrame(0, index=all_codes, columns=all_codes, dtype=float)
            continue

        row_sums = current_stage_counts.sum(axis=1)
        transition_probabilities = current_stage_counts.div(row_sums.replace(0, 1), axis=0).fillna(0)
        all_stage_full_transition_probabilities[stage_code_loop] = transition_probabilities # Store for subplots


        print(f"  转换概率 (来自每个状态的前5个转换，基于此阶段的 {num_transitions_in_stage} 次转换):")
        for from_state in transition_probabilities.index:
            if row_sums[from_state] > 0: 
                top_transitions = transition_probabilities.loc[from_state].sort_values(ascending=False).head(5)
                print(f"    来自 {from_state}:")
                for to_state, prob in top_transitions.items():
                    if prob > 0: 
                        print(f"      -> {to_state}: {prob:.3f} (计数: {int(current_stage_counts.loc[from_state, to_state])})")

        fig_markov, ax_markov = plt.subplots(figsize=(18, 16))
        sns.heatmap(transition_probabilities, annot=True, fmt=".2f", cmap="viridis", linewidths=.5,
                    cbar_kws={'label': 'Transition Probability'}, ax=ax_markov, square=True) # Keep English
        
        ax_markov.set_title(f'Emotion Transition Probability Matrix (Markov Chain)\nStage: {stage_name_for_plot} - Source: {os.path.basename(file_path)}', fontsize=16) # Keep English
        ax_markov.set_xlabel('To Emotion Code', fontsize=12) # Keep English
        ax_markov.set_ylabel('From Emotion Code', fontsize=12) # Keep English
        ax_markov.tick_params(axis='x', rotation=45, labelsize=10) 
        ax_markov.tick_params(axis='y', rotation=0, labelsize=10)
        
        output_filename = os.path.join(base_output_dir, f"markov_stage_{stage_code_loop}_{input_filename_without_ext}.png")
        try:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"  阶段 {stage_name_for_plot} 的马尔科夫链热力图已保存至: {output_filename}")
        except Exception as e:
            print(f"  保存阶段 {stage_name_for_plot} 的马尔科夫链热力图时出错: {e}")
            import traceback
            print(traceback.format_exc())
        plt.close(fig_markov)

    if visualization_available and any(total_transitions_processed_by_stage.values()):
        print("\n--- 正在为重点情绪生成马尔科夫链子图 ---")
        
        focused_codes = HIGHLIGHTED_EMOTION_CODES 
        
        fig_subplots, axes_subplots = plt.subplots(2, 3, figsize=(22, 12))
        axes_subplots = axes_subplots.flatten()

        plot_created_for_subplots = False
        for i, stage_code_loop_subplot in enumerate(range(6)):
            ax = axes_subplots[i]
            stage_name_for_plot_subplot = sleep_stage_map.get(stage_code_loop_subplot, f"Stage {stage_code_loop_subplot}") # Keep English for plot title
            
            full_probs_this_stage = all_stage_full_transition_probabilities.get(stage_code_loop_subplot)

            # Check if full_probs_this_stage is not None and actually has data for this stage
            if full_probs_this_stage is not None and not full_probs_this_stage.empty and total_transitions_processed_by_stage[stage_code_loop_subplot] > 0:
                
                sub_matrix_probs = full_probs_this_stage.reindex(index=focused_codes, columns=focused_codes).fillna(0.0)
                
                sns.heatmap(sub_matrix_probs, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5,
                            cbar=False, ax=ax, square=True, annot_kws={"size": 8}) # Keep English
                ax.set_title(f"Stage: {stage_name_for_plot_subplot} ({len(focused_codes)} Focused Codes)", fontsize=10) # Keep English
                ax.set_xlabel("To Emotion", fontsize=8) # Keep English
                ax.set_ylabel("From Emotion", fontsize=8) # Keep English
                ax.tick_params(axis='x', rotation=45, labelsize=7)
                ax.tick_params(axis='y', rotation=0, labelsize=7)
                plot_created_for_subplots = True
            else:
                ax.text(0.5, 0.5, "无转换数据", ha='center', va='center') # Chinese message
                ax.set_title(f"阶段: {stage_name_for_plot_subplot}\n(无转换)", fontsize=10) # Chinese message in title
                ax.axis('off')

        if plot_created_for_subplots:
            fig_subplots.suptitle(f'Markov Transitions for Highlighted Emotions ({len(focused_codes)} codes) by Sleep Stage\nSource: {os.path.basename(file_path)}', fontsize=16, y=0.99) # Keep English
            fig_subplots.tight_layout(rect=[0, 0, 1, 0.96])
            
            subplot_output_filename = os.path.join(base_output_dir, f"markov_highlighted_subplots_{input_filename_without_ext}.png")
            try:
                plt.savefig(subplot_output_filename, dpi=300)
                print(f"\n重点情绪马尔科夫子图已保存至: {subplot_output_filename}")
            except Exception as e:
                print(f"\n保存重点情绪马尔科夫子图时出错: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            print("\n信息: 无数据可生成重点情绪马尔科夫子图。")
        
        plt.close(fig_subplots)
    elif not any(total_transitions_processed_by_stage.values()):
        print("\n信息: 所有阶段均未找到转换数据，跳过重点情绪马尔科夫子图生成。")

# --- 新增：二阶马尔可夫链分析函数 ---
def analyze_emotion_transitions_markov2(file_path, base_output_dir, input_filename_without_ext):
    """
    Analyzes emotion transitions using a second-order Markov chain approach and saves heatmaps.
    Considers transitions of the form: (S_t-2, S_t-1) -> S_t
    """
    if not visualization_available:
        print("警告 (二阶马尔可夫): 可视化库 (matplotlib, seaborn, pandas) 不可用。跳过二阶马尔可夫链转换分析。")
        return

    all_codes = get_all_16_emotion_codes()
    
    # Create all possible previous state pairs (S_t-2, S_t-1)
    prev_state_pairs_tuples = [(c1, c2) for c1 in all_codes for c2 in all_codes]
    prev_state_pairs_multi_index = pd.MultiIndex.from_tuples(prev_state_pairs_tuples, names=['S_t_minus_2', 'S_t_minus_1'])

    transition_counts_by_stage = {
        stage: pd.DataFrame(0, index=prev_state_pairs_multi_index, columns=all_codes, dtype=float)
        for stage in range(6) # 0-5 for sleep stages
    }
    total_transitions_processed_by_stage = defaultdict(int)
    
    sleep_stage_map = {
        0: "W (Wakefulness)", 1: "N1 (Light Sleep)", 2: "N2 (Intermediate Sleep)",
        3: "N3 (Deep Sleep)", 4: "N4 (Very Deep Sleep)", 5: "REM (Rapid Eye Movement)"
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误 (二阶马尔可夫): 文件未找到于 '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误 (二阶马尔可夫): 无法从 '{file_path}' 解码JSON。请检查文件格式。")
        return
    except Exception as e:
        print(f"错误 (二阶马尔可夫): 读取文件时发生意外错误: {e}")
        return

    total_transitions_processed = 0
    for i, sample in enumerate(tqdm(data, desc="正在分析二阶马尔可夫转换", ncols=100)):
        input_str = sample.get("input", "")
        sequence = parse_emotion_sequence_string(input_str)
        
        sleep_stage_str = sample.get("output")
        current_stage_code = -1
        if sleep_stage_str is not None:
            try:
                current_stage_code = int(sleep_stage_str)
                if not (0 <= current_stage_code <= 5):
                    current_stage_code = -1 # Invalid stage
            except ValueError:
                current_stage_code = -1 # Invalid stage
        
        if current_stage_code == -1:
            continue # Skip if stage is invalid or missing

        if sequence and len(sequence) >= 3:
            for k in range(len(sequence) - 2):
                s_t_minus_2 = sequence[k]
                s_t_minus_1 = sequence[k+1]
                s_t = sequence[k+2]
                
                if re.fullmatch(r"[01]{4}", s_t_minus_2) and \
                   re.fullmatch(r"[01]{4}", s_t_minus_1) and \
                   re.fullmatch(r"[01]{4}", s_t):
                    try:
                        transition_counts_by_stage[current_stage_code].loc[(s_t_minus_2, s_t_minus_1), s_t] += 1
                        total_transitions_processed_by_stage[current_stage_code] += 1
                        total_transitions_processed += 1
                    except KeyError:
                        # This might happen if a code parsed from sequence is not in all_codes, though unlikely with validation
                        print(f"警告 (二阶马尔可夫 阶段 {current_stage_code} 样本 {i}): 状态对 ('{s_t_minus_2}', '{s_t_minus_1}') 或状态 '{s_t}' 查找失败。已跳过。")
                else:
                    pass # Invalid code format in sequence, already handled by parser or should be
        elif sequence and len(sequence) < 3:
            pass 

    if total_transitions_processed == 0:
        print("信息 (二阶马尔可夫): 未找到有效的情绪序列 (长度>=3) 或在所有阶段均未处理任何转换以进行二阶马尔可夫链分析。")
        return

    print("\n--- 按睡眠阶段进行的二阶马尔可夫链转换分析 (完整 256x16 热力图) ---")
    
    all_stage_full_2nd_order_probabilities = {} 

    for stage_code_loop in range(6):
        stage_name_for_plot = sleep_stage_map.get(stage_code_loop, f"Unknown Stage {stage_code_loop}")
        print(f"\n--- 阶段: {stage_name_for_plot} (代码: {stage_code_loop}) ---")

        current_stage_counts_df = transition_counts_by_stage[stage_code_loop]
        num_transitions_in_stage = total_transitions_processed_by_stage[stage_code_loop]

        if num_transitions_in_stage == 0:
            print(f"  信息: 阶段 {stage_name_for_plot} 没有记录到二阶转换。")
            all_stage_full_2nd_order_probabilities[stage_code_loop] = pd.DataFrame(0, index=prev_state_pairs_multi_index, columns=all_codes, dtype=float)
            continue

        # Sum counts for each (S_t-2, S_t-1) pair to get the denominator for probability
        pair_sums = current_stage_counts_df.sum(axis=1) 
        # Calculate probabilities P(S_t | S_t-2, S_t-1)
        transition_probabilities_df = current_stage_counts_df.div(pair_sums.replace(0, 1), axis=0).fillna(0)
        all_stage_full_2nd_order_probabilities[stage_code_loop] = transition_probabilities_df

        print(f"  转换概率 (来自每个状态对的前5个转换，基于此阶段的 {num_transitions_in_stage} 次转换):")
        for pair_idx in transition_probabilities_df.index:
            if pair_sums[pair_idx] > 0: 
                top_transitions = transition_probabilities_df.loc[pair_idx].sort_values(ascending=False).head(5)
                # Only print if there's at least one non-zero probability transition from this pair
                if top_transitions.iloc[0] > 0:
                    print(f"    来自 {pair_idx}:")
                    for to_state, prob in top_transitions.items():
                        if prob > 0: 
                            count_val = int(current_stage_counts_df.loc[pair_idx, to_state])
                            print(f"      -> {to_state}: {prob:.3f} (计数: {count_val})")
        
        # Plotting the full 256x16 heatmap for this stage
        fig_markov2_full, ax_markov2_full = plt.subplots(figsize=(18, 24)) # Adjusted for potentially many rows
        sns.heatmap(transition_probabilities_df, annot=False, fmt=".2f", cmap="viridis", linewidths=0,
                    cbar_kws={'label': 'Transition Probability P(S_t | S_t-2, S_t-1)'}, ax=ax_markov2_full) # Annot off for dense plot
        
        ax_markov2_full.set_title(f'2nd Order Emotion Transition Probability Matrix (Full 256x16)\nStage: {stage_name_for_plot} - Source: {os.path.basename(file_path)}', fontsize=16)
        ax_markov2_full.set_xlabel('To Emotion Code (S_t)', fontsize=12)
        ax_markov2_full.set_ylabel('From Emotion Code Pair (S_t-2, S_t-1)', fontsize=12)
        ax_markov2_full.tick_params(axis='x', rotation=45, labelsize=8) 
        ax_markov2_full.tick_params(axis='y', rotation=0, labelsize=max(4, 10 - len(all_codes)//60)) # Dynamic y-tick size

        # Reduce y-tick label frequency if too many rows to avoid overlap
        num_rows = len(transition_probabilities_df.index)
        if num_rows > 50: # Arbitrary threshold
            step = num_rows // 50
            ax_markov2_full.set_yticks(ax_markov2_full.get_yticks()[::step])

        output_filename_full = os.path.join(base_output_dir, f"markov2_full_stage_{stage_code_loop}_{input_filename_without_ext}.png")
        try:
            plt.savefig(output_filename_full, dpi=300, bbox_inches='tight')
            print(f"  阶段 {stage_name_for_plot} 的二阶马尔可夫链完整热力图已保存至: {output_filename_full}")
        except Exception as e:
            print(f"  保存阶段 {stage_name_for_plot} 的二阶马尔可夫链完整热力图时出错: {e}")
            import traceback; print(traceback.format_exc())
        plt.close(fig_markov2_full)

    # --- Plotting Subplots for Highlighted Emotions (Second Order) ---
    if visualization_available and any(total_transitions_processed_by_stage.values()):
        print("\n--- 正在为重点情绪生成二阶马尔可夫链子图 ---")
        
        focused_codes = HIGHLIGHTED_EMOTION_CODES
        # Create (Prev1, Prev2) pairs from focused codes
        focused_prev_pairs_tuples = [(c1, c2) for c1 in focused_codes for c2 in focused_codes]
        # Ensure the MultiIndex is correctly created for reindexing
        focused_prev_pairs_multi_index = pd.MultiIndex.from_tuples(focused_prev_pairs_tuples, names=['S_t_minus_2', 'S_t_minus_1'])

        fig_subplots_m2, axes_subplots_m2 = plt.subplots(2, 3, figsize=(24, 15)) # Wider for (pair) labels
        axes_subplots_m2 = axes_subplots_m2.flatten()
        plot_created_for_subplots_m2 = False

        for i, stage_code_loop_subplot in enumerate(range(6)):
            ax = axes_subplots_m2[i]
            stage_name_for_plot_subplot = sleep_stage_map.get(stage_code_loop_subplot, f"Stage {stage_code_loop_subplot}")
            
            full_probs_this_stage_m2 = all_stage_full_2nd_order_probabilities.get(stage_code_loop_subplot)

            if full_probs_this_stage_m2 is not None and not full_probs_this_stage_m2.empty and total_transitions_processed_by_stage[stage_code_loop_subplot] > 0:
                # Reindex with the MultiIndex of focused pairs, then select focused columns
                sub_matrix_probs_m2 = full_probs_this_stage_m2.reindex(index=focused_prev_pairs_multi_index, columns=focused_codes).fillna(0.0)
                
                # Ensure y-tick labels are string representations of the tuples
                y_tick_labels_m2 = [str(pair) for pair in sub_matrix_probs_m2.index.to_list()]

                sns.heatmap(sub_matrix_probs_m2, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5,
                            cbar=False, ax=ax, square=False, annot_kws={"size": 8}, 
                            yticklabels=y_tick_labels_m2) # Use custom y_tick_labels
                
                ax.set_title(f"Stage: {stage_name_for_plot_subplot}\n({len(focused_codes)}^2 x {len(focused_codes)} Focused)", fontsize=10)
                ax.set_xlabel("To Emotion (S_t)", fontsize=9)
                ax.set_ylabel("From Pair (S_t-2, S_t-1)", fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.tick_params(axis='y', rotation=0, labelsize=8)
                plot_created_for_subplots_m2 = True
            else:
                ax.text(0.5, 0.5, "无转换数据", ha='center', va='center')
                ax.set_title(f"阶段: {stage_name_for_plot_subplot}\n(无二阶转换)", fontsize=10)
                ax.axis('off')

        if plot_created_for_subplots_m2:
            fig_subplots_m2.suptitle(f'2nd Order Markov Transitions for Highlighted Emotions by Sleep Stage\nSource: {os.path.basename(file_path)}', fontsize=16, y=0.99)
            fig_subplots_m2.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust rect for suptitle
            
            subplot_output_filename_m2 = os.path.join(base_output_dir, f"markov2_highlighted_subplots_{input_filename_without_ext}.png")
            try:
                plt.savefig(subplot_output_filename_m2, dpi=300)
                print(f"\n重点情绪二阶马尔可夫子图已保存至: {subplot_output_filename_m2}")
            except Exception as e:
                print(f"\n保存重点情绪二阶马尔可夫子图时出错: {e}")
                import traceback; print(traceback.format_exc())
        else:
            print("\n信息: 无数据可生成重点情绪二阶马尔可夫子图。")
        
        plt.close(fig_subplots_m2)
    elif not any(total_transitions_processed_by_stage.values()):
        print("\n信息 (二阶马尔可夫): 所有阶段均未找到转换数据，跳过重点情绪二阶马尔可夫子图生成。")
# --- 结束新增 ---


if __name__ == "__main__":
    # --- 新增：命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Analyze emotion data from JSON files.")
    parser.add_argument(
        "--input_file", "-i",
        default="/data/lhc/datasets_new/emotion/train/sleep_st_44_100hz_eeg7.5s-step7.5s_emo2.0s-step0.25s_win_all_tokenizer_qwen_tok9689_bal0.5_sqrt_inverse_202505130318_train.json",
        help="Path to the input JSON file. Defaults to a predefined training data path."
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save output files. If not provided, defaults to './output/<input_filename_base>/' relative to the script's directory."
    )
    
    parser.add_argument("--analyze_ratios", action=argparse.BooleanOptionalAction, default=False, help="Perform emotion ratio analysis.")
    parser.add_argument("--analyze_markov1", action=argparse.BooleanOptionalAction, default=False, help="Perform 1st order Markov chain analysis.")
    parser.add_argument("--analyze_markov2", action=argparse.BooleanOptionalAction, default=True, help="Perform 2nd order Markov chain analysis.")
    
    args = parser.parse_args()

    source_json_file = args.input_file
    # --- 结束新增 ---
    
    # Get the input filename without extension (used for naming output files and the new subfolder)
    input_filename_without_ext = os.path.splitext(os.path.basename(source_json_file))[0]

    # Setup base output directory structure
    if args.output_dir:
        base_output_dir_for_this_run = args.output_dir
    else:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        general_output_dir = os.path.join(current_script_dir, "output")
        base_output_dir_for_this_run = os.path.join(general_output_dir, input_filename_without_ext)
    
    # Ensure the specific output subfolder exists for this run
    os.makedirs(base_output_dir_for_this_run, exist_ok=True)
    args.output_dir = base_output_dir_for_this_run # Store the actual output dir back to args for final message
    
    # --- 修改：根据参数选择性执行分析 ---
    if args.analyze_ratios:
        print("\n--- 开始情绪占比分析 ---")
        calculate_emotion_ratios_per_stage(source_json_file, base_output_dir_for_this_run, input_filename_without_ext)
    else:
        print("\n--- 跳过情绪占比分析 ---")
    
    if args.analyze_markov1:
        print("\n--- 开始一阶马尔可夫链分析 ---")
        analyze_emotion_transitions_markov(source_json_file, base_output_dir_for_this_run, input_filename_without_ext)
    else:
        print("\n--- 跳过一阶马尔可夫链分析 ---")
        
    if args.analyze_markov2:
        print("\n--- 开始二阶马尔可夫链分析 ---")
        analyze_emotion_transitions_markov2(source_json_file, base_output_dir_for_this_run, input_filename_without_ext)
    else:
        print("\n--- 跳过二阶马尔可夫链分析 ---")
    # --- 结束修改 ---

    print("\n--- 所有请求的分析完成 ---") # 修改完成提示
    print(f"所有输出文件已保存至目录: {args.output_dir}")