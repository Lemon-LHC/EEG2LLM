#!/bin/bash
# 直接测试最新检查点的性能

# 默认参数
CHECKPOINT_DIR="/data/lhc/saves/Llama-3.2-1B-Instruct/lora/edf200_100hz_10000ms_tok8363_train"
TEST_DATASET="edf200_100hz_10000ms_tok8363_test"
BATCH_SIZE=8

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -c, --checkpoint  检查点目录 (默认: $CHECKPOINT_DIR)"
    echo "  -t, --test        测试数据集名称 (默认: $TEST_DATASET)"
    echo "  -b, --batch       批处理大小 (默认: $BATCH_SIZE)"
    echo "  -h, --help        显示此帮助信息"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--checkpoint)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -t|--test)
            TEST_DATASET="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "未知选项: $key"
            show_help
            ;;
    esac
done

# 确保检查点目录存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "错误: 检查点目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "================================================================================"
echo "开始测试最新检查点"
echo "  • 检查点目录: $CHECKPOINT_DIR"
echo "  • 测试数据集: $TEST_DATASET"
echo "  • 批处理大小: $BATCH_SIZE"
echo "================================================================================"

# 运行测试脚本
python projects/fine/test_latest.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --test_dataset "$TEST_DATASET" \
    --batch_size "$BATCH_SIZE" \
    --half_precision \
    --verbose

# 检查返回状态
if [ $? -eq 0 ]; then
    echo "✓ 测试完成，结果已保存到TensorBoard"
else
    echo "✗ 测试失败，请检查日志"
fi 