#!/bin/bash
# PINN训练监控脚本

echo "=========================================="
echo "PINN训练状态监控"
echo "=========================================="
echo ""

# 检查进程
echo "1. 检查训练进程:"
if ps aux | grep -q "[t]est_pinn_training.py"; then
    echo "   ✅ 训练正在运行"
    ps aux | grep "[t]est_pinn_training.py" | awk '{print "   进程ID:", $2, "CPU:", $3"%", "内存:", $4"%", "运行时间:", $10}'
else
    echo "   ⏹️  训练未运行"
fi
echo ""

# 检查日志
echo "2. 最新训练日志 (最后10行):"
if [ -f outputs/pinn_train.log ]; then
    tail -10 outputs/pinn_train.log | grep -E "Training:|loss=" | tail -3
else
    echo "   ❌ 日志文件不存在"
fi
echo ""

# 检查输出文件
echo "3. 输出文件:"
if [ -d outputs/pinn ]; then
    ls -lh outputs/pinn/ | grep -v "^total" | awk '{print "   ", $9, "-", $5}'
else
    echo "   ❌ 输出目录不存在"
fi
echo ""

# 检查模型
echo "4. 模型状态:"
if [ -f outputs/pinn/pinn_model.pt ]; then
    echo "   ✅ 模型已保存"
    python3 << 'EOF'
import torch
try:
    ckpt = torch.load('outputs/pinn/pinn_model.pt', map_location='cpu')
    print(f"   训练步数: {len(ckpt['step_history'])}")
    print(f"   最终Loss: {ckpt['loss_history'][-1]:.6e}")
except Exception as e:
    print(f"   ❌ 无法读取模型: {e}")
EOF
else
    echo "   ⏳ 模型尚未保存"
fi
echo ""

# 检查图片
echo "5. 生成的图片:"
if ls outputs/pinn/*.png 1> /dev/null 2>&1; then
    ls outputs/pinn/*.png | while read f; do
        echo "   ✅ $(basename $f)"
    done
else
    echo "   ⏳ 尚未生成图片"
fi
echo ""

echo "=========================================="
echo "快速命令:"
echo "  查看实时日志: tail -f outputs/pinn_train.log"
echo "  查看结果: python test/view_pinn_results.py"
echo "  测试预测: python test/test_pinn_prediction.py"
echo "=========================================="
