#!/bin/bash
# CUDA兼容的安装脚本
# 基于Mac M3安装脚本，增加了CUDA环境支持

set -e

echo "============================================================"
echo "🚀 CUDA兼容的pytracking-pdestrain安装脚本"
echo "============================================================"

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <conda环境名称>"
    echo "示例: $0 pytracking-cuda"
    exit 1
fi

conda_env_name=$1

echo "📋 安装参数:"
echo "   Conda环境: ${conda_env_name}"
echo "   系统架构: $(uname -m)"
echo "   操作系统: $(uname -s)"
echo ""

# 检查conda是否可用
if command -v conda &> /dev/null; then
    echo "✅ 检测到conda: $(conda --version)"
    conda_path=$(command -v conda)
    echo "   Conda路径: ${conda_path}"
else
    echo "❌ 未检测到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 初始化conda
echo ""
echo "****************** 初始化conda环境 ******************"
eval "$(conda shell.bash hook)"
conda info --envs

echo ""
echo "****************** 创建conda环境 ${conda_env_name} ******************"
if conda env list | grep -q "^${conda_env_name} "; then
    echo "⚠️  环境 ${conda_env_name} 已存在，将重新创建"
    conda env remove -n ${conda_env_name} -y
fi

# 创建新的conda环境
conda create -n ${conda_env_name} python=3.9 -y
echo "✅ 已创建conda环境 ${conda_env_name}"

echo ""
echo "****************** 激活环境并安装依赖 ******************"
conda activate ${conda_env_name}

# 安装PyTorch CUDA版本
echo "📦 安装PyTorch CUDA版本..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他核心依赖
echo "📦 安装其他核心依赖..."
conda install -c conda-forge opencv pillow numpy scipy matplotlib tqdm -y

# 安装可选依赖
echo "📦 安装可选依赖..."
conda install -c conda-forge tensorboard visdom -y

# 安装timm
echo "📦 安装timm..."
pip install timm

# 安装其他Python包
echo "📦 安装其他Python包..."
pip install -r requirements.txt

echo ""
echo "****************** 验证安装 ******************"
echo "🔍 检查PyTorch CUDA支持..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA不可用')
"

echo ""
echo "****************** 安装完成 ******************"
echo "✅ 环境 ${conda_env_name} 安装完成！"
echo ""
echo "📋 使用方法:"
echo "1. 激活环境: conda activate ${conda_env_name}"
echo "2. 运行CUDA训练: python train_dimp_pdes_cuda.py --model dimp18"
echo "3. 检查GPU状态: nvidia-smi"
echo ""
echo "🔧 故障排除:"
echo "- 如果CUDA不可用，请检查NVIDIA驱动安装"
echo "- 确保PyTorch版本与CUDA版本兼容"
echo "- 检查GPU内存是否足够"
echo ""
echo "🎉 安装完成！"
