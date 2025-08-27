#!/bin/bash

# Mac M3 专用安装脚本
# 用于安装pytracking-pdestrain项目的依赖

set -e  # 遇到错误时立即退出

echo "****************** Mac M3 PyTracking-PDESTrain 安装脚本 ******************"

# 检查是否在Mac系统上运行
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "错误: 此脚本仅适用于 macOS 系统"
    exit 1
fi

# 检查是否为Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "警告: 此脚本针对Apple Silicon (M1/M2/M3) 优化"
    echo "您的系统架构: $(uname -m)"
fi

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "用法: bash install_mac.sh environment_name"
    echo "示例: bash install_mac.sh pytracking"
    exit 1
fi

conda_env_name=$1

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda命令"
    echo "请确保已安装Anaconda或Miniconda并添加到PATH中"
    exit 1
fi

# 初始化conda（如果需要）
eval "$(conda shell.bash hook)" 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

echo ""
echo "****************** 配置pip设置 ******************"
# 创建pip配置目录和文件，解决Mac系统的外部管理环境问题
mkdir -p ~/.config/pip
echo "[install]
user = true
break-system-packages = true" > ~/.config/pip/pip.conf
echo "✅ 已配置pip.conf以支持Mac系统包管理"

echo ""
echo "****************** 创建conda环境 ${conda_env_name} ******************"
conda create -y --name $conda_env_name python=3.9

echo ""
echo "****************** 激活conda环境 ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo "****************** 安装Mac M3优化的PyTorch ******************"
# 使用conda安装PyTorch，这会自动选择Mac M3优化版本
conda install -y pytorch torchvision torchaudio -c pytorch

echo ""
echo "****************** 安装基础科学计算包 ******************"
conda install -y numpy scipy matplotlib pandas

echo ""
echo "****************** 安装OpenCV ******************"
# 使用conda-forge的OpenCV，对Mac M3更友好
conda install -y -c conda-forge opencv

echo ""
echo "****************** 安装其他依赖 ******************"
pip install tqdm
pip install scikit-image
pip install tensorboard
pip install visdom
pip install Pillow

echo ""
echo "****************** 安装Cython ******************"
conda install -y cython

echo ""
echo "****************** 安装COCO工具包 ******************"
pip install pycocotools

echo ""
echo "****************** 安装LVIS工具包 ******************"
pip install lvis

echo ""
echo "****************** 尝试安装jpeg4py ******************"
# jpeg4py可能在Mac M3上有问题，所以使用try-catch方式
pip install jpeg4py || echo "警告: jpeg4py安装失败，将使用OpenCV作为替代"

echo ""
echo "****************** 安装其他可选依赖 ******************"
pip install tikzplotlib || echo "tikzplotlib安装失败，跳过"
pip install gdown || echo "gdown安装失败，跳过"

echo ""
echo "****************** 检查是否需要安装spatial-correlation-sampler ******************"
echo "注意: spatial-correlation-sampler 仅用于KYS跟踪器"
pip install spatial-correlation-sampler || echo "spatial-correlation-sampler安装失败，如果不使用KYS跟踪器可以忽略"

echo ""
echo "****************** 创建网络目录 ******************"
mkdir -p pytracking/networks

echo ""
echo "****************** 下载DiMP50网络权重 ******************"
if command -v gdown >/dev/null 2>&1; then
    gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth || echo "网络权重下载失败，请手动下载"
else
    echo "gdown未安装，请手动下载网络权重到 pytracking/networks/"
fi

echo ""
echo "****************** 设置环境配置 ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()" || echo "pytracking环境配置创建失败"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()" || echo "ltr环境配置创建失败"

echo ""
echo "****************** 验证PyTorch MPS支持 ******************"
python -c "
import torch
print('PyTorch版本:', torch.__version__)
if torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) 可用')
    print('MPS设备数量:', torch.mps.device_count() if hasattr(torch.mps, 'device_count') else 'N/A')
    
    # 测试MPS基本功能
    try:
        device = torch.device('mps')
        x = torch.randn(5, 5).to(device)
        y = torch.randn(5, 5).to(device)
        z = x @ y
        print('✅ MPS基本运算测试通过')
    except Exception as e:
        print('❌ MPS测试失败:', e)
else:
    print('❌ MPS不可用，将使用CPU')

print('CUDA可用:', torch.cuda.is_available())
print('CPU核心数:', torch.get_num_threads())
"

echo ""
echo "****************** 安装完成! ******************"
echo ""
echo "接下来的步骤:"
echo "1. 配置数据集路径在 ltr/admin/local.py 中"
echo "2. 设置 pedestrain_dir 指向您的行人数据集"
echo "3. 使用以下命令开始训练:"
echo "   conda activate $conda_env_name"
echo "   python example_train_pdes.py  # 使用新的PdesDataset"
echo "   # 或者创建自定义训练配置使用 PdesDataset"
echo ""
echo "注意事项:"
echo "- 此安装已针对Mac M3优化"
echo "- 支持MPS (Metal Performance Shaders) 加速"
echo "- 如遇到问题，请检查依赖版本兼容性"
echo ""
echo "更多网络权重可从以下地址下载:"
echo "https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O"
echo "或访问模型库: https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md"
