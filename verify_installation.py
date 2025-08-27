#!/usr/bin/env python3
"""
Mac M3 PyTracking-PDESTrain 安装验证脚本
验证所有组件是否正确安装和配置
"""

import sys
import os
import subprocess

def check_pip_config():
    """检查pip配置"""
    print("🔍 检查pip配置...")
    
    pip_config_path = os.path.expanduser("~/.config/pip/pip.conf")
    if os.path.exists(pip_config_path):
        with open(pip_config_path, 'r') as f:
            content = f.read()
            if 'user = true' in content and 'break-system-packages = true' in content:
                print("   ✅ pip.conf配置正确")
                return True
            else:
                print("   ⚠️  pip.conf存在但配置不完整")
                return False
    else:
        print("   ❌ pip.conf不存在")
        return False

def check_pytorch_mps():
    """检查PyTorch MPS支持"""
    print("🔍 检查PyTorch MPS支持...")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   ✅ MPS可用")
            
            # 测试MPS基本操作
            try:
                device = torch.device('mps')
                x = torch.randn(5, 5).to(device)
                y = torch.randn(5, 5).to(device)
                z = torch.mm(x, y)
                print("   ✅ MPS运算测试通过")
                return True
            except Exception as e:
                print(f"   ❌ MPS运算测试失败: {e}")
                return False
        else:
            print("   ❌ MPS不可用")
            return False
            
    except ImportError as e:
        print(f"   ❌ PyTorch导入失败: {e}")
        return False

def check_core_dependencies():
    """检查核心依赖"""
    print("🔍 检查核心依赖...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'TQDM'),
        ('PIL', 'Pillow'),
    ]
    
    results = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
            results.append(True)
        except ImportError:
            print(f"   ❌ {name}")
            results.append(False)
    
    return all(results)

def check_optional_dependencies():
    """检查可选依赖"""
    print("🔍 检查可选依赖...")
    
    optional_deps = [
        ('visdom', 'Visdom'),
        ('skimage', 'Scikit-Image'),
        ('tensorboard', 'TensorBoard'),
        ('pycocotools', 'COCO Tools'),
        ('lvis', 'LVIS'),
        ('jpeg4py', 'JPEG4Py'),
        ('scipy', 'SciPy'),
    ]
    
    available = 0
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"   ✅ {name}")
            available += 1
        except ImportError:
            print(f"   ⚠️  {name} (可选)")
    
    print(f"   可选依赖: {available}/{len(optional_deps)} 可用")
    return available >= len(optional_deps) * 0.7  # 70%可用即可

def check_dataset_loader():
    """检查数据集加载器"""
    print("🔍 检查数据集加载器...")
    
    try:
        from ltr.dataset.pdesdataset import PdesDataset
        print("   ✅ PdesDataset导入成功")
        
        # 检查类的基本属性
        print(f"   数据集类名: {PdesDataset.__name__}")
        print(f"   基类: {[base.__name__ for base in PdesDataset.__bases__]}")
        return True
        
    except ImportError as e:
        print(f"   ❌ PdesDataset导入失败: {e}")
        return False

def check_environment_config():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    try:
        from ltr.admin.environment import env_settings
        print("   ✅ 环境设置模块可用")
        
        try:
            env = env_settings()
            print("   ✅ 环境配置加载成功")
            return True
        except RuntimeError as e:
            if "local.py" in str(e):
                print("   ⚠️  local.py未配置（正常，需要手动设置）")
                return True
            else:
                print(f"   ❌ 环境配置错误: {e}")
                return False
                
    except ImportError as e:
        print(f"   ❌ 环境设置导入失败: {e}")
        return False

def check_conda_environment():
    """检查conda环境"""
    print("🔍 检查conda环境...")
    
    try:
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        envs = result.stdout
        
        if 'pytracking' in envs:
            print("   ✅ 找到pytracking相关的conda环境")
            return True
        else:
            print("   ⚠️  未找到pytracking相关的conda环境")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  无法检查conda环境")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("=" * 60)
    print("🚀 Mac M3 PyTracking-PDESTrain 安装验证")
    print("=" * 60)
    
    tests = [
        ("Conda环境", check_conda_environment),
        ("pip配置", check_pip_config),
        ("PyTorch MPS", check_pytorch_mps),
        ("核心依赖", check_core_dependencies),
        ("可选依赖", check_optional_dependencies),
        ("数据集加载器", check_dataset_loader),
        ("环境配置", check_environment_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            results.append((test_name, False))
        print()
    
    # 结果汇总
    print("=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    # 给出建议
    if passed == len(results):
        print("\n🎉 恭喜！所有测试通过，安装完成！")
        print("\n📋 接下来的步骤:")
        print("1. 配置数据集路径在 ltr/admin/local.py")
        print("2. 运行 python example_train_pdes.py 开始训练")
        print("3. 查看 MAC_M3_SETUP_GUIDE.md 了解详细使用方法")
    elif passed >= len(results) * 0.8:
        print(f"\n✅ 大部分测试通过 ({passed}/{len(results)})，基本可用")
        print("\n⚠️  建议:")
        failed_tests = [name for name, result in results if not result]
        for test in failed_tests:
            if test == "pip配置":
                print("- 运行: bash install_mac.sh 环境名 来自动配置pip")
            elif test == "可选依赖":
                print("- 运行: pip install visdom scikit-image tensorboard 安装可选依赖")
    else:
        print(f"\n❌ 多项测试失败 ({len(results) - passed}/{len(results)})，需要检查安装")
        print("\n🔧 建议:")
        print("1. 重新运行安装脚本: bash install_mac.sh pytracking-pdes")
        print("2. 检查pip配置: cat ~/.config/pip/pip.conf")
        print("3. 查看详细错误信息并参考故障排除指南")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
