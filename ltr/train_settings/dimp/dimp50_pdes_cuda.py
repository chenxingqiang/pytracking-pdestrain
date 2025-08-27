from ltr.train_settings.dimp.dimp50_pdes import run as run_mac

def run(settings):
    """CUDA兼容的DiMP50训练配置，基于Mac M3配置优化"""
    
    # 调用Mac配置作为基础
    run_mac(settings)
    
    # CUDA环境特定优化
    settings.num_workers = 4  # CUDA环境可以使用多进程
    settings.multi_gpu = True  # 启用多GPU支持
    
    # CUDA特定的批次大小优化
    if hasattr(settings, 'batch_size'):
        settings.batch_size = 8  # DiMP50需要更多内存，批次大小适中
    
    print("🚀 CUDA兼容训练配置已启用")
    print(f"   多进程工作器: {settings.num_workers}")
    print(f"   多GPU支持: {settings.multi_gpu}")
    print(f"   批次大小: {settings.batch_size}")
