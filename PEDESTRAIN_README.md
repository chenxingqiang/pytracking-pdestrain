# Pedestrain数据集训练指南

本文档介绍如何使用pedestrain数据集训练DiMP18和DiMP50目标跟踪模型。

## 1. 数据集配置

首先，确保在`ltr/admin/local.py`文件中添加pedestrain数据集的路径配置：

```python
self.pedestrain_dir = '[YOUR_DATASET_PATH]/pedestrain'
```

pedestrain目录结构应类似于：
```
pedestrain/
  - Airport_ce/
    - img/
    - groundtruth_rect.txt
  - BlurBody/
    - img/
    - groundtruth_rect.txt
  - Busstation_ce1/
    ...
  (其他行人跟踪序列)
```

## 2. 训练DiMP18模型

要训练DiMP18模型，运行以下命令：

```bash
python ltr/run_training.py dimp dimp18_pedestrain
```

## 3. 训练DiMP50模型

要训练DiMP50模型，运行以下命令：

```bash
python ltr/run_training.py dimp dimp50_pedestrain
```

## 4. 训练配置说明

### DiMP18 Pedestrain训练配置

DiMP18的训练配置使用ResNet18作为主干网络，主要参数包括：
- 批量大小：26
- 搜索区域因子：5.0
- 输出sigma因子：1/4
- 优化器：Adam，学习率为2e-4
- 学习率调度：每15个周期衰减至0.2倍
- 训练周期：50

### DiMP50 Pedestrain训练配置

DiMP50的训练配置使用ResNet50作为主干网络，主要参数包括：
- 批量大小：10（由于模型更大）
- 搜索区域因子：5.0
- 输出sigma因子：1/4
- 优化器：Adam，学习率为2e-4
- 学习率调度：每15个周期衰减至0.2倍
- 训练周期：50

## 5. 数据集划分

在训练过程中，pedestrain数据集会被自动划分为训练集（80%）和验证集（20%）。

## 6. 训练结果

训练日志和模型权重文件将保存在`ltr/checkpoints/`目录下，具体路径为：

- DiMP18 Pedestrain: `ltr/checkpoints/dimp/dimp18_pedestrain/`
- DiMP50 Pedestrain: `ltr/checkpoints/dimp/dimp50_pedestrain/`

## 7. 使用训练好的模型进行评估

训练完成后，可以使用以下命令来评估模型性能（需要先在pytracking中配置相应参数）：

```bash
cd pytracking
python run_tracker.py dimp dimp18_pedestrain --dataset pedestrain
python run_tracker.py dimp dimp50_pedestrain --dataset pedestrain
```

注意：需要在pytracking/parameter/目录下添加相应的参数文件，并配置模型权重路径。

## 8. 注意事项

- 训练时间会根据GPU配置和数据集大小而变化
- 建议使用较大的GPU内存进行DiMP50的训练（建议至少8GB GPU内存）
- 若使用多GPU训练，请修改配置文件中的`settings.multi_gpu = True`

## 9. 自定义训练参数

如需自定义训练参数，可以修改以下文件：
- `ltr/train_settings/dimp/dimp18_pedestrain.py`
- `ltr/train_settings/dimp/dimp50_pedestrain.py`

主要可调整的参数包括批量大小、学习率、训练周期等。
