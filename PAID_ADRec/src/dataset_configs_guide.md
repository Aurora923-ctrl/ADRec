# ADRec数据集配置指南

本文档提供了针对不同数据集的推荐参数配置。

## 数据集规模参考

| 数据集 | 物品数量 | 规模分类 | 推荐batch_size | 推荐hidden_size | 推荐diffusion_steps | 推荐lambda_uncertainty |
|--------|---------|---------|---------------|----------------|-------------------|---------------------|
| ml-100k | 1,008 | 小 | 512 | 128 | 32 | 0.01 |
| baby | 4,731 | 小 | 512 | 128 | 32 | 0.01 |
| beauty | 6,086 | 小-中 | 512 | 128 | 32 | 0.01 |
| toys | 7,309 | 中 | 512 | 128 | 32 | 0.01 |
| sports | 12,301 | 中-大 | 512-1024 | 128 | 32-64 | 0.01 |
| yelp | 64,669 | 大 | 1024 | 256 | 64 | 0.1 |

## 快速配置模板

### 小数据集配置（ml-100k, baby, beauty）

```yaml
dataset: baby  # 或 ml-100k, beauty
batch_size: 512
hidden_size: 128
diffusion_steps: 32
lambda_uncertainty: 0.01
pcgrad: true
```

### 中等数据集配置（toys, sports）

```yaml
dataset: toys  # 或 sports
batch_size: 512  # sports可尝试1024（显存充足时）
hidden_size: 128
diffusion_steps: 32  # sports可尝试64
lambda_uncertainty: 0.01
pcgrad: true
```

### 大数据集配置（yelp）

```yaml
dataset: yelp
batch_size: 1024  # 显存不足可降至512或256
hidden_size: 256  # 显存不足可保持128
diffusion_steps: 64  # 可选，提升性能但增加训练时间
lambda_uncertainty: 0.1  # 大数据集可尝试更大值
pcgrad: true
```

## 关键参数调整说明

### 1. lambda_uncertainty（不确定性权重）

**重要性**: ⭐⭐⭐⭐⭐ (必须调整)

- **原始值**: 0.001（可能过小）
- **推荐值**: 
  - 小-中数据集: 0.01
  - 大数据集: 0.1
- **原因**: 改进文档指出原始值可能过小，影响信息融合效果

### 2. pcgrad（PCGrad梯度处理）

**重要性**: ⭐⭐⭐⭐ (强烈建议)

- **原始值**: false
- **推荐值**: true
- **原因**: 改进版使用true，可能提升多任务训练稳定性

### 3. batch_size（批次大小）

**重要性**: ⭐⭐⭐ (根据显存调整)

- **小数据集**: 512
- **中等数据集**: 512（sports可尝试1024）
- **大数据集**: 1024（显存不足可降至512或256）
- **注意**: 显存不足时优先降低batch_size

### 4. diffusion_steps（扩散步数）

**重要性**: ⭐⭐ (可选优化)

- **标准配置**: 32
- **大数据集优化**: 64（提升性能但增加训练时间）
- **注意**: 步数增加会显著增加训练时间

### 5. hidden_size（隐藏层大小）

**重要性**: ⭐⭐ (可选优化)

- **标准配置**: 128
- **大数据集优化**: 256（显存充足时）
- **注意**: 增大hidden_size会显著增加显存占用

## 使用建议

1. **首次运行**: 使用当前config.yaml中的配置（已针对toys优化）
2. **切换数据集**: 修改`dataset`字段，并根据上表调整关键参数
3. **显存不足**: 优先降低`batch_size`，其次考虑降低`hidden_size`
4. **性能优化**: 大数据集可尝试增加`diffusion_steps`和`hidden_size`

## 实验建议

### 优先级1：必须调整
- ✅ `lambda_uncertainty`: 0.001 → 0.01（小-中数据集）或 0.1（大数据集）
- ✅ `pcgrad`: false → true

### 优先级2：根据显存调整
- ⚠️ `batch_size`: 根据显存情况调整
- ⚠️ `hidden_size`: 大数据集且显存充足时可增大

### 优先级3：可选优化
- 🔄 `diffusion_steps`: 大数据集可尝试增加到64
- 🔄 其他参数保持默认值即可

## 注意事项

1. **yelp数据集**: 规模最大，训练时间会显著增加，建议：
   - 使用更大的batch_size（如果显存允许）
   - 考虑使用更大的hidden_size
   - 可以尝试更多的diffusion_steps

2. **显存管理**: 如果遇到OOM（显存不足）错误：
   - 首先降低batch_size
   - 其次降低hidden_size
   - 最后考虑降低diffusion_steps

3. **训练时间**: 
   - 小数据集：训练较快
   - 中等数据集：训练时间适中
   - yelp数据集：训练时间较长（可能需要数小时）

4. **参数调优**: 建议先使用推荐配置运行，获得基线结果后再进行微调。

