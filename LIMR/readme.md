# 总体

在一个epoch中，对于一个问题prompt 会出现多次，给出多次答案。

然后给出的也是一个prompt，也就是我们对于问题的求解，获取的sequence

```
{prompt: [[epoch1], [epoch2], [epoch3], [epoch4]]}
```

# 疑惑

在文章中，作者使用了全新的方式来操作r来参与计算 r/N = r_new，然后实际过程中，可以使用自己的reward 参与计算。

### 论文中的公式
```
si = 1 - (Σ(rk_i - rk_avg)²) / (Σ(1 - rk_avg)²), i = 1, ..., N
```

这个公式计算的是一个稳定性指标(stability index)，而不是简单的准确率。它衡量了样本 i 在多个训练周期(epochs)中表现的一致性：

- `rk_i` 是样本 i 在第 k 轮的奖励
- `rk_avg` 是第 k 轮所有样本的平均奖励
- 分子部分计算样本 i 在各轮的奖励与平均奖励的偏差平方和
- 分母部分是一个归一化因子

### 与代码的差异

您是对的，这个公式与代码中的 `correct/total` 计算方式确实不同：

1. **计算对象不同**：
   - 代码计算的是每个prompt在单个epoch的准确率
   - 论文公式计算的是单个样本在多个epochs中的稳定性

2. **计算方法不同**：
   - 代码是直接计算正确样本比例
   - 论文公式涉及到样本在多轮训练中与平均表现的偏差

3. **目的不同**：
   - 代码计算准确率是为了评估模型在各个prompt上的表现
   - 论文公式计算稳定性是为了识别"稳定样本"和"不稳定样本"

### 正确的实现

如果您需要实现论文中的稳定性指标，代码需要重新设计：

```python
def calculate_stability(sample_rewards, epoch_avg_rewards):
    """
    计算样本的稳定性指标
    
    Args:
        sample_rewards: 样本在各轮的奖励列表 [r1_i, r2_i, ..., rK_i]
        epoch_avg_rewards: 各轮的平均奖励 [r1_avg, r2_avg, ..., rK_avg]
    
    Returns:
        稳定性指标 si
    """
    numerator = sum((r_i - r_avg)**2 for r_i, r_avg in zip(sample_rewards, epoch_avg_rewards))
    denominator = sum((1 - r_avg)**2 for r_avg in epoch_avg_rewards)
    
    if denominator == 0:
        return 1.0  # 避免除以零
    
    stability = 1 - (numerator / denominator)
    return stability
```



# Process_accuracy_sequence 代码疑问

## 向后填充逻辑

在 `process_accuracy_sequence` 函数中，使用了一种"向后填充"(backward fill)的技术来处理这些缺失值：

```python
for accuracy_sequence in prompt_accuracies.values():
    for i in range(len(accuracy_sequence) - 1):
        # backward fill 技术
        if accuracy_sequence[i] == -1 and accuracy_sequence[i + 1] != -1:
            accuracy_sequence[i] = accuracy_sequence[i + 1]
```

这段代码遍历每个提示的准确率序列，从前往后检查：
- 如果当前位置 `i` 的值是 `-1`（缺失值）
- 并且下一个位置 `i+1` 的值不是 `-1`
- （有有效数据）
- 那么就用下一个位置的有效值来填充当前位置

这种处理基于一个假设：如果某个训练周期没有记录准确率，可以假设它的表现与紧接着的下一个有记录的周期相当。

针对于的是同一个prompt

## 计算逻辑

```python
valid_sequence = [(prompt, sequence) for prompt, sequence in prompt_accuracies.items()
                  if -1 not in sequence[:max_epochs]]
```

### 代码解析

这是一个列表推导式（list comprehension），用于从 `prompt_accuracies` 字典中筛选出符合特定条件的提示及其准确率序列。

1. `prompt_accuracies.items()` - 遍历字典的所有键值对，每一对包含一个提示(prompt)和它对应的准确率序列(sequence)

2. `if -1 not in sequence[:max_epochs]` - 筛选条件：只保留在前 `max_epochs` 个周期中不含有 `-1` 值的序列
   - `sequence[:max_epochs]` 截取序列的前 `max_epochs` 个元素
   - `-1 not in ...` 检查这段序列中是否不包含值 `-1`

3. `[(prompt, sequence) for ... if ...]` - 对于符合条件的每一对，创建一个包含提示和其准确率序列的元组，并将所有这些元组收集到一个列表中

### 功能目的

这行代码的目的是：**筛选出那些在所有关注的训练周期中都有有效准确率数据的提示**。

具体来说：
- 即使经过了前面的向后填充处理，某些提示的准确率序列可能仍然包含 `-1` 值（表示缺失数据）
- 这些不完整的序列可能会影响后续的平均值计算和相似度分析
- 因此，代码只保留那些数据完整的序列进行后续分析

### 实际应用意义

在机器学习模型评估中，这种筛选非常重要：
- 确保分析基于完整、连续的数据
- 避免缺失值引入偏差
- 使得不同提示之间的比较更加公平

如果一个提示在某些训练周期完全没有样本或无法计算准确率（即使经过填充），那么它的性能曲线可能不够可靠，最好从整体分析中排除。



# 返回形式

```python
epoch_accuracies = [calculate_accuracy(epoch, prompts) for epoch in epochs]
'''
    {
        "prompt1": accuracy_value1_1,
        "prompt2": accuracy_value2_1,
        "prompt3": accuracy_value3_1,
        ...
    }
    {
        "prompt1": accuracy_value,
        "prompt2": accuracy_value2,
        "prompt3": accuracy_value3,
        ...
    }
    
 '''
```

```python
prompt_accuracies = {prompt: [epoch[prompt] for epoch in epoch_accuracies] for prompt in prompts}
'''
    {
        "prompt1": [accuracy_epoch1, accuracy_epoch2, ..., accuracy_epochN],
        "prompt2": [accuracy_epoch1, accuracy_epoch2, ..., accuracy_epochN],
        "prompt3": [accuracy_epoch1, accuracy_epoch2, ..., accuracy_epochN],
        ...
    }
    '''
```

