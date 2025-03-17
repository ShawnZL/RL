import matplotlib.pyplot as plt
import numpy as np

def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = -np.random.exponential(size=logits.shape)
    gumbel_sample = (logits + gumbel_noise) / temperature
    return np.exp(gumbel_sample) / np.sum(np.exp(gumbel_sample), axis=-1, keepdims=True)

# 给定 logits 和概率
logits = np.array([0, 1, 2, 3, 4])
logits_proc = np.array([0.10, 0.23, 0.15, 0.19, 0.33])

# 设置温度参数
temperatures = [0.1, 0.5, 1.0, 10.0]

# 设置图表标题和标签
fig, axs = plt.subplots(1, 4, figsize=(10, 4))  # 1行2列的subplot

for i, temp in enumerate(temperatures):
    # 绘制柱状图
    axs[i].bar(np.arange(len(logits)), gumbel_softmax_sample(logits_proc, temp), label=f'Temperature {temp}')

    # 设置每个subplot的标题和标签
    axs[i].set_title(f'Temperature {temp}')
    axs[i].set_xlabel('Actions')
    axs[i].set_ylabel('Probability')

# 调整subplot的布局
plt.tight_layout()

# 显示图表
plt.show()
