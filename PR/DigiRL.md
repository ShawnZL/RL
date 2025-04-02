# 论文主要思想

[abstract](https://digirl-agent.github.io/)

利用offline + online 强化学习训练AutoUI(一个模型)比 gpt-4v与gemini 1.5Pro效果好

offline 利用离线数据集训练，提取有用的数据（SFT）

online 利用交互，继续提升模型

主要是两步，然后再去train actor

**指令级价值函数**（**instruction-level value function**）优先学习对智能体最具信息量的任务

**步级价值函数**（**step-level value function**）**步级价值函数**筛选轨迹中具有优势的动作（即推动目标达成的动作），同时剔除噪声动作（对目标无贡献的动作）

## Learning Curves

RL 对比FilteredBC，效果和学习速度都好

但是RL的Run1 和 Run2 对比，Run2学习更加缓慢

BCrun1 对比 Run2，Run2学习更快，但是最终效果差不多

**这里需要我看论文解释一下**



## 利用gemini做自动评估

我们的核心实验结果均通过**Gemini-1.5-Pro**进行自动化评估。同时，我们在部分数据子集上进行了人工评估，发现自动化评估结果与人工评估高度一致，平均差异小于3%。

**这一步类似于RLHF**



## 通过对于错误的统计

发现对于使用所有的错误都有优化效果，尤其是failing to recover from mistakes

# filtered behavior cloning

### **Filtered Behavior Cloning（过滤行为克隆）解释**

**Filtered Behavior Cluning（FBC）** 是一种改进版的 **行为克隆（Behavior Cloning, BC）** 方法，旨在解决传统BC在模仿学习（Imitation Learning）或离线强化学习（Offline RL）中可能遇到的 **次优数据（sub-optimal data）** 问题。  

---

### **1. 核心思想**
- **传统BC的问题**：  
  行为克隆直接模仿专家（或次优）数据中的状态-动作对 \((s, a)\)，但如果数据质量不高（如包含噪声、错误动作或非专家行为），模型会学习到不良策略。  
- **FBC的改进**：  
  在训练前，**先对离线数据集进行过滤（filtering）**，仅保留高质量（如高回报、低不确定性）的轨迹或状态-动作对，再执行BC。  

---

### **2. 关键方法**
#### **(1) 数据过滤（Filtering）**
- **基于回报（Return-based Filtering）**：  
  只保留数据集中 **回报（cumulative reward）较高** 的轨迹（如 top 10% 的专家演示）。  
- **基于不确定性（Uncertainty-based Filtering）**：  
  使用模型（如集成网络）估计动作的不确定性，过滤掉高不确定性的样本（可能噪声大）。  
- **基于Q值（Q-filtering）**：  
  结合离线RL方法（如CQL、BCQ），用学到的Q函数筛选 **高价值（high Q-value）** 的状态-动作对。  

#### **(2) 过滤后训练**
- 对筛选后的数据执行标准BC（即监督学习，最小化动作预测误差）：  
  \[
  \min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{filtered}}} \left[ \| \pi_\theta(s) - a \|^2 \right]
  \]

---

### **3. 优势**
- **缓解分布偏移（Distribution Shift）**：  
  过滤掉低质量数据后，策略在部署时更可能遇到训练时见过的状态，减少因数据不匹配导致的性能下降。  
- **提升样本效率**：  
  避免在噪声数据上浪费训练资源，专注于高价值样本。  
- **兼容离线RL**：  
  可与离线RL方法（如AWAC、IQL）结合，先用过滤数据初始化策略，再微调优化。  

---

### **4. 典型应用场景**
- **模仿学习（Imitation Learning）**：  
  从混合质量的专家数据中筛选最优演示（如自动驾驶中过滤人类驾驶员的错误操作）。  
- **离线强化学习（Offline RL）**：  
  在离线数据集上预训练策略时，避免学习次优行为（如机器人控制中过滤随机探索的低效动作）。  

---

### **5. 对比其他方法**
| 方法                | 是否需要环境交互 | 数据要求         | 适用场景           |
| ------------------- | ---------------- | ---------------- | ------------------ |
| **传统BC**          | ❌ 纯离线         | 需高质量专家数据 | 专家数据充足时     |
| **Filtered BC**     | ❌ 纯离线         | 容忍次优数据     | 数据质量不均       |
| **DAgger**          | ✅ 需在线         | 需在线查询专家   | 可交互环境         |
| **离线RL（如CQL）** | ❌ 纯离线         | 需覆盖多样状态   | 数据量大但质量一般 |

---

### **6. 代码示例（伪代码）**
```python
# 假设已有离线数据集 D = {(s, a, r, s')}
# 步骤1：过滤数据（例如基于回报）
filtered_D = [(s, a) for (s, a, r, _) in D if r > threshold]

# 步骤2：行为克隆训练
policy = NeuralNetwork()
for s, a in filtered_D:
    loss = MSE(policy(s), a)  # 最小化动作预测误差
    loss.backward()
    optimizer.step()
```
