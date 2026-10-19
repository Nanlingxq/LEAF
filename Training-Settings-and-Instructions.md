# 两阶段训练说明：超参数、语言指令与实际运行命令

更新日期：2026-10-16（Asia/Shanghai）。
本文件说明已经做过的 V1、V2、V2.1 实验，便于阅读和复查；不是让你重新启动训练。

## 先把两个阶段分清楚

你说的“第一阶段”是我们这次复现流程里的**指令微调**，第二阶段是**下游任务微调**。
论文在它们之前还有 EEG 自监督预训练，我们没有重跑，而是直接使用官方 pretrained 权重。

~~~text
官方 pretrained Tower
    ↓ 第一阶段：指令微调，17个任务一起训练
本地 V1 P1 it100.ckpt
    ↓ 第二阶段：每个数据集分别做任务微调
每个任务的逐epoch验证/测试指标
~~~

第一阶段让 EEG 表示学会和指令、标签文本关联。
第二阶段让模型适应某一个具体数据集，用新分类头判断类别。

这份说明以**运行时的归档配置和源码快照**为准。项目根目录里的默认YAML可能随后被修改，不能代替历史实验配置。

## 1. 两个阶段放在一起看

| 设置 | 第一阶段：指令微调（V1 P1） | 第二阶段：默认任务微调（V2） |
|---|---|---|
| 入口 | b_instruct_tuning.py | d_downstream.py |
| 训练范围 | 17个数据集的训练样本混合 | 每次只训练一个数据集 |
| 初始权重 | 官方pretrained，仅严格加载Tower | 本地P1 it100，严格加载完整LEAF |
| 新初始化部分 | Q-Former | 任务分类头 |
| 文本模型 | MPNet，预先算好的固定向量 | 同一份MPNet缓存 |
| 语言输入 | 随机Default/任务指令，可能混合标签集合信息 | 固定字符串None的向量 |
| 训练轮数 | 100 epoch | 200 epoch |
| batch size | 单GPU 256，全局256 | 单GPU 128，全局128 |
| seed | 42 | V2跑42/43/44；本次合并表只取42 |
| fast峰值学习率 | 1e-3 | 1e-3 |
| slow峰值学习率 | 2e-4 | 1e-4 |
| fast参数 | Tower的tokenizer | tokenizer＋新分类头 |
| slow参数 | 其余模型参数 | 其余模型参数 |
| 优化器 | AdamW | AdamW |
| weight_decay | 1e-3 | 1e-3 |
| betas / eps | (0.9,0.999) / 1e-8 | (0.9,0.999) / 1e-8 |
| warmup | 2 epoch | 1 epoch |
| 之后的学习率 | 每step余弦衰减，倍率从1降至约0.1 | 相同方式 |
| 训练精度 | Lightning bf16-mixed | CUDA bf16 autocast，FP32模型参数 |
| 梯度累积 / 裁剪 | 1 / 不启用 | 1 / 不启用 |
| DataLoader workers | 4 | 0 |
| pin_memory | true | false |
| 训练打乱 / 验证打乱 | true / false | true / false；测试也不打乱 |
| EEG长度 | 补零或裁剪到2000点 | 直接用各数据集原长度 |
| 验证频率 | 每10 epoch | 源码实际每epoch |
| 测试频率 | 不做测试评分 | 每epoch评分，但不据此选epoch |
| 保存内容 | 每10 epoch保存完整LEAF state_dict | 只存逐epoch指标NPZ |
| 后续使用哪一轮 | 固定it100 | 取验证bAcc最高的epoch，并列最早 |

**峰值学习率**是warmup结束后达到的最大值，不是整个训练过程中一直使用的数。
V2.1的改动见第8节，不能把这些默认值直接套到所有V2.1运行。

## 2. 两阶段共用的模型与环境

| 项目 | 实际值 |
|---|---|
| 采样率 / 通道数 | 200Hz / 65通道 |
| tokenizer窗口长度 | 100点，即0.5秒 |
| CNN宽度 / token维度 | 64 / 256 |
| Tower | 双向和因果两分支，每个分支12层 |
| 注意力头数 / FF倍数 | 8 / 4 |
| Q-Former | 8个query，query维度256，4层 |
| 文本编码器 | all-mpnet-base-v2，项目短名mpnet-base |
| 文本向量维度 | 768 |
| tokenizer dropout | 0.5 |
| Tower Transformer dropout | 0.1 |
| Q-Former attention dropout | 0.0 |
| Q-Former输出dropout | 0.3 |
| 下游分类头dropout | 0.3 |
| FiLM | 线性生成gamma/beta，不经过tanh |
| 输出表示 | 指令分支输出做L2归一化 |
| 随机设置 | Python/NumPy/Torch CPU/CUDA设种子 |
| CUDA设置 | benchmark=true，deterministic=false |
| FP32矩阵乘法精度 | medium |
| 运行环境 | LEAF，Python3.10.22、PyTorch2.6.0+cu124、CUDA runtime12.4 |
| 其他已归档版本 | Lightning2.5.6、NumPy1.26.4、scikit-learn1.7.2 |

固定seed不能保证不同GPU和不同执行环境逐位一致。
V1指令微调和V2默认下游使用物理GPU1；V2.1六次对照实际依用户执行授权使用GPU2。
这是历史运行设备说明，不是今后自动切换设备的授权。

## 3. 第一阶段到底怎样训练

### 3.1 训练什么、不训练什么

先新建完整LEAF，再把pretrained参数严格加载到Tower，Q-Former从新初始化开始。
tokenizer和其余参与forward的Tower/Q-Former参数共同优化。
文本编码器不在训练循环里运行：直接读取已经缓存好的向量，向量也不更新。

Tower中用于预训练重建的头没有参与这一阶段的forward，可能没有梯度。
所以“优化器包含所有模型参数”不等于“每个参数在每一步都有更新”。

### 3.2 学习率怎样分配

~~~text
lr_base = 1e-3

tokenizer：       1e-3 × 1.0 = 1e-3
其余模型参数：   1e-3 × 0.2 = 2e-4
~~~

前2个epoch从0逐步升到峰值，随后用余弦日程降到峰值的约10%。
因此结束附近fast约1e-4、slow约2e-5。
归档记录每epoch653个优化step，100轮共65300步；warmup覆盖1306步。
学习率实际按step推进，最后一点是有限离散日程的近似终值。

### 3.3 输入、指令与标签怎样组成一个样本

每次读取样本时：
1. 拿到一个EEG片段及其数据集名字、真实标签。
2. 随机选Default或该任务的自然语言指令。
3. 从缓存中取对应文本向量；大多数时候再混入当前任务的候选标签集合信息。
4. 对EEG随机补零/裁剪到2000点。
5. 输出EEG、指令向量、真实标签在全局原型表中的编号。

模型的EEG输出向量与全部唯一标签原型做内积，得到余弦相似度logits，然后计算cross-entropy。
本次17任务共有33个不重复的标签文本。训练时是在这33个全局标签中分类，不是每个batch都只用当前任务的两三类。
没有额外添加可学习温度或logit缩放项。

### 3.4 补零、裁剪与其他增强

少于2000点：训练时随机分配左右补零，验证时只在右侧补零。
超过2000点：训练随机截取，验证从开头截取。
正好2000点：不补不裁。

配置里的mask_ratio=0.5、频带起点1–50Hz、带宽6Hz属于预训练重建机制。
当前指令forward没有调用这些频带抑制/重建目标，不能把它们误记为指令微调期间启用的EEG增强。

## 4. 第一阶段使用的全部语言指令

指令不是交给一个聊天模型现场回答，而是先转成768维向量，用来调节EEG特征。
缓存中有79条文本：46条指令和33条唯一标签文本。

### 4.1 指令如何随机选择

每个样本：
- 10%概率从Default组抽一句；该组有None和Default，组内等概率。
- 90%概率从对应任务组抽一句，组内等概率。

例如MI的任务组有10句，每句在这一步被选中的概率为90%÷10=9%。
Default的None和Default各为10%÷2=5%。
Workload只有2句任务指令，因此每句各有45%的概率被选中。
不同任务组的对应关系和英文原文如下；保留运行时拼写，不自行改写。

### Default（2条）

通用占位指令：所有任务都可能抽到。

1. None
2. Default

### Motor（10条）

运动想象：所有MI任务。

1. This is a motor imagery decoding task
2. Decode motor imagery
3. Please decode the motor imagery
4. Classify the motor imagery task
5. Identify the motor imagery class
6. Predict the type of imagined movement
7. Determine which motor imagery action is performed
8. Infer the imagined motor action from the EEG
9. Which motor imagery category does this trial belong to
10. Analyze this EEG and classify the motor imagery

### Emotion（10条）

情绪识别：所有EMO任务。

1. This is an emotion recognition task
2. Decode emotional states
3. Please decode the emotional states
4. Identify the emotional state
5. Classify the emotion represented in the signal
6. Predict the emotion category
7. Determine the emotional condition from this EEG
8. Infer the participant emotional state
9. What emotion does this EEG signal reflect
10. Analyze the EEG and classify the emotion

### Speech（10条）

想象/隐语音：CS_BCIC_Speech。

1. This is a covert speech decoding task
2. Decode covert speech
3. Please decode the covert speech commands
4. Identify the covert speech command
5. Classify the imagined speech
6. Predict the covert speech category
7. Determine what the subject is silently saying
8. Infer the intended covert speech command
9. What covert speech does this signal represent
10. Analyze the EEG and classify the covert speech content

### SSVEP（9条）

视觉诱发频率：SSVEP_OpenBMI。

1. This is an SSVEP frequency decoding task
2. Decode SSVEP frequency
3. Please decode the SSVEP frequency
4. Identify the SSVEP target frequency
5. Classify the evoked frequency response
6. Predict the stimulus frequency from the EEG
7. Determine which frequency the subject was attending to
8. Infer the visual flicker frequency in this trial
9. Analyze the EEG and recognize the SSVEP frequency

### Workload（2条）

脑负荷：Workload。

1. Decode mental workload states
2. Please decode mental workload states

### ADHD_AliMotie（3条）

ADHD分类：ADHD_AliMotie。

1. Decode mental disorder
2. Decode ADHD vs Healthy
3. Please decode ADHD vs Healthy


### 4.2 标签信息怎样混进指令

选好指令向量后，再单独进行一次随机判断：
- 90%概率：混入当前任务所有候选标签的平均向量。
- 10%概率：保留原指令向量。

过程可以写成：

~~~text
task_labels_mean = 当前任务全部候选标签向量的平均
label_hint = L2归一化(task_labels_mean)

mixed = (原指令向量 + label_hint) / 2
最终指令向量 = L2归一化(mixed)
~~~

例如左右手任务用的是Left和Right两个向量的平均，不是根据当前样本答案只塞入Left或Right。
因此这里提供的是“这一任务要在什么类别中选”的信息，不是直接把真实答案传给模型。
这里没有生成一条“This is ... Left vs Right”新句子，而是在向量层面混合。

两次随机选择分开进行，概率为：

| 指令来源 | 是否混合标签集合 | 概率 |
|---|---|---:|
| 任务指令 | 混合 | 81% |
| 任务指令 | 不混合 | 9% |
| Default | 混合 | 9% |
| Default | 不混合 | 1% |

这些是代码设定的理论概率，不是本次每轮实际采样次数统计。
验证时EEG的补零/裁剪位置固定，但指令抽样和是否混合仍随机；不能认为验证输入完全不变。

## 5. 哪些数据集、哪些标签参与训练

标签列表从左到右对应整数0、1、2……。
列表顺序必须保留，尤其OpenBMI-MI和ShanghaiU为Right/Left，和Shin等任务相反。
下表的样本量来自V1运行归档，已包含loader实际过滤结果。

| 数据集 | 指令组 | 标签顺序 | train | valid | test |
|---|---|---|---:|---:|---:|
| MI_OpenBMI | Motor | Right, Left | 13440 | 3360 | 4800 |
| MI_BCIC_IV2a | Motor | Left, Right, Foot, Tongue | 3148 | 788 | 1152 |
| MI_ShanghaiU | Motor | Right, Left | 7712 | 1929 | 2347 |
| MI_BCIC_Upperlimb | Motor | Cylin, Sphe, Lumbrical | 2640 | 660 | 1200 |
| MI_HighGamma | Motor | Left, Right | 3761 | 941 | 2040 |
| MI_Cho2017 | Motor | Left, Right | 6464 | 1616 | 1800 |
| MI_Shin2017A | Motor | Left, Right | 1056 | 264 | 360 |
| MI_PhysioNet | Motor | Left, Right | 11447 | 2862 | 5262 |
| EMO_FACED | Emotion | Anger, Fear, Disgust, Sad, Amusement, Inspiration, Joy, Tenderness, Neutral | 6720 | 1680 | 1932 |
| EMO_SEED_3_seg4 | Emotion | Negative, Neutral, Positive | 22545 | 7905 | 7620 |
| EMO_SEED_4_seg4 | Emotion | Neutral, Sad, Fear, Happy | 26025 | 5340 | 6210 |
| EMO_SEED_5_seg4 | Emotion | Disgust, Fear, Sad, Neutral, Happy | 8512 | 10672 | 9984 |
| EMO_SEED_7_seg4 | Emotion | Happy, Surprise, Neutral, Sad, Disgust, Fear, Anger | 34000 | 18542 | 17200 |
| SSVEP_OpenBMI | SSVEP | 12.0, 8.6, 6.6, 5.4 | 13440 | 3360 | 4800 |
| CS_BCIC_Speech | Speech | hello, help-me, stop, thank-you, yes | 3750 | 750 | 750 |
| Workload | Workload | Resting, Workload | 1515 | 379 | 240 |
| ADHD_AliMotie | ADHD_AliMotie | Healthy, ADHD | 916 | 115 | 599 |

17任务一起进入指令微调，train总数167091、valid总数61163。
把训练样本混合后shuffle，按样本抽取，没有额外设置“每个数据集同等概率”的平衡采样。
因此样本多的数据集在一个epoch里贡献更多训练样本。

HighGamma文件可能有Feet标签，但loader剔除编号2后使用Left/Right二分类。
ADHD参与了指令训练和扩展下游实验，但不在论文Table III的16项汇总中。
已交付HDF5的真实预处理来源需依据数据记录核实，不能只从shape推断滤波和采样处理全部正确。

## 6. 第二阶段怎样使用指令、特征和标签

### 6.1 下游实际传入的是None

所有本次下游任务都设置：

~~~text
--instrct None
~~~

这里的None是文本缓存中的字符串，不是Python空值，也不代表不提供向量。
程序会读取这个字符串对应的768维MPNet向量，并给每个样本使用同一向量。

下游阶段不再随机抽取任务句子，不做90%的标签集合混合，也没有为每个数据集另写一段prompt。

### 6.2 为什么只换任务指令未必影响这个分类结果

当前DownStream.forward调用LEAF，得到两种输出：
- emb：经过Q-Former得到的语义向量。
- tokens：Tower输出的EEG特征。

然后分类头实际使用的是tokens：

~~~text
EEG → Tower tokens → FlattenClassifier → 类别logits → cross-entropy
          └→ 指令条件Q-Former → emb（没有用于这条分类损失）
~~~

所以这条任务分类路径不依赖Q-Former的emb。
指令向量虽然仍传进LEAF，单纯把None改成某个任务句子，并不会让这里的分类头获得新的指令条件信息。
分类loss更新Tower和分类头；Q-Former不从这条分类loss得到梯度。

### 6.3 分类头与输入长度

分类头将所有Tower tokens展平，依次经过：
输入维度 → 512 → GELU → dropout0.3 → 256 → GELU → dropout0.3 → 类别数。

4秒EEG有800点，每分支8个token，拼接16个，分类头输入16×256=4096维。
10秒EEG有2000点，拼接40个token，输入40×256=10240维。

这一步没有把4秒样本补到10秒。FACED、ADHD为10秒，其余本次任务为4秒。
分类头输出按当前数据集自己的标签编号计算，不使用第一阶段的33类文本原型进行分类。

## 7. 第二阶段默认设置与各数据集结果来源

默认学习率分组：

~~~text
lr_base = 1e-3
fast（tokenizer＋classifier）：1e-3 × 1.0 = 1e-3
slow（其余模型参数）：        1e-3 × 0.1 = 1e-4
weight_decay = 1e-3
~~~

warmup1epoch，然后衰减到约10%；结束附近fast约1e-4、slow约1e-5。
每个数据集都从同一份P1权重重新初始化，优化器与任务分类头重新建立，不能顺着上一个任务训练下一个任务。

V2本来完成了seed42/43/44。当前给你展示的合并结果只使用seed42。
下表说明当前合并表每项究竟选用了哪一份配置；并不表示其他试过的配置被删除。

| 数据集 | 采用方案 | fast峰值LR | slow峰值LR | WD | 输入点数 | 选中epoch | GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| MI_BCIC_IV2a | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 161 | 1 |
| MI_OpenBMI | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 18 | 1 |
| MI_BCIC_Upperlimb | wd_3e3 | 1e-03 | 1e-04 | 3e-03 | 800 | 42 | 2 |
| MI_ShanghaiU | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 78 | 1 |
| MI_HighGamma | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 157 | 1 |
| MI_Cho2017 | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 169 | 1 |
| MI_Shin2017A | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 45 | 1 |
| MI_PhysioNet | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 30 | 1 |
| EMO_FACED | baseline | 1e-03 | 1e-04 | 1e-03 | 2000 | 199 | 1 |
| EMO_SEED_3_seg4 | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 40 | 1 |
| EMO_SEED_4_seg4 | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 148 | 1 |
| EMO_SEED_5_seg4 | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 15 | 1 |
| EMO_SEED_7_seg4 | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 137 | 1 |
| SSVEP_OpenBMI | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 180 | 1 |
| CS_BCIC_Speech | baseline | 1e-03 | 1e-04 | 1e-03 | 800 | 150 | 1 |
| Workload | lr_half | 5e-04 | 5e-05 | 1e-03 | 800 | 158 | 2 |
| ADHD_AliMotie | baseline | 1e-03 | 1e-04 | 1e-03 | 2000 | 8 | 1 |

所有行均batch128、训练200epoch。选中epoch只是读取指标的轮次，不是提前结束训练。
ADHD是扩展项，不混入MI/Emotion/Other/Overall的论文16项统计。
GPU2是V2.1执行时用户明确指定的历史覆盖，和最初计划GPU1不同。

## 8. V2.1到底改了哪些参数

这次只在Upperlimb、ShanghaiU、Workload上各做两项对照，共6次，seed42。

| 方案 | lr_base | fast峰值LR | slow峰值LR | weight_decay |
|---|---:|---:|---:|---:|
| V2 baseline | 1e-3 | 1e-3 | 1e-4 | 1e-3 |
| lr_half | 5e-4 | 5e-4 | 5e-5 | 1e-3 |
| wd_3e3 | 1e-3 | 1e-3 | 1e-4 | 3e-3 |

lr_half保持fast倍率1.0、slow倍率0.1，只把lr_base减半。
wd_3e3保持原学习率，只改weight_decay。没有把两种改动叠加，也没有重训指令微调。
两组使用独立workspace和同一份P1输入权重，均完成200epoch。

当前合并版本的选择规则是：每个方案先按验证bAcc选epoch，再按这个epoch的测试bAcc比较方案。
因此Upperlimb选wd_3e3、ShanghaiU保留baseline、Workload选lr_half。
这用到了测试成绩选择超参数方案，属于事后探索性对照；不是原V2的纯验证选点表。
Kappa取同一个方案、同一个epoch，没有另挑Kappa最大值。

有一处归档字段需要留意：部分V2.1完整config保留了继承的optimizer.AdamW.weight_decay=1e-3，同时在optimizer.weight_decay记录覆盖值。
核对实际WD时应结合该组真正运行的LEAF_mpnet.yaml、参数覆盖和comparison.csv；Upperlimb所选wd_3e3实际为3e-3，不是1e-3。
这些历史字段不在本次说明编写中擅自改写。

## 9. 验证集、测试集与权重保存

### 指令微调
训练集用于更新参数；每10epoch计算验证CE loss与普通accuracy。
loader会读到test数组后立即丢弃，没有加入训练Key或验证Key，也不做test评分。
下游固定使用epoch100，而不是按最好test或最好valid挑P1权重。

### 下游任务微调
每epoch用train更新参数，然后在eval/no_grad下计算valid和test。
每方案按valid bAcc最大选epoch，并列最早；测试指标取对应同一轮。
指标是把整个split的batch拼起来后评分，不是先算各batch再平均。
详细公式见项目根的Metrics.md。

这些运行保存的是八条逐epoch数组：
valBAcc、valAuroc、valwF1、valKappa、testBAcc、testAuroc、testwF1、testKappa。
没有逐样本预测，也没有任务微调的模型checkpoint。

P1每10epoch保存的则是完整LEAF的state_dict，有it010…it100。
它没有optimizer/scheduler/RNG恢复状态，不能当作完整训练断点。
训练后模型state_dict含BatchNorm等buffer，“不在验证时计算梯度”也应结合eval模式一起理解。

## 10. 当时运行的命令怎样读

下面是历史命令模板，仅用于解释参数。实际CWD很重要，不要复制到根目录直接重跑。

### 第一阶段：V1 P1

工作目录：
/userdata/lwk/EEG&LLM/Projects/LEAF-main/artifacts/v1-shin-pretrain-instruct-mpnet-seed42/workspace

~~~bash
python -u b_instruct_tuning.py \
  --config configs/LEAF_mpnet.yaml \
  --pretrain_ckpt cpt/leaf-v1.0-pretrain.ckpt \
  --gpu 1 --bs 256 --seed 42
~~~

--pretrain_ckpt指向官方Tower；--config决定模型与该阶段训练配置；
--bs为每GPU的batch，本次只有一张卡；--gpu为当时可见编号下的物理GPU1。
没有使用--no_warmup，该开关实际表示不加载pretrained，并不是取消学习率warmup。

### 第二阶段：V2默认任务微调

工作目录：
/userdata/lwk/EEG&LLM/Projects/LEAF-main/artifacts/v2-all-downstream-v1p1-mpnet-3seed/workspace

~~~bash
python -u d_downstream.py \
  --config configs/LEAF_mpnet.yaml --ds MI_ShanghaiU \
  --gpu 1 --bs 128 --seed 42 --itEpo 100 --instrct None
~~~

--ds换成对应数据集；--seed为这次任务微调的seed；
--itEpo100用于构造输入it100路径，不是让下游只训练100epoch。下游200epoch来自YAML。
缓存读取等操作也依赖运行目录，所以仅有argv不足以复现。

### V2.1所选Upperlimb、Workload

Upperlimb工作目录：
artifacts/v2-1-lr-wd-ablation-seed42/workspaces/wd_3e3/。

~~~bash
python -u d_downstream.py \
  --config configs/LEAF_mpnet.yaml --ds MI_BCIC_Upperlimb \
  --gpu 2 --bs 128 --seed 42 --itEpo 100 --instrct None
~~~

Workload工作目录：
artifacts/v2-1-lr-wd-ablation-seed42/workspaces/lr_half/。

~~~bash
python -u d_downstream.py \
  --config configs/LEAF_mpnet.yaml --ds Workload \
  --gpu 2 --bs 128 --seed 42 --itEpo 100 --instrct None
~~~

两条命令看起来相似，但不同workspace中的YAML不同，决定了LR/WD。不要只凭命令字符串判断参数一样。

## 11. 权重身份与原始依据

| 文件用途 | 身份 |
|---|---|
| 第一阶段输入pretrained | cpt/leaf-v1.0-pretrain.ckpt |
| pretrained SHA256 | bd5c29ddc047d0765fe72cbd4ee5e95b2bba37dd8632c2bfd31185fc010163ee |
| 第一阶段输出/第二阶段统一输入 | artifacts/v1-shin-pretrain-instruct-mpnet-seed42/workspace/checkpoints/LEAF_mpnet-42-leaf-v1.0-pretrain/it100.ckpt |
| it100 SHA256 | d44baeedd580110b849d9450b7475bab78d0aec7eea87e07d75a2c99f5dcbf85 |
| 第一阶段实际seed | 42 |
| 官方pretrained原始训练seed | 尚未确认，不以本地seed42代替 |

原始归档：
- [V1指令微调完整配置](.codex/runs/v1-shin-pretrain-instruct-mpnet-seed42/configs/instruction.yaml)。
- [V1当时的任务与指令表](.codex/runs/v1-shin-pretrain-instruct-mpnet-seed42/configs/source/tasks.yaml)。
- [V2默认结果及配置入口](.codex/runs/v2-all-downstream-v1p1-mpnet-3seed/README.md)。
- [V2.1六次对照](.codex/runs/v2-1-lr-wd-ablation-seed42/comparison.csv)。
- [当前合并版本的逐项来源](.codex/reports/v2-1-test-selected-merged-seed42/selection-manifest.json)。
- [指标的通俗解释](Metrics.md)。

本文件是对已执行实验的说明；新实验仍须在对应run/configs保存完整生效配置，不能只引用这份总览。
