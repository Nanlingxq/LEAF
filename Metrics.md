# Metrics：怎么看 LEAF 的 Acc 和 Kappa

日期：2026-10-13（Asia/Shanghai）。
Metrics 的意思是“指标”，所以文件名改成了 **Metrics.md**。Materix 拼写不对；Matrix 是“矩阵”的意思。

这份文档主要解释两件事：**模型对每个类别识别得怎么样，以及这个成绩比偶然猜对好多少。** 下面先用例子讲，再给公式，方便核对代码。

## 1. 论文里的 Acc，是哪种“准确率”？

LEAF 论文用的是 **平衡准确率**，英文叫 balanced accuracy，表里写作 B.Acc，代码里通常写 bAcc。[论文 v2，IV-B 与 Table III](https://arxiv.org/html/2509.24302v2)

普通准确率看“所有样本一共猜对了多少”；平衡准确率先看“每个类别分别猜对了多少”，再取平均。区别在于：样本多的类别，不能靠数量把样本少的类别压过去。

| 指标 | 它在问什么？ | 本项目在哪里用？ |
|---|---|---|
| 普通 Accuracy / Acc | 所有样本里，有多少预测对了？ | 指令微调时的内部验证指标 |
| 平衡准确率 / bAcc | 每个类别分别识别对多少？平均下来是多少？ | 任务微调和直接推理的主指标 |
| Cohen’s Kappa | 考虑到偶然也会猜对一些，模型还多做对了多少？ | 任务微调和直接推理的主指标 |

因此，看到日志里的 acc，先确认是哪一种。b_instruct_tuning.py 的 validation_step 用的是 (logits.argmax(dim=-1) == y).float().mean()，算的是普通准确率。它还要在所有训练标签原型中选答案，和 Table III 按各任务计算的 bAcc 不是同一个成绩。

## 2. 先数清楚：哪些猜对了，哪些猜错了？

可以把真实答案和模型答案放进一张表，这就是“混淆矩阵”。

**每一行放真实类别，每一列放模型预测的类别。** 对角线上是猜对的，其余位置是猜错的。

后面的公式用这些字母：

- K：类别数；N：样本总数。
- y：真实答案；ŷ：模型预测。
- C[i,j]：真实是 i，却被模型判成 j 的样本数。
- r[i]：真实属于 i 的样本数，也就是第 i 行加起来。
- c[i]：被模型判成 i 的样本数，也就是第 i 列加起来。
- C[i,i]：类别 i 猜对的数量；全部 r[i] 加起来就是 N。

任务微调时，模型的分类头给每个类别一个分数，选分数最高的类别，这一步叫 argmax。先做 softmax 再选，答案也一样。
直接推理时，则选与 EEG 表示最相似的标签文本。虽然产生答案的方法不同，算指标时都只比较“真实类别”和“预测类别”。

类别编号要对上：OpenBMI-MI 的 0/1 对应 Right/Left，Shin 的 0/1 对应 Left/Right。不能只把一边的左右顺序换掉。

## 3. 普通 Acc：一共猜对了多少？

算法就是：

**猜对的样本数 ÷ 全部样本数。**

$$
\mathrm{Acc}=\frac{\sum_{i=1}^{K}C_{ii}}{N}.
$$

比如100个样本猜对85个，Acc就是85%。但如果90个样本都是左手、只有10个是右手，模型即使不太会认右手，总成绩也可能很高。这就是这里还要看 bAcc 的原因。

## 4. bAcc：左手和右手分别打分，再取平均

以左右手分类为例，分别算：

- 所有真实左手样本中，有多少被识别成左手？
- 所有真实右手样本中，有多少被识别成右手？

把这两个比例平均，就是 bAcc。每个类别都占一半，与它有多少样本无关。

“一个类别的样本被找回来多少”叫 **召回率（recall）**。多个类别也是一样：先算每类召回率，再平均。

$$
\mathrm{Recall}_i=\frac{C_{ii}}{r_i},
\qquad
\mathrm{bAcc}=\frac{1}{K}\sum_{i=1}^{K}\mathrm{Recall}_i.
$$

这也叫 macro recall。它不是 precision 或 macro F1，也不是把 precision 和 recall 平均。
公式要求这些真实类别在测试数据中都有样本；缺了某个类别，就要检查程序实际怎么算，不能悄悄给它补一个0分。

二分类的公式也可以写成：

$$
\mathrm{bAcc}=\frac12
\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right).
$$

其中，TP是正类判对的数，FN是正类判错的数；TN是负类判对的数，FP是负类判错的数。两个分数分别叫 sensitivity 和 specificity。

**每个类别的样本数一样时，bAcc和普通Acc相等。** 数量不一样时，两者通常不同。

代码用 balanced_accuracy_score(y_true, y_pred)，没有给样本额外加权，adjusted=False。因此随机预测的基准没有被减掉：两类独立随机猜，期望 bAcc 是0.5；K类时是1/K。[scikit-learn 说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)

## 5. Kappa：考虑偶然猜对后，这个成绩还剩多少？

只看“猜对多少”还不够。比如某个类别特别多，模型又经常预测成这个类别，即使只是按这个比例随机出答案，也可能碰巧猜对不少。

Kappa先算出这种情况下的“偶然一致率”，再看实际成绩超过它多少。这里的偶然一致率按真实类别比例和预测类别比例计算，不是随便固定一个50%。

公式里的两个量是：

- p_o：实际猜对的比例，就是普通Acc，**不是bAcc**。
- p_e：按两边的类别比例独立随机配对，预计会碰巧一致的比例。

$$
p_o=\frac{\sum_i C_{ii}}{N},
\qquad
p_e=\sum_{i=1}^{K}\left(\frac{r_i}{N}\right)
                         \left(\frac{c_i}{N}\right),
\qquad
\kappa=\frac{p_o-p_e}{1-p_e}.
$$

最后一步可以读成：**实际成绩超过偶然基准的部分，除以从这个基准到完全正确之间的空间。**

| Kappa | 怎么理解？ |
|---|---|
| 1 | 所有答案都对 |
| 0 | 和按这些类别比例偶然猜对的水平一样 |
| 小于0 | 比这个偶然基准还差 |

Kappa既能算二分类，也能算多分类。它是一个没有单位的分数，不是“预测正确的概率”。

项目用 cohen_kappa_score(y_true, y_pred)，weights=None，也就是不加权：不会给某两类之间的错误额外设置距离惩罚。[scikit-learn 说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)

如果 p_e=1，公式分母就是0，程序可能返回NaN。遇到这种特殊情况要查输入，不能直接改成0分再平均。

## 6. 用100个样本算一遍

这是讲解用的例子，不是我们的真实实验预测。把类别0理解为左手、类别1理解为右手，会更容易读。

| 真实 \ 预测 | 类别0 | 类别1 | 真实总数 |
|---|---:|---:|---:|
| 类别0 | 80 | 10 | 90 |
| 类别1 | 5 | 5 | 10 |
| 预测总数 | 85 | 15 | 100 |

它表示：90个左手样本认对80个；10个右手样本只认对5个。

1. **普通Acc**：一共认对80+5=85个，所以是85/100=**0.8500**。
2. **左手召回率**：80/90；**右手召回率**：5/10。
3. **bAcc**：把左右手成绩平均，(80/90+5/10)/2=**0.6944**。
4. **偶然一致率**：真实左手占90%，预测左手占85%；真实右手占10%，预测右手占15%。所以 p_e=(90/100×85/100)+(10/100×15/100)=**0.7800**。
5. **Kappa**：(0.85−0.78)/(1−0.78)=**0.3182**。

同一组预测，普通准确率是85%，bAcc却只有69.44%，Kappa是0.3182。三个数字分别说明不同的事情：总数上认对不少，但右手识别只有一半，而按类别比例偶然也能对上不少答案。

## 7. 为什么我们的 Shin 指标能互相检查？

Shin测试集有180个左手样本、180个右手样本，数量正好相等。

所以普通Acc=bAcc，而偶然一致率一定是0.5。这时公式简化成：

$$
p_e=0.5,\quad p_o=\mathrm{bAcc},\quad
\kappa=2\,\mathrm{bAcc}-1.
$$

V1 seed42 的 bAcc=0.7861111111，代进去得到 Kappa=0.5722222222，和实验保存的值一致。

这个简式只在相应条件下成立，不能拿去套类别不平衡的数据。如果是K个类别，而且每类真实样本数都相同，则是：

$$
\kappa=\frac{\mathrm{bAcc}-1/K}{1-1/K}.
$$

还有一处需要记下：论文Shin的表值是bAcc=0.7256、Kappa=0.4811。按我们这个固定、平衡的二分类测试集，前者对应的Kappa应该约为0.4512。

**目前不知道这两个论文数字为什么对不上。** 可能和划分、汇总方式或表值有关，需要原始预测和实验记录确认。不能因此断言论文用了我们的同一份数据，也不能改公式来凑论文结果。[Table III(b)](https://arxiv.org/html/2509.24302v2)

## 8. 项目代码是怎么把这些数算出来的？

### 8.1 任务微调

在 d_downstream.py::eval_model 中：

~~~python
Logits = torch.cat(Logits, dim=0)
Labels = torch.cat(Labels, dim=0)
y_true = Labels.numpy()
y_pred = Logits.argmax(dim=1).numpy()
bAcc = balanced_accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
~~~

它会先收齐整个验证集或测试集的答案，然后算一次指标。哪怕数据分成很多batch送入模型，最后评分时也合在一起。

这种方式叫 pooled evaluation。不能先算每个batch或每个被试的成绩，再简单平均，认为一定会得到同样结果。

### 8.2 直接推理

在 c_inference.py::eval_subset 中：

~~~python
preds = (embs @ prototypes_cpu.T).argmax(dim=1).numpy()
bAcc = balanced_accuracy_score(labels, preds)
kappa = cohen_kappa_score(labels, preds)
~~~

也是把这一组样本合起来评分。目前 eval_dataset 传进去的是整个测试集。
如果以后改成“每个被试单独评分，再平均”，就要写清楚，不能直接和当前结果混用。

### 8.3 “平均”要说清楚是在平均什么

c_inference.py 的 MEAN，是把不同数据集的指标平均，每个数据集占同样的分量。

V1/V2 的 mean±SD，是对同一个数据集用不同种子跑出的成绩做统计。数据集、epoch、被试之间的差异，都不能当成种子之间的标准差。

## 9. 表里的“均值 ± 标准差”怎么来的？

每个seed训练过程中都有很多epoch。我们先用验证集决定保留哪个epoch，再查看它在测试集上的成绩：

1. 对每个数据集、每个seed，找 valid bAcc 最高的epoch；并列取最早的。
2. 读取那个epoch的test bAcc和test Kappa。
3. 将seed42、43、44各自的三个成绩求平均，并算标准差。
4. 测试成绩不参与挑epoch，也不把三个seed的预测合成一大组再评分。

均值告诉我们“三次平均表现如何”；标准差告诉我们“三次结果有多大波动”。

对某个指标x，S=3个种子：

$$
\bar{x}=\frac{1}{S}\sum_{s=1}^{S}x_s,\qquad
\mathrm{SD}=\sqrt{\frac{\sum_{s=1}^{S}(x_s-\bar{x})^2}{S-1}}.
$$

我们用的是样本标准差，也就是代码里的ddof=1。
论文说Table III是三seed的mean±std，但表格本身没有说明ddof和详细的epoch选择规则。因此这里写的是我们的实际做法，不假定论文所有细节都一样。[Table III](https://arxiv.org/html/2509.24302v2)

V1/V2的三个下游seed还共用了同一份seed42的P1权重。它们反映的是任务微调时的波动，不是把指令微调也各重做一次后的波动。

| 表达 | 怎么读？ |
|---|---|
| bAcc=0.7648 | 平衡准确率76.48% |
| bAcc SD=0.0197 | 三次结果的标准差是1.97个百分点 |
| Kappa=0.5296 | Kappa分数0.5296，不读成52.96%准确率 |
| bAcc差0.0392 | 差3.92个百分点 |
| Kappa差0.0485 | Kappa分数差0.0485 |

Table III用0–1的小数，论文Table II展示百分比尺度，比较前要换成相同单位。
Kappa即使乘100显示，也仍是Kappa，不会变成准确率。

## 10. 想自己核对，可以运行这段代码

在LEAF环境执行：

~~~python
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix,
)

# 第6节教学例子
y_true = np.array([0] * 90 + [1] * 10)
y_pred = np.array([0] * 80 + [1] * 10 + [0] * 5 + [1] * 5)

print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
# [[80, 10], [5, 5]]
print(accuracy_score(y_true, y_pred))          # 0.85
print(balanced_accuracy_score(y_true, y_pred)) # 0.694444...
print(cohen_kappa_score(y_true, y_pred))       # 0.318181...

seed_bacc = np.array([0.7861111111, 0.7472222222, 0.7611111111])
seed_kappa = np.array([0.5722222222, 0.4944444444, 0.5222222222])
print(seed_bacc.mean(), seed_bacc.std(ddof=1))
# 0.764814..., 0.019707...
print(seed_kappa.mean(), seed_kappa.std(ddof=1))
# 0.529629..., 0.039414...
~~~

这段教学代码已经在服务器的LEAF环境中验证，使用scikit-learn 1.7.2，没有运行模型训练。

还有一个限制：任务微调保存的NPZ只有各epoch的指标，没有每个样本的真实标签和预测标签。
因此可以用NPZ重新检查“选了哪个epoch”和“三个seed怎么平均”，但没法只靠它还原混淆矩阵，或者从预测重新算一遍所有指标。

## 11. 以后汇报指标，顺带写清楚这些信息

一个bAcc或Kappa数字，最好和它的来历一起给出来：

- 哪个数据集、哪份验证集或测试集，类别编号对应什么，每类有多少样本。
- 模型怎么得到预测，是整份测试集合起来评分，还是各被试先评分再平均。
- 用什么规则选epoch，跑了几个seed，标准差怎么计算。
- 从哪份权重开始训练。

这样别的Agent才能照着核对。遇到论文数字和本地计算关系不一致，就保留问题和依据，不改指标定义去凑结果。
