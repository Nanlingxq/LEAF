# 指令微调中的可学习温度

更新：2026-10-19。这里说明已经完成的代码改动；V3.3训练与推理尚未执行，没有新的效果结果。

`b_instruct_tuning.py` 现在支持在归一化EEG表示与标签原型的点积前加入全局可学习正尺度：`logits = exp(logit_scale) * cosine_logits`，温度为它的倒数。训练与验证共用同一计算函数。

## 配置与更新

MPNet配置 `configs/LEAF_mpnet.yaml` 中新增：

```yaml
instruct:
  temperature:
    enabled: true
    scale_init: 1.0
    scale_min: 0.01
    scale_max: 100.0
    lr: 2.0e-4
```

这是原instruct配置下的新增字段，不替代其他超参数。温度参数单独进入第三个AdamW组，weight_decay=0，沿用原warmup与cosine调度；每batch结束投影到规定范围。启用时矩阵乘法与尺度运算显式退出autocast，使用FP32；模型本身仍按原精度运行。旧YAML缺少该字段或enabled=false时，恢复原始未缩放分数与两组优化器。

## 日志和保存

每epoch写 `temperature_history.csv`，记录epoch、global step、log尺度、尺度、温度和该组当时LR。验证记录缩放CE，并额外记录unscaled CE以帮助解释；loss_it.npy仍为验证CE/普通accuracy两列。

周期checkpoint仍是原格式LEAF state_dict，可由现有推理程序strict load。旁边新增 `itNNN.ckpt.temperature.pt`，记录温度参数、配置和对应model SHA256。该标量位于外层LightningModule，不会混入LEAF的396项权重。这些文件没有optimizer/scheduler/RNG状态，不能宣称支持精确断点续训。

推理分类不必额外乘尺度：同一正数不会改变类别argmax。可学习温度的作用发生在训练中；它不是对既有it060做一次温度后处理就能改变bAcc。

## 验证和V3.3对照

CPU检查命令：

```bash
python -m unittest discover -s tests -p test_instruction_temperature.py -v
```

检查只使用TinyLeaf和合成向量，覆盖兼容路径、训练/验证一致性、梯度及优化器、FP32与限幅、sidecar、配置校验、epoch日志保存；没有运行真实LEAF训练或EEG推理。

V3.3拟从同一官方pretrained、seed42重训100epoch；固定新it100对比历史seed42 it100为主，新旧it060同轮次对比为补充。保持17任务、采样、标签、其他超参数和推理规则一致。重点观察Speech/Workload的每类召回和类别集中，也报告全部任务。FP32 logit路径是需要说明的数值差异，结果不等于单独标量的严格因果证明。

服务器执行计划位于 `.codex/plans/reproduction-plan-v3-3-seed42-learnable-temperature.md`，实验开始前须按AGENTS.md归档完整配置、源码和输入身份。本次只有代码检查和计划，没有训练结果。

## 绘制it060前后对照

需要本地具备V3.2和V3.3 attempt2的归档结果CSV；数据、权重和实验档案不随GitHub代码发布。脚本只读指标，不运行模型：

```bash
python tools/plot_it060_temperature_comparison.py --project-root /path/to/LEAF-main
```

默认输出到artifacts/v3-3-seed42-learnable-temperature-attempt2/figures/it060-before-after/，也可用--output-dir指定目录。主图分三行对照Default、Task、Task+Target，保留原图16任务顺序；另导出17任务扩展与Fig.3参考风格版本。所有版本均提供PNG/PDF/SVG和来源SHA。
