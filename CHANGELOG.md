# 改动记录

## 2026-10-19：完善指令微调的可学习温度

- 将训练和验证统一为同一温度logit计算，启用时使用FP32分数路径；增加配置开关和参数校验。
- 温度参数使用独立无weight_decay参数组，增加范围投影、逐epoch曲线和checkpoint旁路文件。
- 保留原LEAF checkpoint格式及旧配置无温度路径；增加7项CPU合成检查，没有启动训练或真实推理。
- V3.3拟固定新旧seed42的it100作为主对照、it060作为补充。说明见 [temperature-training.md](docs/temperature-training.md)。
- 根AGENTS.md新增所有执行Agent修改代码后必须提交、推送并说明改动的规则。数据、权重与本地实验档案不随代码提交。

## 2026-10-20：增加it060温度训练前后对照绘图

- 新增tools/plot_it060_temperature_comparison.py，读取已归档V3.2/V3.3 it060指标，绘制三面板前后配对柱状图、含ADHD版和Fig.3参考风格版。
- 导出PNG/PDF/SVG、完整bAcc/Kappa数据CSV和来源SHA；检查17个数据集、51个条件的指标范围及样本数一致。
- 脚本实际运行通过，16数据集主图与紧凑图目视检查通过；没有调用模型、运行新实验或修改历史结果。
