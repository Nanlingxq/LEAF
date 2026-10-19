# 改动记录

## 2026-10-19：完善指令微调的可学习温度

- 将训练和验证统一为同一温度logit计算，启用时使用FP32分数路径；增加配置开关和参数校验。
- 温度参数使用独立无weight_decay参数组，增加范围投影、逐epoch曲线和checkpoint旁路文件。
- 保留原LEAF checkpoint格式及旧配置无温度路径；增加7项CPU合成检查，没有启动训练或真实推理。
- V3.3拟固定新旧seed42的it100作为主对照、it060作为补充。说明见 [temperature-training.md](docs/temperature-training.md)。
- 根AGENTS.md新增所有执行Agent修改代码后必须提交、推送并说明改动的规则。数据、权重与本地实验档案不随代码提交。
