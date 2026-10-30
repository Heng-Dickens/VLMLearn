# VLMLearn

面向工业场景的少样本目标检测学习与实验笔记。目标是**用少量框标注辅助自动标注，优先保证精确率和框边界质量**；目前只需要检测框。

研究对象：特殊吊装目标，以及叉车/牵引车五类目标。内容整理自截至 **2026-10-30** 可检索的项目聊天、已上传实验报告和官方实现；重复问答已合并。历史聊天不是完整导出，缺失的训练代码、配置与预测记录已明确标出。

![技术关系图](docs/assets/technical_map.png)

## 从这里读

| 顺序 | 文档 | 要解决的问题 |
|---|---|---|
| 1 | [技术演化主线](docs/01_technical_path.md) | VLM、MLLM、OVD、文本/视觉提示到底是什么关系 |
| 2 | [YOLOE 与 SAVPE](docs/02_yoloe_savpe.md) | Mask 如何融合，16/32/512 分别是什么，哪些参数可训练 |
| 3 | [Grounding DINO、PET-DINO、WeDetect](docs/03_grounding_pet_wedetect.md) | 900 个参考框从哪来，几条检测路线有何差别 |
| 4 | [已有实验结论](docs/04_experiment_findings.md) | 哪些结论有数值支持，哪些还只是观察 |
| 5 | [评估与下一步](docs/05_protocol_and_next.md) | 怎样衡量少改框、高精度，以及先补哪组实验 |
| 6 | [MLLM 提示与框审核](docs/06_mllm_workflow.md) | 图、框、裁剪、JSON 怎样组合；两步方案如何约束 |
| 查阅 | [常见疑问](docs/07_faq.md) · [来源与覆盖范围](docs/08_sources.md) | 找解释和原始证据 |

## 目前最有用的结论

- 吊装 300 张验证集：YOLOE 视觉提示 AP50 随 1/5/10-shot 为 **53.03/54.81/55.96%**，增加示例的收益有限。
- 早期数据中，YOLOE 视觉与文本向量按 0.75/0.25 混合后明显退化；这只是该融合实验的结果。
- MLLM 在现有汇总中有较强结果，但部分实验缺少逐图坐标，只能保留历史指标，不能复算框误差。
- “去掉 SAVPE、训练基础图像模块更好”来自近期反馈，尚缺配置与定量结果；固定类别训练和动态视觉提示训练应分别比较。

## 文档、数据、代码分开

| 目录 | 内容 |
|---|---|
| `docs/` | 精简解释、技术图、来源链接 |
| `experiments/` | 结构化历史指标 CSV、原报告文本、复现缺口 |
| `prompts/` | 整理后的检测/审核提示模板；不是原 prompt 完整快照 |
| `src/vlmlearn/` | 无模型依赖的框几何、匹配与评估工具 |
| `scripts/` | YOLOE 视觉提示示例、参数检查、指标整理、评估命令 |
| `examples/` | 最小输入格式；全部为合成框 |
| `tests/` | 一对一匹配、空预测、边界约束等行为测试 |

先试无需 GPU 的工具：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/evaluate_boxes.py --gt examples/ground_truth.json --pred examples/predictions.json
```

YOLOE 模型示例见 [代码说明](scripts/README.md)。本仓库的工具为本次整理新增，不冒充已找回的历史训练项目。

