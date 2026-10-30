# 历史实验数据

| 文件 | 内容 |
|---|---|
| [runs.csv](runs.csv) | 两个 val300 原表全部 69 行，含状态和元数据标记 |
| [box_precision.csv](box_precision.csv) | 后续框精度原表 114 行；与 runs.csv部分重叠，不相加计独立实验 |
| [early_prompt_runs.csv](early_prompt_runs.csv) | 早期提示对比全部行，单独口径 |
| [source_manifest.json](source_manifest.json) | 原文件名、日期、字节数与 SHA256 |
| [archive/](archive/) | 四份原表原字节归档 |
| [run_template.json](run_template.json) | 后续运行应保存的元数据模板 |

这些 CSV 是**转录已存指标**，没有运行模型或重算历史 AP。空单元格表示缺失，绝非零。百分数列带 `_pct` 后缀。模型名、状态和异常仍按原表保留。

吊物数据指纹：`470b98531e4a34810e7ee42bf250e39c97480e42427c3d84152e29e4d368f3fc`。

叉车数据指纹：`bf04a484ca78255b5d8a8208dac79a6ac350c0dae1b4aa696bb5d921140b8d2b`。

已知缺口：部分 MLLM 无逐图坐标，`完成（元数据）`不等于逐图复核；原表的 prompts/metrics/records 链接对应原工程。CSV 与归档保留了来源行号，方便回查。

```bash
python3 scripts/import_reports.py  # 从归档重新转录三个 CSV 与来源清单
```

