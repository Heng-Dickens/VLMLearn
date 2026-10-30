# 代码说明

本次新增的代码是独立工具与官方接口适配示例。尚未找回原运行工程中的训练脚本、原始预测和完整配置；不会用新代码声称复现了历史指标。

## 1. 核心评估工具：仅 Python 3.10+ 标准库

```bash
python3 -m unittest discover -s tests -v
python3 scripts/evaluate_boxes.py --gt examples/ground_truth.json --pred examples/predictions.json
python3 scripts/import_reports.py
```

GT 和预测都是记录列表，字段为 `image_id`（字符串）、`label`、原图像素 `box:[x1,y1,x2,y2]`。预测可带0–1的 `score`。空图以没有框记录表示，评估输入的图像范围应由调用方固定。

评估输出0–1率值，包含 IoU=0.50:0.05:0.95 的101点AP、均值AP、自定义四项20%AP、固定阈值 TP/FP/FN/P/R。AP使用全部输入候选，工作点阈值仅影响P/R；maxDets按每图每类限制，默认100。缺任一框分数时不算AP，未带分数的框始终进入固定输出评估；同分按输入顺序，输出明确标记。

这是供排查的透明小工具，没有 COCO crowd/ignore、面积分组或 LVIS Fixed AP 协议。正式 benchmark 继续用官方评估器；不能假定与旧工程逐位一致。原始候选已被上游阈值删除时无法恢复。

## 2. 受限修框检查

```python
from vlmlearn.boxes import check_refinement
ok, reason = check_refinement([100, 100, 200, 200], [110, 100, 210, 200], [640, 480])
```

在仓库根执行时先设置 `PYTHONPATH=src`。此函数只校验建议是否在原框的中心/尺寸20%约束及图像范围内，不负责判断边界是否更准确。失败时调用方应保留原框并标记待审；它不根据GT修正框。

## 3. YOLOE视觉提示：需要单独的上游环境

按 [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) 的安装说明准备环境，固定到 `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`，使用该仓库匹配的v8/11权重。不要混装最新pip版后假定API相同。本仓库不自动安装或下载权重。

复制 `examples/yoloe_reference_config.json`，填写本地参考图、框、目标图，路径相对该JSON所在目录。一个参考条目是一个标框实例；按类平均向量并重新归一化，保持参考方案一致。

```bash
python3 scripts/yoloe_visual_prompt.py --weights /path/yoloe-v8l-seg.pt --config /path/references.json --output runs/demo --device 0
python3 scripts/inspect_yoloe.py /path/yoloe-v8l-seg.pt
```

虽然使用 `-seg` 权重，示例只保存检测框，不导出Mask。推理使用上游默认NMS等设置；正式复现实验须另外显式记录这些参数。

**验证范围**：核心工具有CPU行为测试；YOLOE示例只完成语法与官方接口核对，当前环境无PyTorch/模型权重，尚未执行模型推理。checkpoint参数标记检查也不等同于训练器实际的冻结结果。

训练入口请先读官方 [train_vp.py](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/train_vp.py) 和 [train_pe.py](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/train_pe.py)；数据格式与Trainer不同，不直接给出未经验证的统一微调命令。

## 4. 图片

```bash
pip install -r requirements-visuals.txt
python3 scripts/draw_diagrams.py
```

只重绘三张教学流程图，不访问数据集或模型。`docs/assets/yoloe_figure3.png`为用户提供的论文图3局部，附作者来源，不由此脚本重建。
