# 08 · 来源与覆盖范围

整理日期：2026-10-30。覆盖现有可检索聊天与下列四份实验文件。不是所有聊天的逐字导出；尚未找回的代码、原始逐图记录与完整配置不据摘要补造。

## 项目证据

| 来源 | 存放位置 | 用途 |
|---|---|---|
| results_summary.md（2026-10-28） | [吊物原表](../experiments/archive/diaowu_val300.txt) | 33 组，300 张、350 GT |
| results_summary(1).md（2026-10-28） | [叉车原表](../experiments/archive/chache_val300.txt) | 36 组，300 张、188 GT |
| prompt_results.md（2026-10-22） | [早期提示对比](../experiments/archive/prompt_results_early.txt) | 独立旧实验，评估规模不可与 300 张混合 |
| README.md（2026-10-29） | [高 IoU/几何误差原表](../experiments/archive/box_precision_20261029.txt) | 114 组，含 diaowu_99 与两个 val300 |
| 可检索项目聊天 | [结论与观察](04_experiment_findings.md) | 需求、结构疑问、微调观察与待验证方案 |

归档为 .txt，内容保持来源原字节；其中历史相对路径指向原运行工程，相关文件未随报告上传，不能视为本仓库可用入口。文件哈希见 [来源清单](../experiments/source_manifest.json)。

原报告使用 `qwen3.8-27b` 等实验标签，聊天出现 `27b_int8`；未取得权重/端点快照前不强行合并或改名。这些是历史标签，不是对公开模型命名的认证。

## 已合并的聊天主题

| 主题 | 阅读位置 |
|---|---|
| CLIP/ALBEF/BLIP/BLIP-2/MiniGPT-4/LLaVA | 01：对齐、融合、连接器、指令学习 |
| YOLOE文本、视觉、无提示；SAVPE Mask、16/32/512、参数 | 02 |
| Grounding DINO 两阶段与 900 框；PET-DINO；WeDetect 三角色 | 03 |
| 负例、裁剪、标注JSON、shot、模型比较与饱和趋势 | 04、06、完整 CSV |
| AP/P/R/R50、few-shot划分、训练验证数量、±20% | 05、07 |
| 只训练基础模块更好；MLLM审核后修框；OVD两步方案 | 04、05、06 |

## 官方阅读入口

- [CLIP](https://github.com/openai/CLIP)、[ALBEF](https://github.com/salesforce/ALBEF)、[BLIP](https://github.com/salesforce/BLIP)、[BLIP-2/LAVIS](https://github.com/salesforce/LAVIS)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)、[LLaVA](https://github.com/haotian-liu/LLaVA)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [YOLOE 论文](https://arxiv.org/abs/2503.07465)、[官方代码](https://github.com/THU-MIG/yoloe)
- [PET-DINO 论文](https://arxiv.org/abs/2604.00503)、[官方代码](https://github.com/fuweifuvtoo/PET_DINO)
- [WeDetect 论文](https://arxiv.org/abs/2512.12309)、[官方代码](https://github.com/WeChatCV/WeDetect)

实现核对快照：YOLOE `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`；PET-DINO `7830a462f9a320f34293cce0aabdda1256d9dc15`。源码路径链接固定到快照，避免后续 API 更新让学习笔记对不上。

## 图片说明

三张数据流图由本仓库脚本重绘，为教学简化图，不含工业原图。SAVPE以所查官方代码为准，PET-DINO的 N 随图像尺寸变化。

YOLOE 论文图 3 局部另附在 [SAVPE原图](assets/yoloe_figure3.png)，版权归原作者。已有聊天生成的 PET-DINO 图存在将 IBP 画成记忆库等容易误解的标签，未直接复用，改用核对后的简图。

