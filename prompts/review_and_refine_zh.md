# 候选审核与受限修框模板

本次新增的待验证模板；输出须经代码校验，不可仅相信模型遵守范围。

---

你将收到待测原图、图像宽高、候选框ID与原框坐标、目标类别定义，以及少量已标参考图。
逐个审核候选，返回 accept、reject 或 uncertain，不新增候选ID。

accept：类别与范围明确符合定义。
reject：明确为错误类别或背景。
uncertain：遮挡、边界或类别无法可靠判断。

可以为 accept 提出更贴合边界的修正框，仍使用原图像素 xyxy。修正只允许：
中心横纵移动分别不超过原框宽高20%；新框宽高分别处于原框80%–120%；完整位于图像内。
超出以上范围或无法可靠判断边界时，不提供修正框。

只输出：
`{"reviews":[{"candidate_id":"p1","decision":"accept","refined_box":[10,20,110,220]}]}`

reject/uncertain 不带 refined_box。没有修改时，accept 可以省略 refined_box。

