# 0906 图表调整记录

本版本只调整复现流程的提示信息、参数展示和图形样式；模型核心算法与 `data/` 原始数据未改动。

## 已完成

- 删除 XGBoost 1.7.6 不识别的 `lambdarank_num_pair_per_sample` 与
  `lambdarank_pair_method` 参数。它们原本会被 XGBoost 明确提示“未使用”，
  删除不会改变模型计算。
- 回归基线的目标继续使用一维数组，避免 sklearn `DataConversionWarning`。
- Figure 3/4 两市场线条恢复论文蓝色（Main）和黄色（ChiNext）。
- Figure 5/6 的坐标文字按论文写法统一为 `Total Fund (million)`、
  `Total return`、`Number of stocks`、`Trading Day`；Figure 6 曲线仅在绘图时
  恢复 1,000,000 初始资金尺度，CSV 和回测数据不变。
- Appendix C1.1、C1.3、C1.4、C1.5 的颜色与大小写标签按论文原图调整。
- Table 7 与 Appendix C1.5 共用 `runtime_config.py` 中的 `ESG_THRESHOLDS`：
  25% 为 5.80、50% 为 6.00；NS/PI 在同一档位始终使用同一个阈值，仍从原始 ESG
  数据实时筛选，不保存或读取中间结果。

## 明确未改动

- Appendix C1.2（券商箱线图）`compute_c2`/`plot_c2` 未修改，保持锁定。
- 模型训练、DQN 策略、LambdaRank/LambdaMART 算法和 `data/` 原始数据未修改。
- 附录最后一个表的参数寻优范围暂不处理。

## 复核

`0906` 与 GitHub 上传工作副本中的全部 Python 文件均已通过 `python -m py_compile`。
