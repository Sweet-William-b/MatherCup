# Problem 1 - Model 1

本目录给出问题1（1）中“固定单车型，综合满载率最大化”的一版可运行实现。

## 文件说明

- `solve_model1.py`
  - 主求解脚本
  - 使用“块 + 分层 + 精确搜索 + ALNS”混合流程
- `output/`
  - 运行脚本后自动生成
  - 包含每辆车的明细坐标表、层信息表、汇总 JSON 和 Excel 总表

## 方法对应关系

脚本中的实现与 `agents.md` 的方法一一对应：

1. `块（Superitems）`
   - 将同类、同姿态货物组合成规则块
2. `分层（Layer-based Packing）`
   - 先在二维底面上构造候选层
3. `精确优化`
   - 对候选层序列进行带约束的分支定界搜索
4. `ALNS`
   - 对初始层序列做破坏-修复改进

## 运行方式

在当前目录执行：

```powershell
python solve_model1.py
```

## 输出结果

脚本会为 `车型1` 和 `车型2` 分别输出：

- `vehicle_1_items.csv`
- `vehicle_1_layers.csv`
- `vehicle_1_summary.json`
- `vehicle_2_items.csv`
- `vehicle_2_layers.csv`
- `vehicle_2_summary.json`
- `model1_results.xlsx`

其中：

- `items.csv` 为“一行一件货物”的标准输出
- `layers.csv` 为层级结构信息
- `summary.json` 为利用率与装载数量汇总
- `model1_results.xlsx` 汇总所有结果，便于论文整理

## 说明

- 当前版本优先保证结构清晰、输出完整、便于论文与代码继续扩展。
- 由于本地环境没有 `OR-Tools/CP-SAT`，精确优化部分使用了候选层上的分支定界搜索来替代。
- 该实现适合作为问题1（1）的第一版可运行基线，并可继续向更强的局部精确排布或更细的支撑判定扩展。
