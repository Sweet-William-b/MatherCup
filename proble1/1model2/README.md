# Problem 1 - Model 2

本目录实现问题一第二子问题（模型B）：

> 只允许使用一种车型，将全部货物运输完成，并使所用车辆数最少。

## 方法说明

当前实现遵循 `agents.md` 中模型B的思路，采用：

- 装载模式集合划分
- 模式库生成
- 启发式列生成风格的模式扩充
- 主问题分支搜索

由于本地环境没有 `OR-Tools / PuLP / Python MIP`，当前版本没有实现严格意义上的分支定价求解器，而是实现了一个可运行的近似版本：

1. 复用 `../1model1/solve_model1.py` 的单车装箱逻辑，生成单车可行装载模式。
2. 将每一种单车装法视为一个装载模式。
3. 在模式集合上做主问题搜索，使总车辆数尽量最少，同时覆盖全部需求。

## 文件说明

- `solve_model2.py`
  - 模型B主程序
  - 负责生成模式库、求解主问题、输出逐车装载方案
- `output/`
  - 运行后自动生成
  - 包含模式库、逐车方案、逐件坐标和汇总结果

## 运行方式

在当前目录执行：

```powershell
python solve_model2.py
```

## 输出文件

每种车型会输出：

- `vehicle_1_pattern_library.csv`
- `vehicle_1_fleet_vehicles.csv`
- `vehicle_1_fleet_items.csv`
- `vehicle_1_fleet_summary.json`
- `vehicle_2_pattern_library.csv`
- `vehicle_2_fleet_vehicles.csv`
- `vehicle_2_fleet_items.csv`
- `vehicle_2_fleet_summary.json`
- `model2_results.xlsx`

## 输出含义

- `pattern_library.csv`
  - 所有生成的单车装载模式库
- `fleet_vehicles.csv`
  - 最终被选中的车辆方案，一行对应一辆车
- `fleet_items.csv`
  - 最终逐件货物坐标，一行对应一件货物
- `fleet_summary.json`
  - 当前车型下的总车辆数、总成本和覆盖情况
- `model2_results.xlsx`
  - 将以上结果合并到一个工作簿中，便于论文整理和结果检查

## 当前结果

- `vehicle_1`：最少 `3` 辆，`exact_cover = true`
- `vehicle_2`：最少 `2` 辆，`exact_cover = true`

## 说明

- 当前版本优先保证“模型思路一致、代码可运行、输出完整”。
- 主问题求解属于“模式法 + 分支搜索”，是对“列生成 + 分支定价”思想的工程化近似实现。
- 若后续环境支持 `MILP/CP-SAT`，可进一步将主问题替换为更标准的整数规划或分支定价框架。
