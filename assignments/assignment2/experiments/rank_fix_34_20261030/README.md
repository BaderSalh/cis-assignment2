# 分别固定 34 个参数的独立实验

这个目录包含当前模型与 notebook 的独立源码副本。主作业的源码、notebook、report 均不修改。

## 实验控制

- 机器人及 tracker systematic-field seed：1。
- 全局 tracker 测量噪声 seed：20261025。
- 复用 notebook 中场景、5 次回零、10 个 baseline 配置及 186 条正式校准记录。
- 只采集一次数据。所有 34 种固定参数实验用同一批数据。
- forward 测试使用相同的 729 个配置；真值仅用于测试，不参与拟合。
- Python 参数索引是 0-based；结果同时提供 1-based 编号，避免混淆。

## 两种不同检查

1. **固定在原拟合值，局部秩检查**：先无约束拟合 34 个参数，在该拟合点构造完整 Jacobian。逐一删除其第 k 列，得到 1116 × 33 矩阵。等价于将该参数固定在原拟合值后考察局部可辨识性；没有人为改变其他参数或数据。
2. **固定为 0，重新拟合**：将第 k 个参数固定在原始初始值 0，其他 33 个参数从原拟合结果出发重新优化。每种情况都重新计算 Jacobian、cost 和 forward holdout errors。这能检查该约束是否损害预测。

第二种实验不必与第一种得到相同秩：参数点变了，且固定为 0 可能排除了某些合理模型。为减少重复计算，reduced fit 使用批量计算的等价 residual；在原拟合点、10 个参数扰动点及所有最终解都与原函数逐分量核对（atol=2e-8，rtol=1e-9），Jacobian 诊断仍调用原函数中心差分。优化采用 x_scale='jac'、三个收敛容差 1e-9、最多 100 次报告的 function evaluations。它与原始 baseline 的默认优化设置不同，cost 的小幅下降不应解释为固定参数改善了信息量。

## 秩判定

主要判据与现有代码一致：singular_value > largest_singular_value × 1e-6。结果也记录 1e-5、1e-7 阈值、原始完整 J 的统一阈值，以及列单位范数归一化后的 rank。不同参数单位混合会影响 condition number 和弱向量分量大小，不能把原始 weak_abs 当成单位无关的“冗余程度”。

## 运行

从 assignment2 目录运行：

```sh
venv/bin/python -B experiments/rank_fix_34_20261030/run_experiment.py
```

运行只加载 source_snapshot 中代码，结果写在本实验目录。重跑会覆盖这里的实验结果。

## 输出

- `source_snapshot/`：模型、依赖代码、notebook 副本。
- `snapshot_hashes.json`：副本文件哈希。
- `frozen_data.npz`：同一批校准数据、初始基座估计、原拟合参数、Jacobian、奇异值和弱向量。
- `baseline.json`：无约束基准拟合诊断。
- `results.csv`：全部 34 种检查和重拟合结果。
- `parameters_fixed_00.npy` 至 `parameters_fixed_33.npy`：各重拟合参数。
- `run.log`：逐个参数的运行进度。
- `RESULTS.md`：完成后的对比结论。

“某参数可以固定”不等于“该参数是唯一多余的”；请结合弱方向、不同阈值与固定后预测误差解释结果。

## 未收敛案例的补充检查

初次 100 次 function evaluations 后仍未收敛的案例，使用 `retry_unconverged.py` 从该结果继续最多 1000 次；初始结果另存 `results_initial.csv`，最终更新写入 `results.csv`，日志在 `retry.log`。达到上限仍未收敛的行明确保留 success=False，不将它们当成已找到全局最优。
