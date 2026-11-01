from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'results.csv').open()))
assert len(rows)==34, f'Only {len(rows)} runs finished'
b=json.loads((OUT/'baseline.json').read_text())
num=lambda r,k:float(r[k])
full=[r for r in rows if int(r['delete_rank'])==33]
post=[r for r in rows if int(r['refit_rank'])==33]
passing=[r for r in rows if r['forward_pass']=='True']
equiv=[r for r in rows if abs(num(r,'cost_change'))<1e-3]
lines=['# 逐一固定 34 个参数：实测结果','',
'## 实验范围',
'独立源码副本；同一批 186 条校准数据；robot seed=1，tracker noise seed=20261025；同一组 729 个 forward 测试配置。没有修改主作业或报告。',
'',f"原模型 rank={b['diagnostics']['rank']}/34，cost={b['cost']:.9f}；最大位置误差 {b['forward']['max_pos']:.9f} mm，最大角度误差 {b['forward']['max_angle']:.9f} rad。",
'', '## 汇总',
 f'- 在原拟合点删除对应列（固定为原拟合值），**{len(full)}/34** 种选择得到有效 rank 33/33。',
 f'- 分别固定为 0、重新拟合，其余 33 参数的 Jacobian 有 **{len(post)}/34** 种得到有效 rank 33/33。',
 f'- 固定为 0 后有 **{len(equiv)}/34** 种的 cost 与无约束基准相差小于 0.001（仅描述本次拟合结果，不是严格等价证明）。',
 f'- 固定为 0 后有 **{len(passing)}/34** 种在 729 个 forward 配置上仍同时满足位置/角度最大误差要求。',
'', '## 如何解释',
'“固定后 full rank”并没有只指向一个参数。多个不同参数都可以作为消除同一自由度的约束；这些参数参与了弱方向，不是各自都单独无用。',
'原拟合点的删列实验不改变参数点，更适合检验线性相关性；固定为 0 并重新拟合会改变参数点和允许的模型集合，不能与前者混为一谈。',
'若固定后 cost 上升或测试不达标，可能是该零约束限制了模型，也可能是优化未充分收敛；不能据此宣称它是“更冗余”，也不能把未收敛结果当成约束问题的全局最优。反之，保持 cost 和精度的约束是可用的 gauge 候选，但不意味着该参数物理上没有误差。',
'raw weak_abs 的大小依赖参数单位和参数化，不能用它给“冗余程度”做单位无关排名。应同时考虑固定值的物理解释、rank 阈值敏感性、condition number 和预测误差。',
'', '## 同一拟合点下可固定成 full rank 的参数', ', '.join('`'+r['parameter']+'`' for r in full),
'', '## 固定为零后几乎不改变 cost 的参数（|Δcost|<0.001）', ', '.join('`'+r['parameter']+'`' for r in equiv),
'', '## 全部 34 项',
'编号从 1 开始；CSV 同时保存 Python 从 0 开始的索引。Delete rank 是同一点删列；Refit rank 是固定为零后重拟合。Cost 是加权残差平方和的一半。',
'', '| # | 参数 | 原拟合值 | abs(v_k) | Delete rank | Refit rank | Δcost | Max mm | Max rad | Forward pass |',
'|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|']
for r in rows:
 lines.append(f"| {r['index1']} | `{r['parameter']}` | {num(r,'baseline_value'):.5g} | {num(r,'weak_abs'):.2e} | {r['delete_rank']}/33 | {r['refit_rank']}/33 | {num(r,'cost_change'):.5g} | {num(r,'max_pos'):.6f} | {num(r,'max_angle'):.7f} | {'Yes' if r['forward_pass']=='True' else 'No'} |")
lines+=['','## 阈值与收敛核对',
'以下标记未达到优化器收敛条件的案例应单独看待：'+', '.join(r['parameter'] for r in rows if r['refit_success']!='True'),
'有效秩主阈值为 σ_max×10^-6；CSV 另存 10^-5、10^-7、统一阈值和列归一化后的 rank。Full rank 不代表条件良好。',
f"达到优化器收敛条件：{sum(r['refit_success']=='True' for r in rows)}/34。",
'每个 reduced fit 从原无约束拟合的其余参数值出发，固定参数设为零；SciPy trust-region，x_scale=jac，tol=10^-9，初次 max_nfev=100。未收敛案例从当前解继续最多 1000 次；初次结果保存在 results_initial.csv，继续求解过程见 retry.log。',
'只测试 forward holdout，没有为 34 种约束逐一进行物理 inverse 到达测试；不能把 forward pass 解读为所有 IK 情况保证。',
'','完整数值：`results.csv`；原始矩阵、奇异值和弱向量：`frozen_data.npz`；逐次日志：`run.log`。']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
fig,axs=plt.subplots(3,1,figsize=(14,12),layout='constrained')
x=np.arange(34);labels=[r['parameter'] for r in rows]
axs[0].semilogy(x,[max(num(r,'weak_abs'),1e-14) for r in rows],'o',label='|weak-vector component|')
axs[0].set_ylabel('|v_k| (raw parameter units)');axs[0].set_title('34 one-parameter constraints: same data, separate reduced fits')
colors=['#177e89' if r['delete_rank']=='33' else '#c44e52' for r in rows]
axs[1].bar(x,[num(r,'delete_sigma_min') for r in rows],color=colors)
axs[1].axhline(b['singular_values'][0]*1e-6,color='black',linestyle='--',label='original full-J threshold')
axs[1].set_yscale('log');axs[1].set_ylabel('Reduced-J smallest singular value');axs[1].legend()
axs[2].plot(x,[num(r,'max_pos')/.2 for r in rows],'o-',label='Max position / 0.2 mm')
axs[2].plot(x,[num(r,'max_angle')/.001 for r in rows],'s-',label='Max angle / 0.001 rad')
axs[2].axhline(1,color='black',linestyle='--');axs[2].set_yscale('log');axs[2].set_ylabel('Forward error / target (refit with fixed value 0)');axs[2].legend()
for a in axs:
 a.set_xticks(x,labels,rotation=65,ha='right',fontsize=8);a.grid(axis='y',alpha=.2)
fig.savefig(OUT/'rank_and_accuracy.png',dpi=170)
print('\n'.join(lines[:20]))
