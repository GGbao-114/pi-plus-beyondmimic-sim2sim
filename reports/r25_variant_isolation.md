# 第25轮：单因子隔离结果

## 基线

修正公式、修正质量模型、Isaac fixed-action 数据与 MuJoCo fixed-action 数据；基线 3.5s/50步 max|Δq|=`0.048892`。

## 变体结果

| 变体 | 3.5s/50步 | 相对基线 | 结论 |
|---|---:|---:|---|
| baseline | 0.048892 | 0 | 基线 |
| implicit integrator | 0.047668 | -2.5% | 小幅改善 |
| damping=0 | 0.048892 | 0% | 无明显影响 |
| ankle_roll mesh→box | 0.114244 | +133.7% | 明显变差 |

全时刻/全 horizon 数据：

```text
/data/gaoli/logs/model_align_r16/variant_implicit.json
/data/gaoli/logs/model_align_r16/variant_damping0.json
/data/gaoli/logs/model_align_r16/variant_footbox.json
```

## 结论

1. MuJoCo implicit 相比 Euler 只带来小幅改善，不能单独解释残余 gap。
2. MuJoCo 被动 damping=0 与基线无可见变化，damping 不是主因。
3. ankle-roll mesh 替换为当前等效 box 后，分叉显著变差；脚底碰撞几何对结果敏感，但该变体没有改善跨引擎一致性，不能据此说原 mesh 已解释 gap。
4. 当前最谨慎的归因：接触几何/接触求解仍是高敏感因素，但正确的匹配几何或参数尚未找到；积分器/被动阻尼不是主因。

证据为【动力学对照】：同初始状态、同 fixed action、真实 step 后 q 对比。

## 产物

- `/data/gaoli/logs/model_align_r16/variant_implicit.json`
- `/data/gaoli/logs/model_align_r16/variant_damping0.json`
- `/data/gaoli/logs/model_align_r16/variant_footbox.json`
- `/data/gaoli/logs/model_align_r16/variant_mj.py`
- `/data/gaoli/logs/model_align_r16/mj_step_variant_runtime.py`
