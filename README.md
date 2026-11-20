# shadow-cascade-fit

## 题面

换成级联阴影后，走近物体阴影就抖、级联交界还有条亮缝：分割用线性，近处级联覆盖太远、分辨率浪费；`lambda_=1` 时退化成对数却仍按线性处理，末项不等于 far；正交范围不按 fov 与 aspect 扩展，宽屏两侧被裁；不对齐纹素，相机移动时阴影边缘抖动；级联切换不做稳定，级联边界出现一条亮缝；光空间矩阵未正交化，阴影被拉伸；剔除只按中心点，跨界级联的投射者被漏掉；纹素密度不按分辨率算，四个级联密度差 3 倍；批量共用包围盒缓冲，相机之间串；`count=0` 不校验，返回空列表导致后续除零。shadow/cascade.py 是核心，tests/test_cascade.py 15个红。要修的是：`split_scheme` 按 PSSM 分割比混合线性与对数且末项为 far，`fit_ortho` 按 fov 与 aspect 计算宽高，`snap_to_texel` 将包围盒外扩对齐到纹素网格，`stabilize` 按比例混合前后级联包围盒，`light_space_matrix` 构造正交基的光空间矩阵且行列式非零，`cull_casters` 用包围球与级联深度区间相交保留跨级投射者，`texel_density` 按分辨率与级联尺寸计算每米纹素数，`cascade_batch` 逐相机隔离数据并在任一相机非法时整体回滚。

## 运行方式

```bash
cd D:\gsb-20261119\shadow-cascade-fit
python -m pytest tests -q
```

初始状态应恰好输出：

```text
15 failed
```

15 条测试全部因桩函数 `raise NotImplementedError` 而失败；收集、导入和语法检查必须通过，不能出现 `error`。实现后测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 测试约束

`tests/` 是验收测试，不得修改、删除或新增；实现只能落在 `shadow/cascade.py`。所有距离与包围盒使用世界单位米，`fov` 使用弧度，`light_dir` 是光传播方向（内部归一化），断言容差为 `1e-6`。实现只使用 Python 标准库（`math`）与 pytest，离线、无随机。
