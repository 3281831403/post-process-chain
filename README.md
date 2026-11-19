# post-process-chain

## 题面

加了泛光、色调映射和 FXAA 后，画面开始发灰还掉帧：拓扑排序按注册顺序而不是依赖顺序，依赖倒置的链会把消费者排在生产者前面；循环依赖没有检测，递归直接爆栈；临时 RT 不复用，10 个效果就开 10 张全尺寸图；格式转换插在链路中间，HDR 还没到色调映射就被压成 8 bit；屏障只在跨 pass 时插入，同一 pass 内读写同一张 RT 漏屏障；半分辨率按向下取整，奇数尺寸丢掉一行像素；合并判断漏掉格式差异，rgba16f 和 rgba8 的 pass 被当成同一份；校验不查采样越界，画面边缘被拉伸；超预算时静默把后处理降成直通，用户只看到画面变糊；多帧执行共用同一份 RT 状态，上一帧的残留污染下一帧。render/postfx.py 是核心，tests/test_postfx.py 15个红。要修的是：按依赖拓扑排序、迭代检测循环、临时 RT 复用、HDR 保持到最终输出、补齐同 pass 屏障、半分辨率向上取整、合并前校验格式、校验采样边界、超预算明确失败、逐帧隔离 RT 状态。

## 运行方式

```bash
python -m pytest tests -q
```

初始仓库输出恰好 `15 failed`：15 个测试全部命中 `NotImplementedError`，没有 collection / import / syntax error。仅使用标准库与 pytest，离线可跑，无随机、无 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_postfx.py` 不可修改；实现只改 `render/postfx.py`（必要时可加 `render/` 内的辅助模块）。实现只使用 Python 标准库（`math`、`json`）与 pytest，不联网、不使用随机数。
