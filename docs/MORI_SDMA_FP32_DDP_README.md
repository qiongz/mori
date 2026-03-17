# MORI FP32 + DDP 改造总结

本文总结 MORI 在 fp32 allreduce 路径上的近期改动，目标是让 DDP 训练调用具备：

- 与 NCCL 类似的异步接口形态（work handle）
- 多 bucket 场景的稳定性（避免 hung / 错配）
- 尽可能高的 overlap 与吞吐

## 为什么要这样改

在 DDP 真实训练中，allreduce 不是单次调用，而是连续 bucket 的高频调用。  
如果通信库只在“单次同步调用”下可用，会出现两个问题：

- **功能问题**：多 bucket 时跨 rank 启动时序不一致，可能导致等待条件错配，最终卡死。
- **性能问题**：每个 bucket 都在 CPU 侧强同步，会破坏 `allreduce(bucket N) || backward(bucket N+1)` 的重叠。

因此，本次改造重点不是“单点带宽”，而是“训练场景下的可用性 + 稳定性 + 异步接口一致性”。

## 哪些改动是 DDP 必需的

以下改动属于“没有就容易挂”或“语义不正确”的必需项：

1. **stream-ordered flags reset（替代 host 侧 reset）**
   - 位置：`src/collective/core/twoshot_allreduce_sdma_class.cpp`
   - 变更：在有 stream 时使用 `hipMemsetAsync(flags, ..., stream)`。
   - 原因：避免下一 bucket 的 host reset 与上一 bucket 的在途 kernel 并发，导致 flags 被提前清零。

2. **GPU 侧 launch rendezvous（LaunchSyncKernel）**
   - 位置：`include/mori/collective/allreduce/twoshot_sdma_kernel.hpp`、`src/collective/core/twoshot_allreduce_sdma_class.cpp`
   - 变更：在 collective 内部增加 GPU 端启动对齐。
   - 原因：异步 stream 下各 rank 发射时刻不同，必须在 kernel 内部达成“这一轮都已进入”。

3. **launch token/epoch 化（避免跨轮串代）**
   - 位置：`include/mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp`、`src/collective/core/twoshot_allreduce_sdma_class.cpp`
   - 变更：`launch_ready_epoch_` 单调递增，LaunchSync 按 epoch 等待。
   - 原因：0/1 布尔协议在多 bucket 连续调用时容易把上一轮状态误当作当前轮。

4. **copy 前同步可配置（支持 overlap 语义）**
   - 位置：`src/collective/core/twoshot_allreduce_sdma_class.cpp`
   - 变更：`MORI_SDMA_DEFER_SYNC=1` 时，跳过 copy 前 `hipStreamSynchronize`。
   - 原因：让 C++ 在排队后尽快返回，允许上层与 backward 并行推进。

## 哪些改动是性能/工程优化

以下改动主要用于提升接口一致性、降低上层开销、便于训练集成：

1. **NCCL-like 异步 work handle**
   - 位置：`src/pybind/mori.cpp`
   - 变更：新增 `AllreduceSdmaWork`，提供 `wait()/is_completed()/result()`。
   - 价值：上层可按 torch/NCCL work 模式使用，不必手写 event 管理逻辑。

2. **fp32 handle 新增 `allreduce_inplace_async`**
   - 位置：`src/pybind/mori.cpp`、`python/mori/ccl/collective.py`
   - 变更：`allreduce_inplace_async(data, count, stream)` 返回 work。
   - 价值：训练 hook 只需“发起 + wait work”，逻辑更薄、更稳定。

3. **新增 `allreduce_inplace_avg`（fp32）**
   - 位置：`include/mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp`、`src/collective/core/twoshot_allreduce_sdma_class.cpp`
   - 变更：allreduce 后在同 stream 上执行缩放 kernel。
   - 价值：把平均缩放下沉到 C++ 侧，减少 Python 额外开销（可选路径）。

4. **测试与基准补齐 work 路径**
   - 位置：`tests/python/ccl/test_allreduce_async.py`、`tests/python/ccl/benchmark_allreduce_bw.py`、`tests/python/ccl/benchmark_allreduce_config.yaml`
   - 变更：新增 `mori_async_inplace_work` 模式与对应测试入口。
   - 价值：可直接对比 NCCL in-place，观察异步接口路径收益。

## DDP 训练调用的建议约定

针对 fp32 DDP hook，建议优先使用：

- `ar.allreduce_inplace_async(tensor, count, stream=...) -> work`
- 后台或回调中 `work.wait()` 后再完成 `Future.set_result(...)`

并配合：

- `MORI_SDMA_ASYNC_COPY=1`
- `MORI_SDMA_OVERLAP=1`
- `MORI_SDMA_DEFER_SYNC=1`

说明：上层仍需保证“多 rank collective 发射顺序一致”（例如按 bucket index 有序发射），这点和 NCCL 的训练集成经验一致。

## 后续还能做什么提升性能

### 高优先级（建议先做）

1. **C++ 原生 Work 对象下沉到核心库层**
   - 当前 work 在 pybind 层封装 event；下一步可在 C++ core 暴露统一 async handle，减少绑定层状态管理。

2. **fp16/bf16 补齐 `allreduce_inplace_async`**
   - 让非 fp32 训练也走统一接口，减少上层分支与 fallback 抖动。

3. **在 Work 中补齐错误状态/异常传播**
   - 对齐 torch/c10d Work 语义，提升训练框架可观测性与容错。

### 中优先级

4. **减少 GPU busy-spin 对 CU 占用**
   - 对 `wait` 段采用更温和的轮询策略或分层等待，减少对计算重叠的干扰。

5. **细化 launch/complete tracing**
   - 导出每 bucket 的 launch_ts/complete_ts，定量评估 overlap 率与尾延迟。

6. **分桶与对齐策略自动化**
   - 针对 `npes * pack_size` 对齐约束提供自动 bucket 调整，减少 fallback。

### 低优先级（工程体验）

7. **benchmark 脚本继续增强**
   - 增加重画、指定 xticks、更多模式对比（已部分完成），方便回归分析。

## 结论

这次改动的核心价值是：  
把 MORI fp32 allreduce 从“可跑单点测试”推进到“可用于 DDP 多 bucket 异步训练”的状态，并在接口形态上向 NCCL/torch work 模式靠拢。  
后续若继续把 async handle 下沉到 core 并补齐低精度路径，训练侧集成成本和性能都还有明显上升空间。

