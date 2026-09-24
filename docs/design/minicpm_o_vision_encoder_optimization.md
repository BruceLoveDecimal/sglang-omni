# MiniCPM-o 4.5 Image/Video Encoder 优化思路

> - 代码基线：上游 `sgl-project/sglang-omni` main @ `3a9855f`（下文行号均以此为准），sglang 0.5.20。
> - 关联：[#2284](https://github.com/sgl-project/sglang-omni/issues/2284) 第 2 项（Image/video encoder）、第 4 项（Encoder caches）、第 12 项（Runtime / placement）；[#2273](https://github.com/sgl-project/sglang-omni/issues/2273)（MiniCPM-o 4.5 runtime profiling）。
> - 性质：**代码审查得出的候选方案，尚无 GPU 实测数据**。按 #2284 的原则，任何默认值变更都要先有 profiling 和数值验证。

## 0. 速览

| 阶段 | 工作项 | 数值影响 | 风险 | 参考实现 |
|---|---|---|---|---|
| P0 | vision 负载基线 + 事件/NVTX + 核实视频是否需要 temporal resampler | 无 | 低 | #2273 方法，#2320 的事件命名 |
| P1 | 去掉 vpm embeddings / resampler 里每个 slice 的 host 同步；输入 H2D 合并 | **逐位一致** | 低 | Ming `vision_encoder.py`，Qwen3 `vision_compat.py` |
| P2 | 跨请求 batching + 按 patch 预算分块 + 批内去重；cache 改造 | 需在容差内一致 | 中 | Qwen3 `batch_image_encoder_payloads` |
| P3 | patch embed 换成 Linear、resampler 改变长注意力、跳过 padding 上的计算、CUDA graph | 需在容差内一致 | 中–高 | Qwen3 #132、Ming #539、sglang 上游 Qwen3-VL |
| P4 | 视频预处理不再绕道 PIL；preprocessing 多线程；image encoder 独立进程 | 视实现而定 | 中 | Qwen3 `placement.py`、#2316 |

## 1. 现状

### 1.1 数据流

```
preprocessing (pipeline 进程)
  ├─ 图像: ensure_image_list_async → HF processor 切 slice（图像默认多 slice）
  ├─ 视频: 解码为 (T,C,H,W) tensor → video_to_images() 逐帧转 PIL → HF processor
  │        （视频使用 max_slice_nums=1, use_image_id=False）
  └─ encoder_inputs["image_encoder"] = {pixel_values: list[Tensor], tgt_sizes, cache_key}
        │
image_encoder stage (pipeline 进程，与 thinker 同进程，SimpleScheduler 一次处理一个 payload)
  ├─ StageOutputCache(64 条 / 4 GiB, cache_device="cpu")
  └─ MiniCPMOImageEncoder.forward
       ├─ 每个 slice 单独 .to(device)，再 pad_sequence 成 (B, 3, P, L_max*P)
       ├─ patch_attn_mask 按全局最大 patch 数生成
       ├─ 按 vision_batch_size=16 个 slice 分块：
       │    vpm.embeddings(在 padding 后的 batch 上) → 按布尔 mask 打包 → 变长 encoder → 还原成 padding 形状
       └─ 按 16 个 slice 分块跑 Resampler2_5（nn.MultiheadAttention + key_padding_mask）
            → image_embeds: (num_slices * query_num, hidden)
        │
merge_for_thinker → thinker 按 bound 把 embedding 注入
```

### 1.2 与同类模型对比

| 能力 | MiniCPM-o | Qwen3-Omni | Ming-Omni |
|---|---|---|---|
| 跨请求 batching | 无 | 有（`max_batch_size=32` + 按字节成本限批） | 无 |
| 批内按 cache_key 去重 | 无 | 有 | 无 |
| cache 可观测（trace 开关） | 无 | `SGLANG_OMNI_TRACE_ENCODER_CACHE` | 无 cache |
| 元数据在 host 上一次算好 | 否（每个 slice 都同步） | 是 | 是（numpy 算 cu_seqlens，non_blocking 上传） |
| patch embed 用 Linear | 否（Conv2d） | 是（#132） | 是（#539） |
| image encoder 独立进程 | 否 | 可配置 | — |
| ViT CUDA graph | 无 | 无 | 无（注释说明跳过） |

### 1.3 证据缺口

#2273 的负载是 seed-tts-eval 英文语音克隆（文本 + 参考音频 → 语音），**不经过 image encoder**，模块表里也没有 vision 相关的行。因此 vision 相关的开销目前没有任何测量数据。#2273 测到 GPU busy 只有 22–33%，host 侧的 launch、同步和编排开销值得查。本文 P1 针对的正是这一类开销，但要在 vision 负载上重新测量。

## 2. 问题清单（代码审查）

### V-1　vpm embeddings 每个 slice 都有 host 同步

`image_encoder.py:203` 把 `tgt_sizes` 搬到 GPU 后传给 sglang `Idefics2VisionEmbeddings`。其 `get_position_ids`（sglang `idefics2.py:222-261`）逐个 slice 走 Python 循环：

- `nb_patches_h = tgt_sizes[i][0]` 是 GPU 上的 0 维 tensor。`torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)` 需要取出标量，会同步（h、w 各一次）。
- `position_ids[i][p_attn_mask.view(-1).cpu()]`：mask 在 GPU 上，每个 slice 一次 D2H 同步。

### V-2　打包和还原时用布尔 mask 索引

`image_encoder.py:170` 的 `embeds[valid]` 和 `:188` 的 `out[valid] = ...` 在 GPU 布尔 mask 上做索引，内部要调用 nonzero，每个 chunk 各同步一次。

### V-3　Resampler2_5 每个 slice 都有 host 同步

`image_encoder.py:253/260` 传的是 GPU 上的 `tgt_sizes`。sglang `minicpmv.py:307-360` 里：

- `_adjust_pos_cache` 调用 2 次 `.item()`，`patch_len.max().item()` 1 次，每个 chunk 共 3 次；
- 每个 slice 做 `tgt_sizes[i].tolist()`，并用 `key_padding_mask[i, patch_len[i]:]` 写 mask（GPU 标量当切片下标），每个 slice 约 2 次同步；
- pos_embed 逐个 slice 切出来，再 `pad_sequence`。

合计**每个 slice 约 5 次同步**（估算，需 nsys 或 sync debug 模式确认）。一段 64 帧的视频（每帧 1 个 slice）就是数百次同步。一张多 slice 的图像（最多约 10 个 slice）是几十次。

### V-4　输入 H2D 很碎

`image_encoder.py:206` 对每个 slice 从 pageable 内存单独 `.to(device)`。slice 数多时是大量同步的小拷贝。

### V-5　padding 浪费

- `patch_attn_mask`（`:221`）按**全局**最大 patch 数生成，分块后仍沿用，所以 embeddings 层的 conv 和位置编码是在 padding 后的形状上算的。
- vpm encoder 本身已经是变长 packing（这点做得好），但结果被还原成 padding 形状交给 resampler；resampler 的 `nn.MultiheadAttention` 在 padding 后的长度上计算。
- 分块按 slice 数（`vision_batch_size=16`，`:225`），不是按 patch 或 token 预算。长宽比悬殊的 slice 混在一块时，padding 比例会很高。

### V-6　没有跨请求 batching

`stages.py:88` 是 `SimpleScheduler(_encode_stage)`，一次只处理一个 payload。多个请求并发时，vision encoder 串行排队。#2320 只覆盖了 audio。

### V-7　encoder cache

- `stages.py:63-67`：`cache_device="cpu"`，没开 pinned memory。每次未命中都在关键路径上把输出**同步 D2H**；命中时拿到的是 CPU tensor，thinker 注入时还要再 H2D。
- cache key 粒度太粗：`preprocessor.py:278` 把图像 key 和视频 key `"|".join` 成一个，所以只有整组媒体完全相同才会命中。多轮对话里重复出现的同一张图、同一段视频无法复用。
- 缺少命中率数据，无法判断 cache 本身值不值（未命中时的 D2H 开销是否被命中收益抵消）。

### V-8　视频预处理多绕了一圈

`preprocessor.py:52-77` 的 `video_to_images` 把解码好的 tensor 逐帧转成 numpy 再转 PIL，HF processor 再做一遍 resize 和归一化。CPU 开销随帧数线性增长，而且这一切发生在与 thinker 同一个进程里。

### V-9　放置

`config.py:136-149`：preprocessing、image encoder、audio encoder、thinker、decode 同在 `"pipeline"` 进程，image encoder 复用 thinker 初始化的 TP group。vision 的 CPU 工作（V-1 到 V-4、V-8）会占用 thinker 调度循环所在进程的 GIL 和 CPU 时间。

### V-10　（待确认）视频是否应该用 temporal resampler

- 本实现使用 `Resampler2_5`，和 sglang 里 MiniCPM-o 2.6 的实现一致。
- sglang 里 MiniCPM-V 4.5 用的是 `Resampler4_5`：它通过 `temporal_ids` 做 3D 打包，把多帧合成一组 query token。
- `Resampler4_5` 多出来的 `temporal_pos_embed` 是**非持久 buffer**，不在 checkpoint 权重里，所以"权重能正常加载"**不能**证明 o-4.5 不需要它。
- 核实方法：
  1. 用 checkpoint 自带的 processor 处理一段视频，看输出里有没有 `temporal_ids`（当前 `preprocessor.py` 不读这个字段）；
  2. 对照官方 `modeling_minicpmo.py` 的视频路径。
- 如果确认需要，这首先是**正确性**问题（视频 token 数和语义都会变），优先级高于本文其余各项。

## 3. 方案

### P0　可观测性、基线与核实（必须先做）

1. **事件**：image encoder 发 `encoder_start/end`，以及和 #2320 同名的 `encoder_cache_hit/miss`、`encoder_forward_start/end`。metadata 带上 slice 数、有效 patch 数、padding 后的 patch 数、是否命中 cache。
2. **NVTX 范围**（和 #2273 一样用环境变量开关，不新增同步）：
   - 预处理：`preprocess.image_fetch`、`preprocess.video_decode`、`preprocess.video_to_pil`、`preprocess.hf_processor`；
   - encoder：`image_encoder.h2d`、`vpm.embeddings`、`vpm.encoder`、`vpm.unpack`、`resampler`、`cache_put`；
   - thinker 侧：`thinker.mm_inject`。
3. **cache trace**：照搬 Qwen3-Omni 的 `SGLANG_OMNI_TRACE_ENCODER_CACHE`，记录 hit/miss/store 和字节数。
4. **负载**（用 text 变体，排除语音链路的干扰；再补一组 speech 变体）：
   - 图像：`benchmarks/eval/benchmark_omni_mmmu.py`，多 slice；
   - 视频：`benchmark_omni_videomme.py`，按 32 帧和 64 帧分两档；
   - 视频 + 音频：`benchmark_omni_videoamme.py`；
   - 并发 c1 / c8 / c16。吞吐只用关闭 profiler 的运行，nsys 只用来做归因（#2273 测到 nsys 会带来 1.5–1.7× 扰动）。
5. **指标**：
   - E2E、TTFT；
   - image_encoder 的排队时间和执行时间；
   - GPU busy；
   - **每条请求的 `cudaStreamSynchronize` 和 D2H 次数**；
   - slice 数和 patch 数的分布、padding 比例、cache 命中率；
   - 准确率（MMMU 和 Video-MME 分数）作为数值护栏。
6. **核实 V-10**。

**产出**：在 #2273 下补一节 "vision workload"，或者单独开 #1798 的子 issue，并链到 #2284 第 2、4 项。

### P1　去掉同步，合并 H2D（逐位一致）　→ PR-A

1. **元数据在 host 上一次算好**（对应 V-1、V-2、V-3）。在 `image_encoder.py` 里实现本地函数，不 monkey-patch sglang：
   - 用 `tgt_sizes_cpu` 在 CPU 上一次性算出：
     - 所有 slice 的 `position_ids`（等价于 `get_position_ids` 的 bucketize 逻辑）；
     - 打包用的 gather 下标和还原用的 scatter 下标（代替布尔 mask 索引）；
     - resampler 的 `key_padding_mask`，以及 pos_embed 的下标（从 2D pos cache 按 h×w 取 index）；
     - `_adjust_pos_cache` 需要的 max_h、max_w。
   - 这些都放进 pinned tensor，一次 `non_blocking` 上传。
   - embeddings 的 forward 改为 `patch_embedding` 加上用预计算 `position_ids` 查出的位置编码；resampler 的 forward 改为用预计算的 mask 和 pos_embed，内部计算不变。
   - 参考：Ming `vision_encoder.py`（`grid_thw.tolist()` 只调一次，cu_seqlens 用 numpy 算后 non_blocking 上传），Qwen3 `vision_compat.py`。
2. **输入 H2D 合并**（对应 V-4）：在 host 上先把各 slice 拼进一块 pinned 暂存区，一次拷贝，再在 GPU 上 reshape 或 pad。更进一步，可以让 preprocessing 直接产出打包好的扁平 tensor 加 offsets，但这会改 payload 格式，放到 P2 一起做。
3. **验收**：
   - 固定输入（不同长宽比、多 slice 图像、视频帧、slice 数超过 16 触发分块）下，`image_embeds` 与现实现**逐位相等**；
   - 在 `torch.cuda.set_sync_debug_mode("error")` 下跑 forward，确认除了一次必要的同步外没有其他同步（写成 GPU 单测）；
   - P0 负载下对比每条请求的同步次数和 image_encoder 执行时间。

### P2　stage 层：batching、分块、cache　→ PR-B、PR-C

1. **跨请求 batching**（对应 V-6）：
   - 照 Qwen3 的 `batch_image_encoder_payloads` 写法：`batch_compute_fn` 加 `request_cost_fn`（按有效 patch 数或字节估算）加 `max_batch_cost`，`max_batch_wait_ms` 默认为 0。
   - 批内按 cache_key 去重（一个 leader，其余等结果）。
   - 输出按每个请求的 `num_slices * query_num` 切回去。
   - 结构上与 #2320 的 `audio_encoder_batching.py` 对齐；可以考虑抽一个 audio 和 image 共用的 encoder batching 骨架，但抽不抽以两边合入的先后为准，不要预先抽象。
2. **按 patch 预算分块**（对应 V-5）：slice 按 patch 数排序后按 patch 预算分块，每块只 pad 到本块的最大值；每块内的 patch mask 单独生成。
3. **cache 改造**（对应 V-7，先看 P0 的命中率再决定）：
   - 命中率低：未命中时的 put 改成 pinned + side stream 上的 `non_blocking` 拷贝，不阻塞关键路径；或者直接关掉 cache。
   - 命中率高：考虑加一层小的 GPU LRU，省掉命中时的 H2D；显存从 thinker 的预算里扣。
   - **按单个媒体做 key**：每张图、每段视频单独一个 key，输出按 slice 边界切开再缓存。每个媒体对应多少个 slice 可以从 processor 输出的 `pixel_values` 嵌套结构里拿到。
4. **验收**：
   - batch 后单个请求的 embedding 与单独运行时在容差内一致。cuBLAS 的结果可能随 batch 形状变化，不保证逐位一致；以 cosine 或 max-abs 加下游准确率来判断。
   - c8 / c16 下 MMMU 和 Video-MME 的吞吐和 p95、p99。
   - 视频负载下的峰值显存，确认不挤压同 GPU 上 thinker 和 talker 的静态显存。

### P3　计算优化（逐项验证，按 profiling 结果决定做不做）　→ PR-D*

1. **patch embed 换成 Linear**：Idefics2 的 `patch_embedding` 是 kernel=stride 的 Conv2d，理论上等价于 reshape 后乘矩阵。MiniCPM 的像素排布是 `(3, P, L*P)`，需要单独推导 reshape。Qwen3 的 7–15× 收益来自 Conv3d 的慢速 cuDNN 路径，Conv2d 未必有同等收益，要实测。
2. **resampler 改用变长注意力**：把 `nn.MultiheadAttention` 加 padding 改成 SDPA 或变长 flash 交叉注意力（每个 slice 的 Q 固定为 query_num 个，K 用各自的有效长度）。不是逐位一致，要做准确率护栏。
3. **不在 padding 上算 embeddings**：只对有效 patch 做 patch embed 和位置编码，结果直接交给打包后的 encoder，省掉 V-5 里 padding 部分的计算和一次打包。
4. **CUDA graph**：vpm 和 resampler 按 patch 数分桶捕获。仓库内没有现成实现，可以参考 sglang 上游 Qwen3-VL 的 vision CUDA graph runner。只有在 P1 做完后，profiling 仍然显示 vision 受 launch 限制时才做。

### P4　预处理与放置　→ PR-E、PR-F

1. **视频不再绕道 PIL**（对应 V-8）：解码后的帧 tensor 直接走 torch 的 resize 和归一化，或者使用 processor 的 tensor 输入路径。PIL 和 torch 的 resize 数值不同，要和 HF processor 的输出做容差比对，并以 Video-MME 分数作为护栏。
2. **preprocessing 多线程**（#2316）：Qwen3-Omni 默认串行，原因见其 `stages.py` 里的注释：多线程会改变 thinker 的 batch 组成，bf16 贪心解码的答案随之漂移，Video-MME CI 因此翻过车。MiniCPM-o 开多线程前，要在 MMMU 和 Video-MME 上做同样的准确率检查。
3. **image encoder 独立进程**（对应 V-9，#2284 第 12 项）：
   - `init_sglang_tp()`（`image_encoder.py:33`）已经支持在未初始化时自建 TP=1 的 group，所以拆进程可能主要是 config 和显存预算的改动。
   - 需要处理的是：GPU 显存预算要从 thinker 的 `mem_fraction_static` 里留出来；encoder 输出通过 CUDA IPC 或 relay 交给 thinker 的开销。
   - 在 c1 和 c16 下比较同进程与拆进程，参考 Qwen3-Omni 的 `process_by_stage` 和 `placement.py`。

## 4. 验收标准（通用）

- **数值**：P1 必须逐位一致。P2 到 P4 对 embedding 设 cosine 或 max-abs 容差，并且 MMMU、Video-MME 的分数不低于基线的噪声范围（贪心解码，固定顺序，至少跑 3 次）。
- **性能**：沿用 #2273 的方法。固定数据集和顺序；吞吐只用关闭 profiler 的运行；nsys 只用于归因；报告 GPU busy、每条请求的同步和拷贝次数、p50、p95、p99。
- **显存**：c16 视频负载下的峰值显存，以及和 thinker、talker 同卡时有没有 OOM。
- **每个 PR** 挂回 #2284 的对应条目，并附上前后对比数据。

## 5. PR 拆分与顺序

| PR | 内容 | 依赖 | 数值 | #2284 条目 |
|---|---|---|---|---|
| PR-0 | 事件、NVTX、cache trace；vision 基线报告；核实 V-10 | — | 无变化 | 2、4 |
| PR-A | P1：去掉同步 + 合并 H2D | PR-0（用它做前后对比） | 逐位一致 | 2 |
| PR-B | P2-1、P2-2：跨请求 batching + 按 patch 预算分块 | PR-A | 容差 | 2 |
| PR-C | P2-3：cache 改造（按单个媒体做 key、异步 put 或 GPU 层） | PR-0 的命中率数据 | 逐位一致 | 4 |
| PR-D* | P3 各项，一项一个 PR | PR-A、PR-B | 容差 | 2 |
| PR-E | P4-1、P4-2：视频预处理、多线程 | PR-0 | 容差 | 1 |
| PR-F | P4-3：image encoder 独立进程 | PR-B | 无变化 | 12 |
| （条件）| V-10 视频 temporal resampler | PR-0 的核实结论 | 语义变化 | 2 |

## 6. 风险与开放问题

1. **V-10** 如果成立，视频路径的 token 数和语义都会变，P2 的成本模型和 P3 的分桶都要跟着改，所以应该最先核实。
2. **跨请求 batching 与数值漂移**：batch 形状变了，GEMM 的结果可能不逐位一致；再叠加 thinker 批次变化，贪心解码的答案可能翻转（Qwen3 Video-MME 的先例）。
3. **显存**：同卡同进程时，batching、GPU cache 和 thinker 的 KV cache 互相争显存，需要一个明确的 encoder 显存预算。
4. **视频帧数上限**：`video_max_frames` 和总像素数决定一次视频请求的 slice 数上限，也就决定了 P2 成本模型的上限和 P3 的分桶范围。
5. **收益是否值得**：如果 P0 显示 vision 在端到端里占比很小（例如短文本问答里单张图的场景），P2 到 P4 可以降级，只保留 P1。

## 7. 参考

- MiniCPM-o：
  - `sglang_omni/models/minicpm_o/components/image_encoder.py`
  - `sglang_omni/models/minicpm_o/components/preprocessor.py`
  - `sglang_omni/models/minicpm_o/stages.py`
  - `sglang_omni/models/minicpm_o/config.py`
- sglang 0.5.20：`sglang/srt/models/idefics2.py`（`Idefics2VisionEmbeddings`）、`sglang/srt/models/minicpmv.py`（`Resampler2_5`、`Resampler4_5`）
- Qwen3-Omni：
  - `sglang_omni/models/qwen3_omni/stages.py`（`batch_image_encoder_payloads`、`create_image_encoder_request_cost_fn`、cache trace、preprocessing 串行的注释）
  - `sglang_omni/models/qwen3_omni/components/image_encoder.py`（`optimize_patch_embed`，#132）
  - `sglang_omni/models/qwen3_omni/components/vision_compat.py`
  - `sglang_omni/models/qwen3_omni/placement.py`
- Ming-Omni：`sglang_omni/models/ming_omni/components/vision_encoder.py`（host 侧元数据、`linear_patch_embed`，#539）
- 相关 PR：#2316（preprocessing 多线程）、#2320（audio encoder batching 与事件命名）、#1904 / #1972（旧的事件与 encoder batching 实现，已过时）
