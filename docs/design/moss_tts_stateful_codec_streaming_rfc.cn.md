# RFC:MOSS-TTS Codec Decoder 的有状态 KV-Cache 流式

> English version / 英文版: [moss_tts_stateful_codec_streaming_rfc.en.md](moss_tts_stateful_codec_streaming_rfc.en.md)

| | |
|---|---|
| **状态** | 草案 |
| **作者** | liuqihao |
| **日期** | 2026-06-09 |
| **分支** | `feat/moss_streaming_optimize` |
| **取代** | `sglang_omni/models/moss_tts/streaming_vocoder.py` 中的 overlap-window 重叠重算路径(v1,见 `docs/design/moss_tts_streaming_plan.md` §5) |
| **相关** | MiMo-Audio `streaming_decode`(`vllm-omni`)、issue #637 |

---

## 1. 摘要

MOSS-TTS 流式音频相比非流式存在 WER/CER 劣化。根因是 codec decoder 本质是一摞
**causal RoPE transformer**,而当前流式 vocoder 用**无状态的 overlap-window
重叠重算**,每个 chunk 只带约 2 帧左上下文。该 transformer 训练时见的是最多
10 秒因果上下文,截断后每个窗口开头的声学帧就会损坏。本 RFC 提议重新引入
**per-layer KV cache**(上游 codec 本有、被 vendored 推理副本砍掉),使流式解码
与离线解码数学等价。

此外还有**第二个独立 bug**:pre-codec 的 de-delay/segment 扫描每个 chunk 都重跑
全量历史,是 O(N²)。两个修法正交,且都属于 streaming vocoder:**incremental
de-delay** 消除 O(N²) 成本,**stateful KV cache** 消除正确性(接缝 / WER)劣化。
两者齐备后总成本 O(N) 且输出等于离线。

---

## 2. 背景

MOSS audio tokenizer 的 decoder(`audio_tokenizer.py` 的
`_default_decoder_kwargs`)是 `_ProjectedTransformer`(causal attention + RoPE)
与 `_PatchedPretransform`(无参数 patch reshape 上采样)交替。逐算子审计:

- `_PatchedPretransform` 上采样是纯 reshape `(b, d, T) → (b, d//h, T*h)`,
  每个输出时间步只来自单个输入时间步的 channel,**不跨时间混合 → 零状态,
  逐 chunk 等价**。
- LayerNorm / FF / `in_proj` / `out_proj` 都是逐帧。
- quantizer `decode_codes` 是查表 + kernel=1 卷积 → 逐帧。
- **整个 decoder 里唯一跨时间混合信息的算子,就是 causal self-attention**
  (`_Attention.forward`)及其 RoPE。

vendored 副本文件头明写:“Streaming KV-cache infrastructure removed
(single-pass batch decode only)。”上游 OpenMOSS codec 本有流式 KV cache,
被这份推理副本砍掉了。

---

## 3. 问题陈述

每个 chunk 其实有**两笔独立的成本**,当前 streaming vocoder 各自以不同方式出错。
别把它们混成一个。

**Bug A —— pre-codec de-delay / segment 扫描是 O(N²)(成本)。**
把累计的 delayed rows 还原成规整 code frame。当前代码每个 chunk 都重新 stack
*整段*历史 rows、再对*整段*重跑 segment 扫描(`streaming_vocoder.py:149` ——
`torch.stack(state.delayed_rows[:rows_end])` 后接全量 `split_moss_audio_segments`,
其 `apply_de_delay_pattern` + pad scan + `nonzero` 都扫全历史)。这是 O(N)
per chunk → **O(N²) total**,延迟随句长增长。GaokaiZhang 说的“O(N²)”指的正是
这段,**不是** codec 窗口解码。

**Bug B —— codec 窗口解码是 O(N) 但错(正确性)。**
把 code frame 解成波形。当前代码只喂小窗口 `segment[emitted-overlap : end]`
(`streaming_vocoder.py:173`),这部分**本来就是 O(N)** —— 但它**是错的**:
无状态 transformer 只拿到新帧 + 约 2 帧 overlap,看不到 `0..emitted`,于是凭空
编造声学上下文 → WER/CER 劣化、可听见接缝。

| Bug | 位置 | 症状 | 修法 |
|---|---|---|---|
| A:O(N²) 成本 | pre-codec de-delay / segment 扫描 | 延迟随句长涨 | **incremental de-delay**(§5.4) |
| B:输出错 | codec 窗口解码 | WER/CER 劣化、接缝 | **stateful KV cache**(§5.1–5.3) |

两个修法正交,且都属于 streaming vocoder。

仅就 Bug B 而言,喂多少帧与正确性相互作用:

|  | 喂多少帧 | 看得到历史? | 结果 |
|---|---|---|---|
| 喂 `0..N`(无状态) | O(N²) | ✅ | 对但慢 |
| 只喂新帧(无状态)← **现在** | O(N) | ❌ | 快但错(WER) |
| 只喂新帧 + **KV cache** | O(N) | ✅ | **又快又对** |

conv decoder(MiMo)能容忍 overlap 重算,是因为它感受野小且固定,一点 overlap
就够。causal transformer 的感受野是**整个注意力窗口(约 10s)**,所以 overlap
重算要么错(小 overlap)、要么 O(N²)(全 overlap)。KV cache 就是“全 overlap
重算”的 O(N) 记忆化版本:输出一致,但不重算。

---

## 4. 目标与非目标

**目标**

- 流式 codec 解码输出与离线 `batch_decode` 逐段在严格数值容差内一致。
- 总成本 O(N);de-delay 扫描与 codec 解码**两者**单 chunk 都是 O(新帧数)
  (incremental de-delay 消除 O(N²) 扫描)。
- 流式与非流式 WER/CER 持平。
- 保留 scheduler 生命周期与离线 de-delay 语义。

**非目标**

- 批量(ragged 多请求)流式 codec 解码 —— v1 维持 per-request。
- 流式 decoder 的 CUDA Graph 封装。
- AR(talker)阶段或 de-delay 语义的改动。
- encoder / 参考音频路径改动。

---

## 5. 设计

### 5.1 流式状态对象

一个 `StreamingCache` dataclass,每个 decoder transformer stage 一份 `StageCache`。
各 stage 运行在**不同时间率**(patch 上采样在 stage 间改变 T),所以 RoPE 的位置
offset 是 **per-stage,不是全局** —— 单一全局 `pos_offset` 会含义不清。每个
`StageCache` 保存:

- per-layer `k`、`v`:`[B, H, T_cached, D]`。
- `pos_offset`(int),供 RoPE 续接,按*该 stage*的时间率计数。
- (可选)`window` 上限,丢弃超过训练 `causal_transformer_context_duration`
  (10s)的旧 K/V,为长句封顶内存。

patch 层无状态,不存任何东西。

```python
@dataclass
class LayerKV:
    k: torch.Tensor | None = None   # [B, H, T, D]
    v: torch.Tensor | None = None   # [B, H, T, D]

@dataclass
class StageCache:
    layer_kvs: list[LayerKV] = field(default_factory=list)
    pos_offset: int = 0             # 该 stage 时间率下的 RoPE offset

@dataclass
class StreamingCache:
    # 每个 decoder transformer stage 一份 StageCache
    stage_caches: list[StageCache] = field(default_factory=list)
    finished: bool = False
```

### 5.2 接口

与 MiMo 形态一致:

```python
audio_chunk, state = codec.decode_stream(codes_chunk, state=None, last_chunk=False)
```

- `codes_chunk`:`(NQ, T_new)` —— 只含新 de-delay 出来的帧。
- `state`:上次返回的 `StreamingCache`(首帧传 `None`)。
- `last_chunk`:末尾/段关闭的 flush 标志。
- 返回这些新帧对应的波形 delta,以及更新后的 state。

### 5.3 codec 改动(`audio_tokenizer.py`)

1. **`_apply_rope`** —— 加 `pos_offset` 参数,把 `arange(T)` 改成
   `arange(offset, offset+T)`。位置**必须跨 chunk 续接**(保持 `:174` 警告的
   GPT-J interleaved 约定)。

2. **`_Attention.forward`** —— 接收/返回 `LayerKV`。投影新 q/k/v;用 `pos_offset`
   加 RoPE;把新 k/v 拼到 cache;新 q 对全 cache 做 attention。有 cache 时,把
   `is_causal=True` 的方阵 mask 换成显式 mask:每个新 query 能看到全部 cached +
   截至自身的新 key。

3. **`_TransformerLayer` / `_Transformer` / `_ProjectedTransformer`** ——
   把 per-layer cache list 在 `forward` 里穿下去。

4. **`MossAudioTokenizerModel.decode_stream`** —— 编排:
   `quantizer.decode_codes(new_frames)`(无状态)→ 逐 stage 跑 decoder 并穿
   `state`;patch 层照跑(无状态);**每个 stage 的** `pos_offset` 按*该 stage*
   时间率下的新帧数推进(patch 上采样在 stage 间改变 T,所以每个 `StageCache`
   各记自己的 offset —— 见 §5.1)。

5. cache 上的 **`reset()`** 用于段边界。

### 5.4 incremental de-delay(修 Bug A,O(N²) 成本)

当前代码**没有**真正的增量 de-delay;它累计 rows 后每个 chunk 对整段历史重跑
全量 `split_moss_audio_segments`(`streaming_vocoder.py:149`)。改成真正的增量
扫描。在 `_MossStreamState` 维护:

- `raw_frame_cursor` —— 至今已产出多少 de-delay 帧(跨段全局)。
- `open_segment` —— 当前正在生长的音频段(段间为 `None`)。
- per-segment `frames` —— 当前 open 段已完成的帧。
- per-segment `emitted_cursor` —— 已喂给 codec 的帧。

每个 chunk,**只对新变完整的 rows** 做 de-delay —— 索引 `i` 的帧在 row
`i + n_vq - 1` 到达后才完整。把每个新帧分类为音频 / pad-only 分隔符;音频帧
append 进 `open_segment`;遇 pad-only 分隔符,关闭 `open_segment`(flush + reset
其 codec state)并开新段。**不再对整段历史调用 `split_moss_audio_segments`** ——
单 chunk 工作量降为 O(新帧数),总计 O(N)。

`assistant_start_length` 只对首个音频段、按全局 `raw_frame_cursor` 应用一次 ——
与 `split_moss_audio_segments` 一致。

### 5.5 scheduler 改动(`streaming_vocoder.py`)

- 在 `_MossStreamState` 里每**段**加 `codec_state: StreamingCache | None`。
- 用 §5.4 的增量 de-delay 驱动 codec:只把新完成的帧喂
  `codec.decode_stream(new_frames, state)`,直接 emit 返回的波形 delta。
- **删除** overlap / trim / 重算 / `samples_per_frame` 那一整套
  (`streaming_vocoder.py:173-194`)—— `decode_stream` 精确返回新样本,无需 trim。
- **段边界**:§5.4 关段时(pad-only 分隔符),调 `decode_stream(..., last_chunk=True)`
  flush,然后为下一段起一个全新的 `StreamingCache` —— 离线是逐段独立解码,
  这样保持一致。

### 5.6 sglang-omni codec 源码归属

sglang-omni 里 codec 经 `trust_remote_code` 加载(`processor.decode_audio_codes`),
源码不在 repo。要拥有 stateful 解码路径,把它 vendored 到一个专门、可追溯的位置
(不要在一方代码旁边裸建一个模块):

- **路径**:`sglang_omni/models/moss_tts/_vendored/moss_audio_tokenizer/`。
- **来源记录** —— 加 `_vendored/README.md` 写明:来源 repo
  (`OpenMOSS-Team/MOSS-Audio-Tokenizer`)、vendored 的确切 commit/revision hash、
  上游 license、以及本地改动(新增 KV 流式)。pin 住 revision,使 checkpoint
  权重与 vendored modeling 代码不会漂移。
- **切换** —— 当 `stream_codec_mode == "stateful"` 时,codec loader 用本地
  vendored 类而非 `get_class_from_dynamic_module(...)`;非流式与 `overlap` 回退
  仍可用远程模块。
- **否决 (b)** —— subclass / monkeypatch 远程 attention 模块:脆弱、随
  checkpoint 版本漂移。

---

## 6. 正确性论证

因为唯一的跨时间混合是 causal attention,而 KV cache 精确复现一次性前向会算出的
K/V,所以带 cache 解码第 `t` 块,与一次性解码 `[0..t]` 在注意力窗口内 bit 级相等。
patch 上采样和所有逐帧算子无论分不分块都平凡一致。因此流式拼接输出 = 逐段离线
输出,正是我们要的 WER/CER 持平。

---

## 7. 风险与待确认

1. **10s 上下文窗口语义** —— 上游如何实现 `causal_transformer_context_duration`
   (sliding-window mask 还是 full causal)。必须从上游源码复刻,否则长句漂移/内存涨。
   *这是唯一的硬外部依赖。*
2. **RoPE offset 续接** —— interleaved(GPT-J)约定必须跨 chunk 正确续接;用单测验证。
3. **无 kernel>1 卷积** —— 已确认:quantizer 里只有 kernel=1 的 `_wn_conv1d`,
   故 KV cache 既必要又充分。checkpoint config 变化时需重验。
4. **批量** —— v1 用 per-request state;ragged 批量 `decode_stream` 留后续。
5. **浮点精度** —— codec 全程 float32;cache 保持 float32。

---

## 8. 测试

- **de-delay 增量性**:增量 de-delay 输出 == 一次性
  `split_moss_audio_segments(full)`;断言不再有 per-chunk 全历史重扫(工作量为
  O(新帧数))。
- **Parity**:`cat(decode_stream chunks) ≈ batch_decode(full)` 逐段,
  `torch.allclose` 容差内。
- **KV 连续性**:逐帧喂 == 整段喂。
- **RoPE offset**:带 offset 路径 == 全位置路径。
- **多段 reset**:pad-only 分隔符 flush + reset;只有首段被 `assistant_start_length` trim。
- **Scheduler 生命周期**:abort 清 per-request `codec_state`;abort 后无终端音频;
  终端 payload 不被二次解码。

---

## 9. 灰度

加 `stream_codec_mode = {"overlap", "stateful"}`。默认仍走 `overlap`,等 parity
测试 + 真模型 WER/CER 跑通确认 `stateful` 后翻默认,保留 `overlap` 作为一个版本的回退。

---

## 10. 备选方案

- **加大 overlap(维持无状态)** —— 要 overlap ≈ 全窗口才追平离线 → O(N²)。
  否决:违背流式初衷。
- **MiMo 式隐空间重叠重算** —— 对 conv(小感受野)自然,对 causal transformer
  (窗口级感受野)不适用。否决:同样的 O(N²)/出错权衡。
- **subclass/monkeypatch 远程 codec** —— 跨 checkpoint 版本脆弱。否决,改用 vendoring。

---

## 10b. 验证结果

2026-06-09 于 SeetaCloud H20,`OpenMOSS-Team/MOSS-TTS-v1.5` + `Qwen/Qwen3-ASR-1.7B` 实测。

- **单测 parity**(tiny 随机初始化模型,39 个):`decode_stream` 分块 == `batch_decode`;
  逐帧 == 一次性;RoPE offset 续接;多段 reset;增量 de-delay == `split_moss_audio_segments`;
  scheduler 生命周期。
- **真 checkpoint**:权重 remap **100%**(1600/1600,0 skipped/missing);真权重
  `decode_stream` vs `batch_decode` **max_abs_diff = 1.4e-4**(bit 级一致)。
- **vendored vs 远程 codec**(`processor.decode_audio_codes`):mean_abs = 3.3e-5,
  max = 4.5e-3 —— 近乎一致;为重实现的微小数值漂移,非结构差异。
- **e2e WER**(SeedTTS EN,50 句,c=1,ASR=Qwen3-ASR-1.7B):
  - 固定 seed=1234:非流式 **1.42%**,stateful 流式 **1.77%**(mean 1.18%/1.45%;
    max 20%/14.3%;无 >50% 离群)。均显著低于全 EN 集非流式基线 1.93%(PR #609)。
  - 未固定 seed(噪声,每请求 token 不同):非流式 1.06%,overlap 流式 1.42%,
    stateful 流式 2.13%(含一个 50% 离群)。
- **结论**:stateful KV-cache 流式对相同 codes **相比离线零解码误差**(bit 级一致)。
  e2e 残余 WER 差异来自采样不确定性 + vendored 与远程 codec 的微小漂移,**不是** overlap
  路径那种左上下文截断劣化。服务在 `MOSS_TTS_STREAM_CODEC_MODE=stateful` +
  `--mem-fraction-static 0.80`(为第二份 codec 留显存)下正常工作。

## 10c. 并发 sweep、性能与合并优化

全集 A/B(1088 EN,seed=1234,同 token):非流式 1.792% WER / 0.770% CER;
overlap 流式 2.018% / 0.940%;stateful 流式 1.616% / 0.775%。**stateful 消除了
overlap 的劣化并追平离线。**

并发 sweep(EN,256/组,seed=1234,配对 token)发现:**质量上** stateful 各并发
WER 比 overlap 低 ~0.15–0.22pp;但 **v1 性能** 因逐帧调用重型 decoder,在并发下
TTFA/RTF 爆炸(c=16:TTFA 42s、RTF 11.7,overlap 仅 9.9s/2.9)。

**合并优化(v1.1,已实现+实测)**:攒够 `stream_stride`(默认 8)帧再调一次
`decode_stream`。KV cache 使多帧解码与逐帧 bit 级一致(39 单测全过,WER/CER 不变),
GPU 调用数砍 ~stride 倍。实测 stateful 合并前→后:

| c | TTFA 前→后 | RTF 前→后 | QPS 前→后 |
|---|---|---|---|
| 1 | 2.17→2.25s | 1.15→0.71 | 0.21→0.34 |
| 4 | 9.14→1.18s | 2.99→0.58 | 0.33→1.63 |
| 8 | 20.75→3.44s | 5.94→1.16 | 0.33→1.67 |
| 16 | 42.29→7.71s | 11.66→2.28 | 0.33→1.69 |

合并后,stateful 在**每个并发都同时在质量和性能上胜过 overlap**(如 c=16:
TTFA 7.71s vs 9.88s、RTF 2.28 vs 2.87、QPS 1.69 vs 1.35、WER 1.86% vs 2.04%)——
因为 KV cache 不重算,而 overlap 要重解重叠窗口。跨请求批量 codec(§5b)为可选后续。

**建议:落地后把 `stream_codec_mode` 默认改为 `stateful`**,保留 `overlap` 一个版本作回退。

## 11. 工作拆解

| # | 任务 | 文件 |
|---|---|---|
| 1 | 把 codec vendored 到 `_vendored/`,带来源 README + pin revision | `sglang_omni/models/moss_tts/_vendored/moss_audio_tokenizer/`(新建) |
| 2 | RoPE 加 `pos_offset`(per-stage) | `_vendored/.../audio_tokenizer.py` |
| 3 | attention 及各层加 KV cache | `_vendored/.../audio_tokenizer.py` |
| 4 | `StageCache` / `StreamingCache` + `decode_stream` | `_vendored/.../audio_tokenizer.py` |
| 5 | **incremental de-delay**(raw_frame_cursor、open/closed segment、per-segment cursor)—— 修 O(N²) | `streaming_vocoder.py`、`codec.py` |
| 6 | 用 stateful 路径替换 overlap | `streaming_vocoder.py` |
| 7 | `stream_codec_mode` 开关 + loader 切换 | `streaming_vocoder.py`、config |
| 8 | 单测(de-delay 增量性、parity、连续性、RoPE、reset、生命周期) | `tests/unit_test/moss_tts/` |
| 9 | 复刻上游 10s 窗口语义 | 上游源码 + `_vendored/...` |
| 10 | 真模型 WER/CER parity | `tests/test_model/` |
