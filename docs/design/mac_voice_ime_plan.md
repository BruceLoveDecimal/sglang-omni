# Mac 本地语音输入法开发方案（基于 SGLang-Omni MLX 后端）

> 状态：调研 / 设计稿。基线分支 `feat/nemotron_asr_0.6B_mlx`（`5442d95`）。
> 目标：在 Apple Silicon 上，用本仓库的 Nemotron 3.5 ASR Streaming 0.6B（MLX）作为引擎，
> 复用仓库原生的 `/v1/realtime` 流式协议与 Silero 服务端 VAD，实现一个"按住说话 / 免提听写"
> 的 macOS 语音输入工具，最终可选升级为真正的 InputMethodKit 输入法。

---

## 1. 结论先行

1. **引擎层现状**：仓库里 Nemotron MLX 是纯离线整段推理（一次 `encode` + 一次 `decode`），没有跨块缓存，
   也没有接入 `/v1/realtime`。但模型编码器是"按 chunk 因果"的（chunk = lookahead+1 个 80 ms 帧，
   左侧滑窗 70 帧，卷积全部左填充），特征提取不做整句归一化，且 `hf_compat` 的 Torch 路径已经实现了完整的
   cache-aware 流式（conv padding cache + 注意力滑窗 KV cache + 固定 chunk 帧数 + LSTM 状态）。
   **流式 MLX 移植有现成参考，工程风险可控。**
2. **服务层现状**：`/v1/realtime?intent=transcription` WebSocket 协议、Silero VAD 端点检测、PCM 环形缓冲、
   分段/提交/清空语义都已存在，且不依赖 CUDA。目前只有 Qwen3-ASR 声明了 `realtime_transcription`。
   **客户端只需对接这一个协议，引擎（Nemotron / Qwen3-ASR）可互换。**
3. **客户端结论**：第三方 Mac 听写工具（VoiceInk、Handy、Superwhisper、Wispr Flow…）几乎全部采用
   "全局热键 + 剪贴板粘贴 Cmd+V"，松键后才出文字；只有系统听写用真正的 marked text。
   推荐路线：**先做 菜单栏 App + Fn 按住说话 + 悬浮窗实时显示 partial + 松键粘贴**，
   再把"真 IMK 输入法（marked text 原地合成）"作为可选的高级里程碑。
4. **主要风险**：Nemotron 对普通话的 CER 约 19%（官方卡片，1.12 s chunk），英文强、中文弱。
   方案中保留 Qwen3-ASR MLX 作为同协议的备选引擎，用真实数据做 A/B 后再定默认。

---

## 2. 调研摘要

### 2.1 仓库已有能力（可直接复用）

| 能力 | 位置 | 备注 |
|---|---|---|
| Realtime WS 端点 | `sglang_omni/serve/openai_api.py:1231`，`--enable-realtime` 开启 | `?intent=transcription` |
| 转写会话状态机 | `sglang_omni/serve/realtime/transcription_session.py` | append / commit / clear / done；`transcription.segment{segment_id,text,is_final}`（全量替换）、`transcription.completed` |
| 事件 schema | `sglang_omni/serve/realtime/events.py` | OpenAI Realtime 兼容子集 |
| 服务端 VAD | `sglang_omni/serve/realtime/vad.py` | Silero ONNX，512 样本/32 ms 帧，`threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500`；CPU，可在 Mac 运行 |
| PCM 缓冲 | `sglang_omni/serve/realtime/audio_buffer.py` | 上限 `max_audio_clip_s + 4` 秒 |
| 流式策略接口 | `StreamingASRStrategy`（`transcription_session.py:46`） | `create_state / build_decode_request / update_hypothesis` |
| 现有实现示例 | `sglang_omni/models/qwen3_asr/streaming.py` | 周期性全段重解码 + 5 token 回滚前缀 |
| 协议测试 | `tests/unit_test/serve/test_realtime_*.py`，`tests/test_model/test_qwen3_asr_realtime.py` | 后者是最好的 Python WS 客户端参考 |
| Apple 平台 | `sglang_omni/platforms/apple.py`，`SGLANG_USE_MLX=1`，`install.sh` → `.venv-apple` | 单设备、无 TP |
| Nemotron MLX | `sglang_omni/models/nemotron3_5_asr/mlx/{model,runner}.py` | FP32、greedy、batch=1、≤60 s |
| Nemotron 流式参考实现 | `hf_compat/modeling_nemotron_asr_streaming.py`、`generation_nemotron_asr_streaming.py` | Torch，未被服务路径使用 |

仓库**没有**的：麦克风客户端（Python/Swift 均无）、Nemotron 的 realtime 声明、任何跨块模型缓存的会话调度。

### 2.2 Nemotron 3.5 ASR Streaming 关键参数

| 项 | 值 |
|---|---|
| 输入 | 16 kHz 单声道，80 维 log-mel，hop 160（10 ms），预加重 0.97，**无归一化** |
| 下采样 | 8×（3 层因果 Conv2d，stride 2） → 编码器帧 80 ms |
| 注意力 | 左上下文 70 帧（`sliding_window=71`），右上下文 = lookahead ∈ {0,3,6,13} |
| chunk 时长 | lookahead 0/3/6/13 → 80 / 320 / 560 / 1120 ms |
| 首块 mel 帧数 | `1 + 8*lookahead`；后续块 `8*(lookahead+1)` |
| 解码 | 贪心 RNN-T，`max_symbols_per_step=10`，LSTM 预测网络 |
| 输出特性 | token 一旦发出**不会被修正**（单调），自带标点与大小写；`language=auto` 会输出 `<xx-YY>` 标签 |
| 语言 | 40 个 locale，含 zh-CN / ja-JP / ko-KR / en-US；普通话 CER ≈ 19%（弱项） |
| 已测性能 | M1 Pro FP32：5 s 音频整段请求 0.12–0.16 s；峰值显存 2.6 GB |

对听写的意义：
- 单调输出 ⇒ partial 可以安全地"边出边打"，无需回滚 UI。
- 延迟下限 ≈ chunk 时长 + 计算；听写推荐 lookahead=3（320 ms）或 6（560 ms）。
- "final" 不是模型概念，由端点检测（VAD 静音尾 / 松键）+ 尾块补零 flush 决定；最后几个词会晚一个 lookahead 窗到达。

### 2.3 macOS 生态与 API

**文字注入方式**

| 方式 | 需要权限 | 优点 | 缺点 |
|---|---|---|---|
| A. 剪贴板 + 合成 Cmd+V（`CGEvent`） | Accessibility | 几乎所有 App 可用，一次 Undo | 松键后才出字；要处理剪贴板恢复（macOS 15.4+ 读剪贴板可能弹窗） |
| B. `CGEventKeyboardSetUnicodeString` 逐字打字 | Accessibility | 无剪贴板副作用，可边说边打 | 长文本慢、部分 App 丢字、Undo 碎片化 |
| C. AXUIElement `kAXSelectedText` 原地替换 | Accessibility | 可原地修正 partial | Electron/终端支持差 |
| D. InputMethodKit 真输入法（`setMarkedText`/`insertText`） | 无需 Accessibility（注入本身） | 系统级 marked text 体验，最接近 Apple 听写 | 需切换输入源、与用户的中文 IME 冲突、客户端兼容性差、macOS 15.2 有已知 IMK 回调失效问题、独立进程需 IPC |

**其它要点**
- 热键：Fn/Globe 是修饰键，需通过 `flagsChanged`（`NSEvent` 全局监听或 `CGEventTap`，keyCode 63）判断按下/抬起；
  Carbon `RegisterEventHotKey` / KeyboardShortcuts 不支持 Fn。需提示用户把"按下 🌐 键时"设为"无操作"。
  惯例：按住 ≥200 ms = push-to-talk，快速双击 = 免提切换，Esc 取消。
- 安全输入（密码框、Terminal Secure Keyboard Entry）会屏蔽事件注入，用 `IsSecureEventInputEnabled()` 检测并提示。
- 权限：麦克风（TCC）、Accessibility（发事件/AX 写）、Input Monitoring（`CGEventTap` 监听）。
  Python 进程的权限会归属到宿主二进制（Terminal），因此**采集与热键放在 Swift 壳里**更干净。
- 麦克风：`AVAudioEngine.inputNode.installTap` 在 macOS 上固定 ~100 ms 缓冲；更低延迟用 `AVAudioSinkNode`
  或 HAL AudioUnit 设 256–512 帧；设备原生 44.1/48 kHz，需用持久 `AVAudioConverter` 重采样到 16 kHz Int16。
- 现有开源参考：VoiceInk（Swift，粘贴实现最成熟）、Handy（Tauri，剪贴板"owner-provided data"回执技巧）、
  our-voice（Python + Quartz 事件监听 Fn，最接近纯 Python 原型）。

---

## 3. 总体架构

```
┌──────────────────────────── macOS ────────────────────────────┐
│                                                               │
│  ┌──────────── VoiceIME.app (Swift, 菜单栏常驻) ───────────┐  │
│  │ HotkeyMonitor  (CGEventTap flagsChanged: Fn 按下/抬起)   │  │
│  │ AudioCapture   (AVAudioEngine → 16k mono Int16, 80ms 块)  │  │
│  │ RealtimeClient (WebSocket /v1/realtime?intent=transcription)│
│  │ OverlayPanel   (非激活 NSPanel, 显示 partial / 状态)      │  │
│  │ TextInjector   (剪贴板+Cmd+V | Unicode 打字 | AX 替换)    │  │
│  │ EngineSupervisor (拉起/健康检查 sgl-omni 子进程)          │  │
│  │ Settings       (热键、语言、引擎、lookahead、命令词)      │  │
│  └───────────────┬──────────────────────────┬───────────────┘  │
│                  │ ws://127.0.0.1:PORT       │ launchd / 子进程  │
│  ┌───────────────▼──────────────────────────▼───────────────┐  │
│  │ sgl-omni serve --enable-realtime  (Python, SGLANG_USE_MLX=1)│
│  │  serve/realtime: TranscriptionSession + Silero VAD + buffer │
│  │  models/nemotron3_5_asr: NemotronStreamingStrategy          │
│  │      └─ asr stage: StreamState{feat tail, conv cache,       │
│  │            attn KV window, LSTM state, tokens}  (MLX/Metal) │
│  │  备选引擎: models/qwen3_asr (MLX)  ── 同一协议              │
│  └────────────────────────────────────────────────────────────┘  │
│  可选: VoiceIME-IMK.app (~/Library/Input Methods, IMKInputController)│
│        通过 XPC/Unix socket 从 VoiceIME.app 收 partial → setMarkedText│
└───────────────────────────────────────────────────────────────┘
```

**分层职责**
- **引擎（本仓库）**：把 Nemotron MLX 从"整段"升级为"cache-aware 增量"，并接入 realtime 协议。所有改动留在
  `sglang_omni/models/nemotron3_5_asr/` 与 `sglang_omni/serve/realtime/`，遵守现有 pipeline/stage 结构。
- **客户端（新目录 `apps/mac-voice-ime/` 或独立仓库）**：Swift 菜单栏 App，负责权限、采集、热键、UI、注入、
  进程托管。不含任何模型代码。
- **协议**：沿用 OpenAI Realtime 兼容事件；新增两个可选字段（见 §4.3），保持向后兼容。

---

## 4. 引擎侧实现方案

### 4.1 两步走的流式策略

**Step A：零模型改动的"全段重解码"（M1）**
- 让 `Nemotron3_5ASRPipelineConfig` 声明 `realtime_transcription = RealtimeTranscriptionConfig(strategy_cls=NemotronStreamingStrategy, decode_interval_ms=320)`。
- `NemotronStreamingStrategy.build_decode_request` 复用现有 `build_speech_to_text_generate_request`，
  把当前 segment 的全部 PCM 打成 WAV 请求（与 Qwen3-ASR 策略相同，但不需要文本前缀回滚，因为输出单调）。
- 代价：每 320 ms 对整个活动段重新 encode + decode。M1 Pro 上 5 s 音频约 0.15 s，10 s 段约 0.3 s，
  可用但会随段长退化；由 VAD 分段（静音 500 ms）把段长压在 10 s 内基本可接受。
- 价值：一周内跑通端到端，验证协议、VAD、客户端，并拿到延迟/准确率基线。

**Step B：cache-aware 增量推理（M2，核心工作）**

新增 `sglang_omni/models/nemotron3_5_asr/mlx/streaming.py`：

```python
@dataclass
class StreamState:
    lookahead: int
    prompt_id: int
    mel_carry: np.ndarray            # 未满一块的 mel 帧（首块 1+8L 帧，其后 8(L+1) 帧）
    pcm_carry: np.ndarray            # STFT 需要的上一块尾部样本（win 400, hop 160）
    is_first_chunk: bool
    conv_cache: dict[str, mx.array]  # subsampling.{i} / conv.{i} 的左填充缓存
    attn_cache: list[(K, V)]         # 每层最近 70 帧
    frames_seen: int                 # 用于相对位置编码 cached_frames
    lstm_state: list[(h, c)]
    last_token: int
    tokens: list[int]                # 累计 token
    emitted_text_len: int
```

- `Encoder.step(chunk_feats, state)`：只编码新 chunk，卷积从 `conv_cache` 取左上下文，注意力 K/V 与缓存拼接后
  以 `[70, lookahead]` 掩码计算，然后截断缓存。对应 Torch 参考 `NemotronAsrStreamingEncoderCausalConvPaddingCache` /
  `DynamicCache(sliding_window)` / `_resolve_attn_context`。
- `Model.decode_step(encoded_chunk, state)`：把现有 `decode` 循环改为可续跑（LSTM 状态、`last_token` 来自 state）。
- 特征：复用 `Nemotron3_5AsrProcessor(is_streaming=True, is_first_audio_chunk=...)`，首块 `center=True`、
  后续 `center=False`，与 Torch 参考完全一致；块大小由 `_required_stream_chunk_frames` 决定。
- flush：段结束时把不足一块的 mel 补零到整块再跑一次，保证最后一个 lookahead 窗内的词被吐出。
- 验证：以 `hf_compat` Torch 流式（generator 输入）为黄金参考，断言 **token 序列逐块一致**（与现有
  `verify_nemotron_mlx.py` 的验证方法一致）。

**MLX 性能要点**
- 解码循环中每帧 `.item()` 同步是主要开销；每块只有 lookahead+1 帧（4 帧 @ L=3），可接受。
  可选优化：把 joint + argmax 的 blank 判断向量化，或用 `mx.compile` 编译单步。
- 每层 `mx.eval` 改为整块一次 eval。
- FP16/BF16 编码器留作后续（当前 runner 要求 FP32；先保证正确性）。

### 4.2 会话状态如何进入引擎

现有 `StreamingASRStrategy` 是"无状态整段请求"模型，Step B 需要引擎侧持有 `StreamState`：

- 新增请求类型 `Nemotron3_5ASRStreamRequest{session_id, segment_id, pcm_chunk, is_final, language}`，
  `GenerateRequest.extra_params["_asr_stream"] = {...}` 承载。
- `stages.py` 中的执行器维护 `dict[str, StreamState]`（按 `session_id:segment_id`），带 TTL 与
  `input_audio_buffer.clear` / 会话断开时的清理；MLX 下 `max_batch_size=1`，天然串行，无需并发控制以外的锁。
- 策略层 `build_decode_request` 只发送**自上次以来的新 PCM**（而不是整段），并携带 `is_final`；
  `update_hypothesis` 返回累计文本。会话层 `transcription.segment` 语义（全量替换）保持不变。
- 内存：单个 StreamState ≈ 24 层 × 70 帧 × 1024 维 × 2 (K,V) × 4 B ≈ 14 MB，可忽略。

### 4.3 协议微扩展（向后兼容）

- `session.update.session.decode_interval_ms`：允许 ≥ chunk 时长（当前默认 2000 ms，Nemotron 建议 320）。
- `transcription.segment` 增加可选字段 `delta`（自上次事件新增的文本），客户端可用它做"边说边打"；
  不发送 `delta` 的旧引擎仍按全量 `text` 处理。
- `session.update.session.turn_detection=null` 用于按住说话模式（客户端在松键时发 `commit`）。
- 新增 `session.update.session.asr={"num_lookahead_tokens": 3}` 运行时切换延迟档（模型无需重载，只影响掩码与块大小）。

### 4.4 VAD 使用方式

| 模式 | VAD 角色 | 端点 |
|---|---|---|
| 按住说话（默认） | `turn_detection=null`；仅用于裁掉开头静音与超长按住时分段（>25 s 自动 commit） | 松键 → `commit` |
| 免提/切换模式 | `server_vad`，`silence_duration_ms` 450–600，`prefix_padding_ms` 300 | 静音尾 → 自动 final |
| 长时听写（会议记录） | `server_vad` + `max_audio_clip_s` 硬切 | 同上 |

`transcription_session._is_silent` 的峰值阈值短路可避免空按调用模型。后续可评估 TEN VAD（更快的 speech→silence 转换，
Mac arm64 原生），但 Silero 已够用且已集成，M1–M3 不换。

---

## 5. 客户端实现方案（Swift 菜单栏 App）

### 5.1 模块

| 模块 | 实现 |
|---|---|
| 热键 | `CGEvent.tapCreate(.cgSessionEventTap, .listenOnly, flagsChanged)`；维护 Fn(63)/右 Option(61) 状态；`kCGEventTapDisabledByTimeout` 时自动 `tapEnable`。按住 ≥200 ms = PTT，双击 = 切换免提，Esc 取消 |
| 采集 | `AVAudioEngine` + `AVAudioConverter` → 16 kHz Int16；累积到 80 ms 的整数倍（1280 样本）后 base64 发送 `input_audio_buffer.append`。可选 `setVoiceProcessingEnabled(true)` 降噪 |
| 传输 | `URLSessionWebSocketTask`；连接复用；断线重连后重新 `session.update`（append 不幂等，重连即丢当前段并提示） |
| 悬浮窗 | `NSPanel(.nonactivatingPanel)`，`level=.floating`，`canBecomeKey=false`，显示录音波形 + 累计 partial 文本；固定于屏幕底部或跟随光标（AX `kAXInsertionPointLineNumber`/`kAXSelectedTextRange` 位置） |
| 注入 | 默认：final 到达后一次性 剪贴板写入 → `CGEvent` Cmd+V → 250 ms 后若 `changeCount` 未变则恢复剪贴板。可选"实时打字"模式：对 `delta` 用 `CGEventKeyboardSetUnicodeString` 逐段打出（利用 RNN-T 单调性，无需回退） |
| 安全输入 | `IsSecureEventInputEnabled()` 为真时只显示悬浮窗并复制到剪贴板，提示用户手动粘贴 |
| 引擎托管 | 启动时以子进程拉起 `.venv-apple/bin/sgl-omni serve --enable-realtime ...`，轮询 `/health`；崩溃自动重启；也支持连接外部已运行的服务 |
| 后处理 | 空格/首字母规则（中文不加空格、英文句末加空格）、命令词（"换行"/"new line"、"删除上一句"）、去 `<xx-YY>` 标签（服务端 adapter 已做） |
| 设置 | 热键、语言（auto/zh-CN/en-US…）、引擎（Nemotron / Qwen3-ASR）、延迟档（lookahead）、注入方式、开机自启 |

### 5.2 为什么先不做真 IMK 输入法

- 用户平时用中文 IME 时，切到"语音输入法"会失去拼音输入，需要频繁切换输入源，体验反而差；
  几乎所有成熟产品都因此选择热键 + 注入。
- IMK 客户端兼容性（Electron、JetBrains、终端）与 macOS 15.2 的回调失效问题增加不可控风险。
- 热键方案在 M3 即可交付；IMK 作为 M5 增强：把 partial 通过 `setMarkedText` 显示为下划线文本，
  final 用 `insertText` 提交，获得与系统听写一致的原地合成体验。

### 5.3 Python 原型（M1 用，之后弃用或保留为 CLI）

`playground/mac_voice_ime/dictate.py`：`sounddevice` 采集 → `websockets` 客户端 → 终端打印 partial → 松键
（`pynput` 或 Quartz 监听 Fn）后 `pbcopy` + `osascript keystroke "v" using command down`。
目的只有一个：在没有 Swift 工程前验证协议、延迟与 VAD 参数。

---

## 6. 延迟预算（目标：松键后 ≤ 400 ms 出最终文本，说话中 partial 滞后 ≤ 600 ms）

| 环节 | 估计 | 说明 |
|---|---|---|
| 采集缓冲 | 80–100 ms | `installTap` 固定 ~100 ms；HAL AU 可到 20 ms |
| 网络（本机 WS） | < 2 ms | |
| 特征 + chunk 对齐 | ≤ chunk 时长 | lookahead=3 → 320 ms 等待凑齐一块 |
| 编码一块（MLX） | 10–30 ms | 4 帧 × 24 层，待实测 |
| 解码一块 | 5–20 ms | ≤ 4 帧 × 若干 `.item()` |
| final flush | 一块补零 + 解码 | 松键时立即触发，不等 VAD |
| 注入 | 50–150 ms | 剪贴板 + Cmd+V 惯例延时 |

Step A（全段重解码）阶段 partial 滞后为 `decode_interval + 段长相关的整段重算`，不作为最终指标。

---

## 7. Milestones

| 里程碑 | 目标 | 主要产出 | 验收 |
|---|---|---|---|
| **M0 基线（~3 天）** | 环境与基线数据 | `install.sh` 装好 `.venv-apple`；Nemotron MLX 离线服务跑通；`verify_nemotron_mlx.py` 通过；记录 encode/decode 各段耗时；跑 `test_realtime_*` 单测确认 realtime 层在 Mac 可用 | 有一份延迟/显存基线表 |
| **M1 端到端原型（~1 周）** | 零模型改动接入 realtime | `Nemotron3_5ASRPipelineConfig.realtime_transcription`（Step A 策略）；`decode_interval_ms` 可配置；Python 原型客户端（按住 Fn 说话 → 终端 partial → 松键粘贴）；`tests/unit_test/serve/test_realtime_nemotron.py` | 在任意 App 中按住 Fn 说一句英文/中文能粘贴出正确文字；partial 可见 |
| **M2 真流式引擎（~2–3 周）** | cache-aware 增量推理 | `mlx/streaming.py`（StreamState、Encoder.step、decode_step、flush）；stage 会话状态管理与清理；策略改为增量 PCM；协议加 `delta` 与运行时 lookahead；单测：逐块 token 与 Torch 流式参考一致、状态清理、超长会话；基准脚本 `benchmarks/eval/bench_nemotron_stream.py`（每块耗时、RTF、首字延迟） | 10 s 以上连续说话 partial 稳定滞后 < 600 ms；token 与参考 100% 一致；内存不随时长增长 |
| **M3 Swift 菜单栏 App（~2–3 周）** | 可日常使用的客户端 | `apps/mac-voice-ime/`：权限引导、AVAudioEngine 采集、Fn 热键、悬浮窗、剪贴板注入与恢复、安全输入检测、引擎子进程托管、设置面板、开机自启；打包为未沙盒的 `.app`（Developer ID 签名 + notarize） | 冷启动 → 首次可用 < 10 s；连续使用 1 h 无崩溃；在 Safari/VS Code/Slack/Terminal 中注入成功 |
| **M4 质量与打磨（~2 周）** | 准确率、体验、备选引擎 | 中/英测试集（自录 + 公开集）上 Nemotron vs Qwen3-ASR MLX 的 WER/CER 与延迟对比，据此定默认引擎与 lookahead；后处理规则与命令词；实时打字模式；免提模式 VAD 参数调优；文档 `docs/cookbook/mac_voice_ime.md` | 有对比报告；两种引擎可在设置中切换且行为一致 |
| **M5 可选：IMK 输入法（~2 周）** | marked text 原地合成 | `VoiceIME-IMK.app`（IMKServer/IMKInputController，`setMarkedText` 显示 partial，`insertText` 提交），通过 XPC 与主 App 通信；自动切换/切回输入源 | 在原生 AppKit 文本框中获得与系统听写一致的下划线合成体验 |

M1 与 M3 的前半段（权限、采集、热键、悬浮窗）无依赖，可并行。

---

## 8. 仓库改动清单（引擎侧）

```
sglang_omni/models/nemotron3_5_asr/
  config.py                 # + realtime_transcription ClassVar, 默认 decode_interval_ms=320
  streaming.py              # NemotronStreamingStrategy (M1: 全段; M2: 增量 PCM + is_final)
  stages.py                 # M2: StreamState 表、TTL/清理、流式请求分派
  request_builders.py       # M2: Nemotron3_5ASRStreamRequest
  mlx/streaming.py          # M2: StreamState, Encoder.step, decode_step, flush
  mlx/model.py              # 抽出可复用的 block 前向（带/不带 cache 两条路径）
sglang_omni/serve/realtime/
  transcription_session.py  # decode_interval_ms 下限放宽; segment 事件可选 delta; asr 运行时参数
  events.py                 # 新字段
tests/unit_test/nemotron3_5_asr/
  test_mlx_streaming.py     # 逐块 token == Torch 流式参考; flush; 状态清理
tests/unit_test/serve/
  test_realtime_nemotron.py
benchmarks/eval/
  bench_nemotron_stream.py
playground/mac_voice_ime/   # Python 原型客户端 (M1)
apps/mac-voice-ime/         # Swift App (M3+), 或独立仓库
docs/cookbook/mac_voice_ime.md
```

---

## 9. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| Nemotron 普通话 CER 偏高 | 中文听写体验差 | M4 用真实数据 A/B；同协议下切换 Qwen3-ASR MLX；`language` 显式指定而非 auto |
| MLX 流式与 Torch 参考数值漂移 | 偶发错字 | 逐块 token 一致性测试作为 CI 门槛；FP32 先行 |
| Python 逐帧解码 `.item()` 开销 | 每块 >50 ms | 单步 `mx.compile`；blank 判断向量化；必要时 lookahead=6 |
| Fn 键与系统"🌐 键"功能冲突 | 触发表情/听写 | 首次引导设为"无操作"；提供右 Option 备选 |
| macOS 15.4+/26 剪贴板隐私弹窗 | 恢复剪贴板被打断 | 只在 `changeCount` 未变时恢复；提供"实时打字"模式绕过剪贴板 |
| 安全输入场景 | 无法注入 | 检测并降级为仅复制 + 提示 |
| 引擎子进程冷启动慢（模型加载） | 首次可用等待 | 开机自启常驻；健康检查后再显示可用状态 |
| `append` 不幂等 / 断线 | 丢一段话 | 客户端本地保留当前段 PCM，重连后作为离线 `/v1/audio/transcriptions` 补发 |

---

## 10. 待确认的决策

1. 默认引擎与语言：先按 Nemotron + `language=auto` 做，M4 后依据数据决定。
2. 客户端代码放本仓库 `apps/` 还是独立仓库：建议独立仓库（Swift 工具链、签名、发布节奏不同），本仓库只保留 Python 原型与协议文档。
3. 是否做 M5 IMK：视 M3 后的实际需求（是否强需求"原地下划线合成"）。
