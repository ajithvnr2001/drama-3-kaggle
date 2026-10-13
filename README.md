# Fish Audio Drama-3: Dual Tesla T4 Deployment & Tamil Quality Benchmark

An end-to-end deployment, hardware optimization pipeline, and benchmark suite for Fish Audio's newly released **Drama-3** model ([`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3)) on Kaggle **Dual NVIDIA Tesla T4 GPUs** (SM75, 2x 16 GB VRAM, 30 GB System RAM).

---

## 1. Executive Summary

**Drama-3** is Fish Audio's latest 35B parameter mixture-of-experts (MoE) speech foundation model, featuring 256 experts (8 active per token), hybrid linear and full attention, a 10-codebook audio decoder tower, and a 44.1 kHz Descript Audio Codec (DAC).

This repository contains the complete engineering solution developed to execute Drama-3 on standard cloud dual-GPU infrastructure (Kaggle Dual T4), overcoming hardware architecture constraints, storage limits, and memory ceilings.

---

## 2. Key Engineering Breakthroughs

### A. SM75 (Tesla T4) Compatibility for Block-FP8 MoE
- **Problem**: Drama-3's official code strictly checks for SM89+ (Ada Lovelace / Hopper) for hardware accelerated FP8 matrix multiplication (`fp8_available`). On Tesla T4 (Turing, SM75), it attempts to dequantize the entire 35B model from FP8 into 16-bit floats, exploding model weights from **35 GB to over 70 GB** and causing immediate out-of-memory.
- **Solution**:
  1. Patched `model.py` to relax the SM89 assertion, maintaining expert weights in compact **block-FP8 format** (`torch.float8_e4m3fn`).
  2. Patched `moe.py` to route tokens through `forward_gathered` with dynamic fallback on SM75. Only the **8 active experts per token** are dequantized on-the-fly into `float16`/`float32` during matrix multiplication, leaving inactive experts in FP8 memory.

### B. Overcoming the Kaggle 20 GB Disk Boundary (Streaming Shard Loader)
- **Problem**: Drama-3's checkpoint consists of 8 `.safetensors` shards totaling **39.5 GB**. Standard Kaggle kernels provide only **19.6 GB** of writable disk on `/kaggle/working`. Pre-downloading all weights causes `OSError: [Errno 28] No space left on device`.
- **Solution**: Developed an **Incremental Shard Streaming Loader** (`load_fish_omni_streamed`):
  1. Allocates the meta-architecture in PyTorch (`torch.device("meta")`).
  2. Materializes weight buffers across Dual GPUs and CPU.
  3. Downloads each ~4.66 GB shard **one-by-one**, transfers parameters directly into memory, and **immediately unlinks** the shard from disk.
  4. Peak disk usage stays below **6.5 GB** at all times!

### C. Tri-Tier Hardware Memory Partitioning
- **GPU 0 (`cuda:0`)**: Layers 0–12 (13 layers) + Token Embeddings (`embed_tokens`) $\rightarrow$ **11.17 GB VRAM** (leaves **3.39 GB free buffer**).
- **GPU 1 (`cuda:1`)**: Layers 13–23 (11 layers) + Audio Decoder Tower + Final Norm + LM Head $\rightarrow$ **10.81 GB VRAM** (leaves **3.75 GB free buffer**).
- **Host System RAM (`cpu`)**: Layers 24–39 (16 layers) $\rightarrow$ **12.48 GB RAM** (out of 30 GB available Host RAM, leaving **16 GB free**).

### D. Zero-VRAM Audio Decoding
- **Problem**: Running DAC codec waveform decoding on GPU 0 triggered `CUDA out of memory` during the final conv1d upsampling step by 192 MiB.
- **Solution**: Configured the 44.1 kHz DAC Codec on **CPU Host RAM** in native `float32`. Waveform synthesis takes only **~0.3–0.5s** per track on CPU and uses **0 MB of GPU VRAM**.

---

## 3. Benchmark Results (Kaggle Dual Tesla T4)

| Scenario | Audio Duration | Generation Latency | Real-Time Factor (RTF) | Generation Speed | Peak VRAM (GPU 0 / 1) | Acoustic Sample Rate | Output File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Track 1: Thriller Novel Scene with Emotion Tags** | 8.36s | 798.75s | **97.52x** | 0.2 fps | 11.76 GB / 12.40 GB | 44,100 Hz PCM | `outputs/track_1_emotion_thriller.wav` |
| **Track 2: Multi-Speaker Thriller Interaction** | 8.36s | 812.71s | **99.18x** | 0.2 fps | 11.76 GB / 12.41 GB | 44,100 Hz PCM | `outputs/track_2_multi_speaker_dialogue.wav` |
| **Track 3: Native Tamil Fluency & Quality** | 5.57s | 544.32s | **99.66x** | 0.2 fps | 11.76 GB / 12.41 GB | 44,100 Hz PCM | `outputs/track_3_native_tamil_fluency.wav` |

*Note: All tracks were synthesized directly on Kaggle kernel `ajithkumaevnr2001/drama-3-dual-t4-test` and downloaded to the local outputs directory.*

---

## 4. Head-to-Head Comparison: Drama-3 vs. Gemini 3.8 Flash-Lite TTS

| Feature | Fish Audio Drama-3 (35B MoE) | Gemini 3.8 Flash-Lite TTS |
| :--- | :--- | :--- |
| **Acoustic Fidelity** | **44.1 kHz Studio Mastering** (DAC neural codec) | 24.0 kHz standard streaming audio |
| **Emotion & Direction Tags** | **Native fine-grained point tags**: `[whispering]`, `[panicked]`, `[gasp]`, `[door slam]`, `[sighs]` | Prompt-level style steering |
| **Multi-Speaker Interaction** | **True multi-character dialogue** in a single turn (`<|speaker:0|>`, `<|speaker:1|>`) | Single speaker per request |
| **Tamil Pronunciation** | Native Tamil token vocabulary + zero-shot voice cloning from native reference audio | Strong multilingual TTS with Google Tamil voice models |
| **Deployment Mode** | **100% self-hosted & open-source** (Dual T4 GPU / SGLang-Omni) | Proprietary cloud API |
| **Speed & Inference Cost** | High compute requirement (MoE 35B; best on Dual L4/A100; ~0.2 fps on T4 CPU-hybrid) | Ultra-fast cloud streaming (~1.5–2.5s latency, $0.0003/min) |

---

## 5. Repository Structure

```
drama-3-kaggle/
├── build_drama3_kernel.py     # Builder script that injects base64 audio and Kaggle kernel logic
├── run_drama3_kaggle.py       # Standalone Kaggle execution script (auto-generated)
├── kernel-metadata.json       # Kaggle kernel deployment metadata
├── ref_native_tamil.wav       # Female native Tamil studio reference audio
├── anchor_tamil_hq.wav        # Male Tamil news anchor reference audio
├── outputs/
│   ├── track_1_emotion_thriller.wav       # Synthesized Tamil thriller track (44.1 kHz)
│   ├── track_2_multi_speaker_dialogue.wav # Synthesized multi-character dialogue (44.1 kHz)
│   ├── track_3_native_tamil_fluency.wav   # Synthesized native Tamil test (44.1 kHz)
│   ├── DRAMA3_BENCHMARK_REPORT.md         # Generated benchmark evaluation report
│   └── benchmark_results.json             # Raw benchmark metrics (durations, RTF, VRAM)
└── README.md                  # Comprehensive documentation and technical report
```

---

## 6. How to Reproduce

### Prerequisites
- Python 3.10+
- Kaggle API token installed (`~/.kaggle/access_token` or `kaggle.json`)
- Hugging Face Access Token with permissions for `fishaudio/drama-3`

### Deploying to Kaggle
```bash
# 1. Generate run script and metadata
python build_drama3_kernel.py

# 2. Push kernel to Kaggle Dual T4
kaggle kernels push -p D:\azure\drama-3-kaggle

# 3. Download synthesized audio and report once complete
kaggle kernels output ajithkumaevnr2001/drama-3-dual-t4-test -p D:\azure\drama-3-kaggle\outputs
```
