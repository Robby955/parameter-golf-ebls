# Empirical Bayes Layer Sharing (EBLS) for Parameter Golf

**EBLS achieves 1.2105 BPB pre-quantization on FineWeb validation, using learned Bayesian parameter sharing to automatically discover optimal weight-tying patterns in tiny transformers.**

> Entry for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) (March–April 2026).
> Non-record submission exploring a novel architecture direction.

## Key Finding

Instead of manually choosing which layers to share (as in hard recurrence or XSA4), EBLS **discovers the sharing pattern from data**. Three shared transformer blocks are each applied three times (9 effective layers), with per-virtual-layer LoRA deviations gated by learned shrinkage factors $\gamma_i = \sigma(\text{logit}_i)$.

After training on 8×H100 SXM (4572 steps, 10-min wallclock), the gammas converge:

| Virtual Layer | Attention $\gamma$ | MLP $\gamma$ |
|:---:|:---:|:---:|
| 0 | 0.0046 | 0.0006 |
| 1 | 0.0029 | 0.0000 |
| 2 | 0.0018 | 0.0000 |
| 3 | 0.0006 | 0.0000 |
| 4 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 |
| 6 | 0.0000 | 0.0000 |
| 7 | 0.0000 | 0.0000 |
| 8 | 0.0000 | 0.0000 |

**MLP weights converge to full sharing** ($\gamma = 0$) across all virtual layers. Attention shows minimal specialization only in early layers. The model independently discovers that MLP computation should be identical across virtual layers while attention barely needs to deviate from the shared prototype. This provides empirical evidence for architectural choices that other submissions make by intuition.

For the statistical foundations connecting James-Stein shrinkage to neural network parameter sharing, see [`docs/ebls_theory.md`](docs/ebls_theory.md).

## Method

### 1. Shared Prototype Blocks with LoRA Deviations

Three shared transformer blocks serve as learned prototypes. Each block is applied $k=3$ times in sequence (9 effective layers), with per-virtual-layer specialization via rank-8 LoRA:

$$W_{\text{effective}}^{(i)} = W_{\text{shared}} + \gamma_i \cdot A_i B_i$$

where $\gamma_i = \sigma(\text{logit}_i)$ is a learned shrinkage factor that controls how much each virtual layer deviates from the shared prototype. Separate gammas for attention and MLP modules allow the model to discover different sharing patterns per component.

The architecture uses U-Net skip connections between encoder and decoder halves of the virtual layer stack.

### 2. Shrinkage Regularization

A penalty $\lambda \sum_i \sigma(\text{logit}_i)$ encourages the model toward sharing unless deviation genuinely helps. This is analogous to the James-Stein estimator shrinking individual estimates toward the grand mean — the shared weights serve as an empirical Bayes prior.

### 3. Competitive Stack

Built on the established competitive baseline from the challenge:

- **Int6 STE QAT**: Fake quantization to [-31, 31] during training with straight-through estimator
- **zstd-22 compression**: ~1.86x compression ratio on trained int8 weights
- **3x MLP expansion** with ReLU² activation
- **SmearGate**: Per-dimension sigmoid gate blending current and previous token ([PR #162](https://github.com/openai/parameter-golf/pull/162))
- **BigramHash(10240)**: Hash-based bigram embeddings projected to model dimension
- **SWA**: Stochastic weight averaging over last 40% of warmdown (9 checkpoints)
- **Muon optimizer** with WD=0.04, momentum=0.99, Newton-Schulz orthogonalization
- **GQA**: 16 query heads, 4 KV heads at 1024 model dimension

## Results

| Configuration | BPB | Steps | Artifact | Notes |
|---|---|---|---|---|
| Official baseline | 1.2244 | 13,780 | 15.9 MB | 9L, 512-dim, int8+zlib |
| Leaderboard #1 (thwu1) | 1.1428 | ~8K | ~15.9 MB | 10L, mixed int5/int6 |
| **EBLS (pre-quant)** | **1.2105** | **4,572** | — | 9L (3×3), 1024-dim |
| **EBLS (post-quant)** | **1.3441** | 4,572 | 16.2 MB | int6+zstd roundtrip |

### What Works

- EBLS pre-quantization BPB (1.2105) **beats the baseline** (1.2244) despite only 4572 training steps (vs baseline's 13,780) — the wider 1024-dim model learns faster per step.
- Gamma convergence is robust and consistent across runs, confirming the Bayesian sharing pattern.
- EBLS frees ~50% of parameters from duplicated storage, enabling the wider model within the same byte budget.

### Current Status

Pre-quantization BPB beats the baseline, but the 1024-dim model is too wide for the 10-minute budget: only 4572 training steps (vs 13,780 baseline) and a 16.22MB artifact (225KB over limit). The next run addresses both by reducing to 768-dim, which halves step time and fits under 16MB.

### Path Forward

| Change | Expected Impact | Status |
|---|---|---|
| Train + eval at seq_len=2048, batch=786K | ~0.02 BPB from longer context | Implemented |
| Mixed int5/int6 (int5 for MLP weights) | Saves ~1.86MB → funds extra layer | Implemented |
| Magnitude pruning (3%) before compression | Better zstd compression on zeros | Implemented |
| Orthogonal init | Better gradient flow at init | Implemented |
| Sliding window eval always-on (stride=64) | More accurate BPB measurement | Implemented |
| Dimension test: 512-dim×12L vs 768-dim×9L | Determine optimal width/depth tradeoff | Testing |

Target: **1.16–1.19 BPB post-quant** on 8×H100.

## Reproducing

### Setup
```bash
git clone https://github.com/Robby955/parameter-golf-ebls.git
cd parameter-golf-ebls
bash prepare.sh
```

### Train + Evaluate (8×H100)
```bash
bash eval/eval.sh
```

### Quick Smoke Test (1×GPU, no compile)
```bash
SKIP_COMPILE=1 ITERATIONS=200 VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Architecture

```
Input tokens
    │
    ├──► tok_emb(1024 → 1024-dim)
    ├──► BigramHash(10240 buckets, 128-dim → 1024-dim)
    ├──► SmearGate (per-dim current/prev blend)
    ├──► RMSNorm
    │
    ▼
┌─────────────────────────────────────┐
│  Shared Block 0 (applied 3×)       │
│    Attn: Q/K/V/O + γ_attn·LoRA     │  ◄── Encoder (layers 0-3)
│    MLP:  up/down  + γ_mlp·LoRA     │      + U-Net skip connections
│                                     │
│  Shared Block 1 (applied 3×)       │
│    ...                              │  ◄── Decoder (layers 4-8)
│                                     │      + skip weight mixing
│  Shared Block 2 (applied 3×)       │
│    ...                              │
└─────────────────────────────────────┘
    │
    ▼
  RMSNorm → tied embedding logits → softcap(30)
```

## License

MIT. See [LICENSE](LICENSE).
