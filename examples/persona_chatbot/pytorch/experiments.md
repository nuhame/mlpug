# Persona Chatbot Experiments (2025-12-28)

Experiments run on 6x AMD MI100 GPUs (32 GiB each) with ROCm 7.1.1.

## Experiment 1: Baseline (No Activation Checkpointing)

```bash
python -m examples.persona_chatbot.pytorch.train \
    --experiment-name baseline-28122025 \
    --num-dataloader-workers 2 \
    --use-mixed-precision \
    --batch-size 39 \
    --micro-batch-size 13 \
    --num-choices 8 \
    --learning-rate 6.25e-5 \
    --lr-warmup-schedule \
    --num-epochs 6 \
    --progress-log-period 10 \
    --distributed
```

**Configuration:**
- batch_size=39, micro_batch_size=13 (3x gradient accumulation)
- lr=6.25e-5, LR warmup schedule (1 epoch warmup)
- torch.compile enabled, no activation checkpointing
- 6 GPUs (DDP), mixed precision

**Results:**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Duration |
|-------|------------|----------|-----------|---------|----------|
| 0 | 1.563 | 1.601 | 86.0% | 83.9% | 4:13 |
| 1 | 1.432 | 1.507 | 90.6% | 88.1% | 4:17 |
| 2 | 1.357 | 1.455 | 93.3% | 89.7% | 4:20 |
| 3 | 1.315 | 1.439 | 94.6% | 90.4% | 4:21 |
| **4** | **1.289** | **1.431** | **95.5%** | **90.9%** | 4:22 |
| 5 | 1.278 | 1.435 | 95.9% | 90.8% | 4:25 |

**Best model: Epoch 4** — Val Loss: 1.431, Val Acc: 90.9%

---

## Experiment 2: Activation Checkpointing with Baseline LR

```bash
python -m examples.persona_chatbot.pytorch.train \
    --experiment-name actckpt-baseline-lr-28122025 \
    --num-dataloader-workers 2 \
    --use-mixed-precision \
    --batch-size 150 \
    --micro-batch-size 50 \
    --num-choices 8 \
    --learning-rate 6.25e-5 \
    --lr-warmup-schedule \
    --num-epochs 6 \
    --progress-log-period 10 \
    --distributed \
    --activation-checkpointing
```

**Configuration:**
- batch_size=150, micro_batch_size=50 (3x gradient accumulation)
- lr=6.25e-5 (same as baseline), LR warmup schedule (1 epoch warmup)
- torch.compile enabled, activation checkpointing enabled
- 6 GPUs (DDP), mixed precision

**Results:**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Duration |
|-------|------------|----------|-----------|---------|----------|
| 0 | 1.848 | 1.863 | 80.1% | 78.6% | 5:43 |
| 1 | 1.550 | 1.590 | 86.0% | 84.5% | 5:47 |
| 2 | 1.484 | 1.532 | 88.6% | 86.7% | 5:52 |
| 3 | 1.449 | 1.512 | 89.8% | 87.3% | 5:48 |
| 4 | 1.426 | 1.504 | 90.8% | 87.9% | 5:53 |
| **5** | **1.420** | **1.497** | **91.0%** | **88.0%** | 5:52 |

**Best model: Epoch 5** — Val Loss: 1.497, Val Acc: 88.0%

---

## Experiment 3: Activation Checkpointing with Scaled LR (4x)

```bash
python -m examples.persona_chatbot.pytorch.train \
    --experiment-name actckpt-scaled-lr-28122025 \
    --num-dataloader-workers 2 \
    --use-mixed-precision \
    --batch-size 150 \
    --micro-batch-size 50 \
    --num-choices 8 \
    --learning-rate 2.5e-4 \
    --lr-warmup-schedule \
    --num-epochs 6 \
    --progress-log-period 10 \
    --distributed \
    --activation-checkpointing
```

**Configuration:**
- batch_size=150, micro_batch_size=50 (3x gradient accumulation)
- lr=2.5e-4 (4x baseline, following linear scaling rule), LR warmup schedule (1 epoch warmup)
- torch.compile enabled, activation checkpointing enabled
- 6 GPUs (DDP), mixed precision

**Results:**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Duration |
|-------|------------|----------|-----------|---------|----------|
| 0 | 1.583 | 1.618 | 85.3% | 84.0% | 5:54 |
| 1 | 1.450 | 1.530 | 89.8% | 86.9% | 5:56 |
| 2 | 1.367 | 1.477 | 92.7% | 89.1% | 5:55 |
| 3 | 1.316 | 1.462 | 94.2% | 89.5% | 5:58 |
| 4 | 1.285 | 1.452 | 95.3% | 90.1% | 6:01 |
| **5** | **1.272** | **1.450** | **95.7%** | **90.5%** | 5:58 |

**Best model: Epoch 5** — Val Loss: 1.450, Val Acc: 90.5%

---

## Summary Comparison

| Experiment | Batch Size | LR | Act. Ckpt | Best Val Loss | Best Val Acc | Epoch Time |
|------------|------------|-----|-----------|---------------|--------------|------------|
| Baseline | 39 | 6.25e-5 | No | 1.431 | 90.9% | ~4:20 |
| Act. Ckpt + Baseline LR | 150 | 6.25e-5 | Yes | 1.497 | 88.0% | ~5:50 |
| Act. Ckpt + Scaled LR | 150 | 2.5e-4 | Yes | **1.450** | **90.5%** | ~5:57 |

**Key Findings:**

1. **Activation checkpointing** enables ~4x larger batch sizes (39 → 150) by reducing memory usage.

2. **Linear LR scaling** is important: with 4x larger batch size, using 4x higher LR (2.5e-4 vs 6.25e-5) significantly improves results.

3. **Baseline still wins** on validation accuracy (90.9% vs 90.5%), likely because:
   - Smaller batch size means more gradient updates per epoch
   - The model sees more diverse mini-batches during training

4. **Trade-off**: Activation checkpointing adds ~35% time overhead per epoch but enables training larger models or using larger batch sizes when memory-constrained.
