# MLPug Development Context

## Project Overview
MLPug is a flexible ML training framework for which development started in 2019, development continued
until mid-2024 and is now picked up again. The idea of MLPug is to make the training loop and evaluation
code reusable over different deep learning backends. The main focus is PyTorch, but in the future JAX and
Apple MLX might be added.

We're preparing MLPug for modern LLM training, specifically for the "Minimal NTP → RLP" research project
(training 1.7B parameter models with novel reinforcement learning during pretraining).

## Current Status: Micro-batch Refactor MERGED ✓

**Merged to master:** 2025-12-27
**PR:** https://github.com/nuhame/mlpug/pull/1

### What Was Done
The refactor from "batch → chunks" to "micro-batch × accumulation_steps" is complete and merged:

1. **Trainer owns accumulation logic** (`mlpug/pytorch/trainers/training.py`)
   - `gradient_accumulation_steps` calculated from `batch_size / micro_batch_size`
   - `train_on()` returns `(MicroBatchResults, did_update_model)`
   - Loss scaling: `loss / gradient_accumulation_steps`
   - `epoch_complete()` handles partial accumulation at epoch end

2. **TrainingManager simplified** (`mlpug/trainers/training.py`)
   - No more batch slicing - just iterates micro-batches
   - Tracks `micro_batch_step` (dataloader iterations) and `batch_step` (optimizer steps)
   - `on_batch_training_completed` fires only at accumulation boundaries
   - `on_micro_batch_completed` fires after every micro-batch

3. **MetricEvaluator cleaned up** (`mlpug/evaluation.py`)
   - Removed `batch_chunk_size` and `chunkable_batch_wrapper` parameters
   - Handles `MicroBatchResults` (list of micro-batch outputs) transparently

4. **Deleted chunking infrastructure**
   - Removed `mlpug/batch_chunking.py` entirely
   - Removed all `ChunkableBatch`, `ChunkableBatchWrapper` references

5. **Deleted TensorFlow backend**
   - Removed `mlpug/tensorflow/` entirely

### Verified Working
- `examples/hello_world/pytorch/train.py` ✓
- `examples/fashion_mnist/pytorch/train.py` ✓
- Gradient accumulation produces identical loss (diff ~5.5e-10, floating point only)
- Batch counting shows effective batches (1875) not micro-batches (7500)

### All Examples Tested
- `examples/hello_world/pytorch/train.py` ✓
- `examples/fashion_mnist/pytorch/train.py` ✓
- `examples/persona_chatbot/pytorch/train.py` ✓ (full DDP + AMP + compile grand test passed)

## Key Architecture

### Batch vs Micro-batch
- **"Batch"** = semantic unit for metrics/logging (effective batch size: 32)
- **"Micro-batch"** = memory unit (what fits in GPU: 8)
- Users think in batches, framework handles micro-batch details internally

### MicroBatchResults
```python
class MicroBatchResults(list):
    """List of normalized result dicts from micro-batches within accumulation window."""
    # Each element: {'loss': tensor, 'num_samples': int, 'auxiliary_results': Any}
```

### Callback Timing
- `on_micro_batch_completed`: After every dataloader iteration
- `on_batch_training_completed`: Only at accumulation boundaries (when optimizer steps)

## Framework Strengths (Keep These)

- **Callback system**: Stateful with `get_state()`/`set_state()`, good lifecycle hooks, hash-based identity
- **Separation of concerns**: TrainingManager orchestrates, Trainer trains, Callbacks extend
- **Checkpoint management**: Full state save/restore works well
- **Multi-optimizer support**: Already handles complex optimization scenarios
- **Mixed precision**: GradScaler integration works

## Known Issues to Address Later

1. **Magic string paths**: `set_value_at("training.batch.raw.model_outputs", ...)` — consider typed dataclasses for core `logs` structure
2. **Long files**: `training.py` at 1181 lines — split TrainingManager/Trainer/DefaultTrainer
3. **Missing types**: Add type hints throughout
4. **`basics` dependency**: Inline this (it's just Base class with logger from visionscaper-pybase)

## Dependencies

The `basics` package is `visionscaper-pybase` on PyPI. Contains:
- `basics.base`: Base class with logger
- `basics.logging`: Simple logging setup  
- `basics.base_utils`: Re-exports from validation_utils and logging_utils
- `basics.logging_utils`: `log_exception()` helper

## Research Context

This refactor prepares MLPug for the "Minimal NTP → RLP" research:
- Training Qwen3-1.7B (28 layers, 1.7B params)
- Phase 1: 1B tokens NTP (next-token prediction)
- Phase 2: RLP (Reinforcement Learning Pretraining) with information-gain rewards
- Goal: Develop reasoning capabilities with minimal pretraining data

See `@docs/private/minimal-ntp-rlp-proposal.md` for full research proposal.

## Streaming Dataset Integration

After micro-batch refactor, add streaming support:
```python
from datasets import load_dataset

# HuggingFace streaming - yields samples one at a time
dataset = load_dataset("path", streaming=True)

# Wrap in DataLoader with micro_batch_size
# Framework handles accumulation transparently
```

## Testing the Refactor

1. Run existing examples after refactor:
   - `examples/fashion_mnist/pytorch/train.py`
   - `examples/persona_chatbot/pytorch/train.py`

2. Verify gradient accumulation produces identical results:
   - micro_batch=8, accumulation=4 should match batch=32 (within floating point)

3. Check DDP still works:
   - `no_sync()` context properly skips gradient sync on non-boundary steps

## Test Results (ROCm 7.1.1 on AMD MI100)

### fashion_mnist Results
Tested 12 combinations (Single GPU/DDP × Compile/Eager × Accum/No-Accum × AMP/No-AMP):
- **12/12 PASSED** (after lowering AMP GradScaler init_scale to 1024)

All configurations work correctly, including Single GPU + compile + AMP.

### persona_chatbot Results
Memory-intensive example with multiple response choices per sample, 50k vocab.

**Baseline (master branch, pre-refactor):**
- Configuration: `batch_size=39, batch_chunk_size=13, num_choices=8, --use-mixed-precision --distributed`
- 6 epochs on 6 GPUs with DDP + AMP (no torch.compile)
- Epoch duration: ~14m 04s
- Average batch training time: **1877ms**
- Training loss: ~1.33, Validation loss: ~1.45
- Training accuracy: ~91%

**Grand Test (6 GPUs, DDP, AMP, compile) - After Refactor:**
- Configuration: `batch_size=39, micro_batch_size=13, num_choices=8, --use-mixed-precision --distributed`
- **PASSED**: Full 6 epochs completed successfully
- Epoch duration: ~4m 11s (consistent across all epochs)
- Average batch training time: **470ms**
- Training loss: 1.33, Validation loss: 1.45
- Training accuracy: 91.2%, Validation accuracy: 87.9%
- Memory usage: ~73% per GPU (23.4 GiB / 32 GiB)
- **Speedup: ~4x** (1877ms → 470ms per batch)

**Key Fixes:**
- AMP GradScaler init_scale lowered from 65536 to 1024 for ROCm 7.1.1 stability.
  See `DefaultTrainerMixin.__init__` parameter `amp_init_scale`.
- Compiled optimizer step implemented in `DefaultTrainerMixin._setup_optimizer_step_funcs()`.

**Memory Test (micro-batch size limits):**
| mbs | batch_size | Memory | Status |
|-----|------------|--------|--------|
| 13  | 39         | ~73%   | ✓ (original config) |
| 14  | 42         | ~78%   | ✓ |
| 18  | 54         | ~99%   | ✓ (maximum before OOM) |
| 19  | 57         | >100%  | ✗ OOM |

The refactored code allows **38% larger micro-batches** (13→18) at the memory limit.

### torch.compile + DDP Compatibility
Uses `DDPModelWrapper` pattern (commit `6c5f784`):
- **Non-DDP mode**: Compile entire `_training_step` function
- **DDP mode**: `DDPModelWrapper` compiles model first, then wraps with DDP
- Trainer detects wrapper and skips redundant compilation
- `no_sync()` context moved outside compiled region to avoid torch.compile issues

Uses fixed-length padding (`drop_last=True` in DataLoader) for static tensor shapes.

## Known Limitations

### ROCm AMP GradScaler Issue (FIXED)
On ROCm 7.1.1, the default AMP GradScaler init_scale (65536) causes overflow issues.
- **Root cause**: Micro-batch gradient scaling changes gradient magnitudes; default scale too aggressive
- **Solution**: `amp_init_scale` parameter added to `DefaultTrainerMixin.__init__`, defaulting to 1024
- **Status**: FIXED - all DDP + AMP + compile configurations now work correctly

### Activation Checkpointing + torch.compile (RESOLVED)
This was previously incompatible but **works correctly on PyTorch 2.x** (tested on PyTorch 2.6).

The compatibility check was removed from `persona_chatbot/pytorch/train.py` on 2025-12-26.

**Test Results** (6 GPUs, DDP, AMP, compile, activation checkpointing):
- Configuration: `batch_size=39, micro_batch_size=13, --activation-checkpointing --use-mixed-precision --distributed`
- **PASSED**: Full 6 epochs completed successfully
- Training loss: 1.417, Validation loss: 1.496
- Training accuracy: 90.9%, Validation accuracy: 87.7%
- Average batch time: ~546ms (vs ~470ms without activation checkpointing = ~16% overhead)
- Memory savings from activation checkpointing allows larger models or batch sizes

**Modern Techniques Available** (PyTorch 2.4+):
- **Memory Budget API**: `torch._dynamo.config.activation_memory_budget = 0.5` - automatic checkpointing
- **Selective Activation Checkpointing (SAC)**: Fine-grained control with policy functions
- **Best practice**: Always use `use_reentrant=False` when calling `checkpoint()` directly

**Memory Scaling Test** (6 GPUs, DDP, AMP, compile, activation checkpointing):

| Config | Micro-batch | Memory | Status |
|--------|-------------|--------|--------|
| No activation checkpointing | 18 | ~99% | Max (OOM at 19) |
| With activation checkpointing | 35 | ~71% | ✓ |
| With activation checkpointing | 50 | ~98% | ✓ |

Activation checkpointing enables **2.8x larger micro-batch sizes** (18 → 50) while staying under GPU memory limits.

## Completed Optimizations

### Generic Model Wrapper Injection (DONE)
Implemented in commit `6c5f784`. Allows injection of a `model_wrapper_func` for DDP, FSDP, or custom wrappers.

**Architecture:**
- `ModelWrapperFunc` protocol in `mlpug/trainers/model_wrappers.py`
- `DDPModelWrapper` in `mlpug/pytorch/model_wrappers/ddp.py`
- `model_wrapper_func` parameter added to `DefaultTrainer`

**How it works:**
1. User creates wrapper: `model_wrapper_func = DDPModelWrapper(rank, device)`
2. Passes to trainer: `DefaultTrainer(..., model_wrapper_func=model_wrapper_func)`
3. Trainer calls wrapper in `set_training_model()`, injecting `eager_mode` and `compile_kwargs`
4. `DDPModelWrapper` compiles model first (if not eager), then wraps with DDP

**Benefits:**
- Follows PyTorch's "compile then wrap" recommendation
- Enables future FSDP support with same pattern
- Removes DDP-specific logic from trainer internals

**Examples updated:**
- `examples/fashion_mnist/pytorch/train.py`
- `examples/persona_chatbot/pytorch/train.py`

### Compiled Optimizer Step (DONE)
PyTorch docs show ~48% speedup by compiling `optimizer.step()` separately.
Reference: https://pytorch.org/tutorials/recipes/compiling_optimizer.html

Implemented in `DefaultTrainerMixin._setup_optimizer_step_funcs()`:
- Compiles optimizer step with `torch.compile(step_func, fullgraph=False)`
- For AMP: wraps `scaler.step(optimizer)` before compilation
- Validation added: AMP with multiple optimizers raises error (GradScaler tracking limitation)

## Future TODOs

### Training Log File Location
Training log files should be stored at `../training-logs/` relative to the mlpug root directory,
not within the examples directory structure.

## Performance Analysis

### 4x Performance Improvement (EXPLAINED)
The micro-batch refactor + torch.compile shows a ~4x speedup (1877ms → 470ms per batch).

**Root cause**: Master tried to compile the entire `_training_step` function including DDP
synchronization calls. This is known to cause issues with torch.compile.

**Solution**: `DDPModelWrapper` now handles this correctly:
- DDP mode: Compile model first, then wrap with DDP (via `DDPModelWrapper`)
- Non-DDP: Compile entire `_training_step`

This follows PyTorch's recommendation: compile the model before/inside DDP, not code that
uses DDP from outside.

Contributing factors (in order of impact):
1. **Proper torch.compile handling for DDP** - `DDPModelWrapper` implements "compile then wrap"
2. **Compiled optimizer step** - additional speedup from `_setup_optimizer_step_funcs()`
3. **Removal of batch chunking overhead** - ChunkableBatch wrapper eliminated
4. **Cleaner gradient accumulation** - simpler code path

## Remote Testing Rig (AMD 6xMI100)

### Connection Details
- **Host**: `freddy@192.168.178.85`
- **MLPug root**: `/home/freddy/workspace/Nuhame/mlpug`
- **Training logs**: `../training-logs/` (relative to mlpug root)
- **Python environment**: `source ~/.virtualenvs/mlpug/bin/activate`

### Hardware
- 6x AMD MI100 GPUs (32 GiB each)
- ROCm 7.1.1

### Running Experiments
Structure experiment commands as follows (log to `../training-logs/<experiment-name>.log`):
```bash
ssh freddy@192.168.178.85 'cd /home/freddy/workspace/Nuhame/mlpug && \
source ~/.virtualenvs/mlpug/bin/activate && \
python -m examples.persona_chatbot.pytorch.train \
    --experiment-name persona-bot-test-26122025 \
    --num-dataloader-workers 2 \
    --use-mixed-precision \
    --batch-size 39 \
    --micro-batch-size 13 \
    --num-choices 8 \
    --learning-rate 6.25e-5 \
    --lr-warmup-schedule \
    --num-epochs 6 \
    --progress-log-period 10 \
    --distributed \
    2>&1 | tee ../training-logs/persona-bot-test-26122025.log'
```

## Coding Style

### Method Ordering in Classes
Define methods in this order:
1. `__init__` and other dunder methods
2. Public methods (user-facing API)
3. Protected methods (`_single_underscore`)
4. Private methods (`__double_underscore`)

Users of a class should see the methods they can use first.

### Docstring Formatting
From the second line of any docstring section (parameters, returns, etc.), indent only 4 spaces:

```python
def example(self, param1, param2):
    """
    Short description.

    :param param1: First parameter description that may span
        multiple lines with 4-space continuation indent.
    :param param2: Second parameter.

    :return: Description of return value that may also
        span multiple lines.
    """
```
