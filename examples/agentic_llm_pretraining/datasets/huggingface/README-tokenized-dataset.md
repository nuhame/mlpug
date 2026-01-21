---
license: cc-by-sa-4.0
language:
- en
tags:
- pretraining
- tokenized
- agentic
- qwen3
- ntp
size_categories:
- 100K<n<1M
task_categories:
- text-generation
---

# Agentic LLM Pretraining Dataset - Tokenized (Qwen3, 4K context)

Pre-tokenized version of [visionscaper/agentic-llm-pretraining-1.7b](https://huggingface.co/datasets/visionscaper/agentic-llm-pretraining-1.7b) for pre-training small language models for agentic AI use cases.

## Overview

| Property | Value |
|----------|-------|
| Source dataset | [visionscaper/agentic-llm-pretraining-1.7b](https://huggingface.co/datasets/visionscaper/agentic-llm-pretraining-1.7b) |
| Tokenizer | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Context length | 4,096 tokens |
| EOD token | `<\|endoftext\|>` (ID 151643) |
| Token dtype | uint32 |
| Total samples | 375,384 |
| Total tokens | ~1.54 billion |
| Storage | ~5.8 GB |
| License | CC-BY-SA-4.0 |

### Why `<|endoftext|>` as EOD token?

Qwen3's `eos_token` is `<|im_end|>` (ID 151645), which is used to mark the end of chat turns. Using it as a document separator would create ambiguity with the chat format in the training data.

Instead, we use `<|endoftext|>` (ID 151643, Qwen3's `pad_token`) as the end-of-document separator. This clearly marks document boundaries without conflicting with chat turn markers.

## Splits

| Split | Samples | Size | Tokens | Purpose |
|-------|---------|------|--------|---------|
| train | 265,894 | 4.1 GB | ~1.09B | Model training |
| val | 54,650 | 856 MB | ~224M | Monitor generalization during training |
| test | 54,840 | 859 MB | ~225M | Final model evaluation |

## File Format

Uses [Data Forager](https://github.com/visionscaper/data-forager) format for O(1) random access via memory-mapped byte offsets:

```
{split}/
├── index/
│   ├── sample_locations.bin  # Byte offsets (24 bytes per sample)
│   └── file_location.txt     # File paths
└── tokenized-samples/
    └── {split}-tokenized-samples.bin  # Token data (uint32)
```

Each sample is exactly 4,096 tokens. Documents are packed sequentially, separated by `<|endoftext|>` tokens.

## Usage

### Installation

```bash
pip install data-forager huggingface_hub torch
```

### Download from HuggingFace

```python
from huggingface_hub import snapshot_download

# Download the full dataset (~5.8 GB)
local_path = snapshot_download(
    repo_id="visionscaper/agentic-llm-pretraining-1.7b-tokenized-qwen3-4k",
    repo_type="dataset",
)
print(f"Downloaded to: {local_path}")

# Or download only specific splits
train_path = snapshot_download(
    repo_id="visionscaper/agentic-llm-pretraining-1.7b-tokenized-qwen3-4k",
    repo_type="dataset",
    allow_patterns=["train/**"],
)
```

### Load with Data Forager

```python
from data_forager.datasets.tokens import TokensDataset
from torch.utils.data import DataLoader
import numpy as np

# Load the training split
dataset = TokensDataset.create_from_index_on_filesystem(
    f"{local_path}/train",
    token_dtype=np.uint32,
)

print(f"Dataset size: {len(dataset)} samples")
print(f"Sample shape: {dataset[0].shape}")  # (4096,)

# Create DataLoader with true random shuffling
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
)

# Training loop
for batch in loader:
    # batch shape: (batch_size, 4096)
    input_ids = batch[:, :-1]   # (batch_size, 4095)
    labels = batch[:, 1:]       # (batch_size, 4095)

    # Your training code here
    ...
```

### Decode Samples (Optional)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Decode a sample to inspect
sample = dataset[0].numpy()
text = tokenizer.decode(sample, skip_special_tokens=False)
print(text[:500])
```

## Tokenization Details

This dataset was tokenized using a script from the [MLPug](https://github.com/nuhame/mlpug) library example code:

```bash
python -m examples.agentic_llm_pretraining.datasets.tokenize_dataset \
    --splits-dir /path/to/splits \
    --output-dir /path/to/output \
    --tokenizer Qwen/Qwen3-1.7B \
    --context-length 4096 \
    --eod-token pad
```

The `--eod-token pad` flag uses `<|endoftext|>` (the pad token) as the document separator.

## Related Resources

- **Source dataset**: [visionscaper/agentic-llm-pretraining-1.7b](https://huggingface.co/datasets/visionscaper/agentic-llm-pretraining-1.7b)
- **Data Forager library**: [GitHub](https://github.com/visionscaper/data-forager) | [PyPI](https://pypi.org/project/data-forager/)
- **MLPug training framework**: [GitHub](https://github.com/nuhame/mlpug)
- **Training script**: [train.py](https://github.com/nuhame/mlpug/blob/feature/pretraining-agentic-llm-example-29122025/examples/agentic_llm_pretraining/training/pytorch/train.py)

## License

This dataset is licensed under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/), the same license as the source dataset.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{agentic_llm_pretraining_qwen3_4k_tokenized_2026,
  author = {Snijder, Freddy},
  title = {Agentic LLM Pretraining Dataset - Tokenized (Qwen3, 4K context)},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/visionscaper/agentic-llm-pretraining-1.7b-tokenized-qwen3-4k}
}
```
