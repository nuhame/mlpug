# Data Preprocessing for Agentic LLM Pretraining

This document describes how raw dataset samples are preprocessed into training data for next-token prediction (NTP).

> **Status:** This document is pending final review.

## Overview

The preprocessing pipeline:
1. **Transform** — Convert raw samples to serialized text (see `transform_analysis.md`)
2. **Tokenize** — Convert text to token IDs using Qwen3 tokenizer
3. **Pack** — Concatenate tokenized documents with `<|endoftext|>` separator
4. **Index** — Build Forager index for O(1) random access

---

## Qwen3 Tokenizer Special Tokens

| Token | ID | Usage |
|-------|-----|-------|
| `<|endoftext|>` | **151643** | End-of-document (EOD), inserted between documents in packing |
| `<|im_start|>` | 151644 | Start of turn (chat fine-tuning) |
| `<|im_end|>` | 151645 | End of turn (chat fine-tuning), generation EOS |

**For pretraining**, only `<|endoftext|>` is used. The `<|im_start|>` and `<|im_end|>` tokens are for chat/instruction fine-tuning.

---

## Document Packing Strategy

Following Qwen3's pretraining approach, documents are **concatenated** (not padded) with `<|endoftext|>` as separator:

```
[Doc1 tokens...] <|endoftext|> [Doc2 tokens...] <|endoftext|> [Doc3 tokens...] <|endoftext|>
|<-------------------------- packed sequence (e.g., 32K tokens) ------------------------->|
```

### Why Packing?

- **Efficiency**: No wasted compute on padding tokens
- **More tokens per batch**: Reduces training time
- **Cross-document attention**: Model learns document boundaries naturally

### Implementation

```python
EOD_TOKEN_ID = 151643  # <|endoftext|>

def pack_documents(tokenized_docs: list[list[int]], max_length: int) -> list[int]:
    """Pack multiple tokenized documents into a single sequence."""
    packed = []
    for doc_tokens in tokenized_docs:
        if packed:
            packed.append(EOD_TOKEN_ID)
        packed.extend(doc_tokens)
        if len(packed) >= max_length:
            break
    return packed[:max_length]
```

---

## Context Length Strategy

### Qwen3-1.7B Specifications

| Property | Value |
|----------|-------|
| Max context length | 32,768 tokens (32K) |
| Vocab size | ~152K tokens |

### Baseline Approach: 32K Constant

For initial experiments, use **32K context length** throughout training:
- Simpler setup
- Clear baseline for comparison
- No mid-training context switches

### Staged Training (Future)

Following Qwen3's approach, training can be staged:

| Stage | Context | Focus | Data Selection |
|-------|---------|-------|----------------|
| S1 | 4K | Basic language, general knowledge | Any samples, chunked to 4K |
| S2 | 4K | STEM, coding, reasoning | Reasoning/code samples, chunked to 4K |
| S3 | 32K | Long-context comprehension | Long samples (filter by length) |

**Key insight**: Tokenize at 32K once, then:
- S1/S2: Split 32K sequences into 8x 4K chunks
- S3: Use full 32K sequences

This avoids re-tokenization between stages.

---

## Length-Based Filtering for Staged Training

Instead of dedicated "long" datasets, filter **any** dataset by character/token length:

```python
MIN_CHARS_FOR_LONG = 32_000  # ~8K tokens, adjust as needed

def is_long_sample(sample: dict, text_field: str = "text") -> bool:
    """Filter for long samples (S3 training)."""
    text = sample.get(text_field, "")
    return len(text) >= MIN_CHARS_FOR_LONG
```

This provides diverse long-context data:
- Long reasoning chains (math, code)
- Long articles (fineweb-edu, wikipedia)
- Long dialogues
- Long code examples

---

## Data Categories for Staged Training

Our datasets map to Qwen3's stages:

| Stage | Qwen Focus | Our `primary_purpose` Labels |
|-------|------------|------------------------------|
| S1 | Language, general knowledge | grammar, knowledge, dialogue |
| S2 | STEM, coding, reasoning | reasoning, procedural, debugging, agentic, code, rag_grounding |
| S3 | Long-context | Any dataset, filtered by length |

---

## Forager Integration

[Forager](https://github.com/nuhame/forager) provides O(1) random access to tokenized data.

### Tokenization and Indexing

```python
from forager.indexers.tokenization_indexer import create_tokenize_and_index_jsonl_text_func
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
EOD_TOKEN_ID = 151643

# One-time: tokenize corpus and build index
tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
    input_base_path='./corpus',
    tokenizer_func=tokenizer.encode,
    eos_idx=EOD_TOKEN_ID,  # <|endoftext|> as document separator
    sample_size=32768,      # 32K context length
)
tokenize_and_index()
```

### Training with Random Access

```python
from forager.datasets.tokens import TokensDataset
from torch.utils.data import DataLoader

dataset = TokensDataset.create_from_index_on_filesystem('./corpus')
dataset.initialize()

dataloader = DataLoader(
    dataset,
    batch_size=micro_batch_size,
    shuffle=True,  # True random shuffling via index
    num_workers=2,
)

for batch in dataloader:
    # batch shape: (micro_batch_size, 32768)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    # ... training step
```

---

## Preprocessing Pipeline Summary

```
Raw JSONL files (per dataset)
        │
        ▼
┌───────────────────┐
│ Transform         │  Apply dataset-specific templates
│ (transform_analysis.md) │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Merge & Shuffle   │  Combine datasets with mixing ratios
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Tokenize          │  Qwen3 tokenizer, add <|endoftext|> (151643)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Pack              │  Concatenate to 32K sequences
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Forager Index     │  Build index for O(1) random access
└───────────────────┘
        │
        ▼
   Training Data
   (tokenized .bin + index)
```

---

## References

- [Qwen3 Blog: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Qwen Tokenization Notes](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md)
- [HuggingFace: LLM Sequence Packing](https://huggingface.co/blog/sirluk/llm-sequence-packing)
- [Forager GitHub](https://github.com/nuhame/forager)
