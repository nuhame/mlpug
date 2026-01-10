---
license: cc-by-sa-4.0
language:
- en
size_categories:
- 1M<n<10M
task_categories:
- text-generation
- question-answering
tags:
- pretraining
- agentic
- function-calling
- reasoning
- code
- math
- dialogue
- rag
pretty_name: Agentic LLM Pretraining Dataset (1.7B tokens)
---

# Agentic LLM Pretraining Dataset

A pretraining corpus for small language models (1-3B parameters) optimized for agentic tasks. The corpus emphasizes learning to comprehend language, reason, follow instructions, and use tools over memorizing factual knowledge — the assumption is that domain knowledge will be provided at runtime via RAG. The idea is that this could enable much smaller pretraining corpora by omitting the large volumes of text typically needed to memorize facts.

## Dataset Summary

| Split | Samples | Estimated Tokens | Purpose |
|-------|---------|------------------|---------|
| Train | 2,655,695 | ~1.19B | Model training |
| Validation | 569,077 | ~255M | Monitor generalization during training |
| Test | 569,078 | ~255M | Final model evaluation |
| **Total** | **3,793,850** | **~1.7B** | |

## Intended Use

- **Primary**: Pretraining language models (1-3B parameters) for agentic applications
- **Secondary**: Fine-tuning, evaluation, or research on multi-task learning

## Dataset Composition

The dataset combines 27 source datasets across 8 categories:

| Category | Datasets | Samples | Description |
|----------|----------|---------|-------------|
| Grammar/Language | fineweb-edu, fineweb-edu-long, wikipedia, simple-wikipedia | 585,446 | High-quality educational and encyclopedic text |
| Reasoning/Math | openmath-instruct-1, openmath-instruct-2, gsm8k, math, ecqa | 772,308 | Mathematical reasoning with step-by-step solutions |
| Procedural | cosmopedia-wikihow, stackexchange | 335,594 | How-to articles and technical Q&A |
| Code | swe-bench, code-contests, codesearchnet | 442,273 | Programming tasks, debugging, and code documentation |
| Agentic | toolace, hermes-function-calling, glaive-function-calling | 126,153 | Function calling with tool use patterns |
| Knowledge | generics-kb, openbookqa | 1,025,825 | Factual knowledge and science reasoning |
| Dialogue | soda, multiwoz, wizard-of-wikipedia, sharegpt, empathetic-dialogues, samsum | 480,329 | Conversational data with various styles |
| RAG | rag-dataset-12000, ragbench-hotpotqa | 11,481 | Retrieval-augmented generation patterns |

### Per-Dataset Details

| Dataset | Total | License | HuggingFace Source | Filter | Quality* |
|---------|-------|---------|-------------------|--------|---------|
| generics-kb | 1,020,868 | CC BY 4.0 | [generics_kb](https://huggingface.co/datasets/generics_kb) | score > 0.5 | Medium |
| openmath-instruct-1 | 482,677 | NVIDIA License | [nvidia/OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | Correct solutions only | Good |
| codesearchnet | 412,178 | Various | [claudios/code_search_net](https://huggingface.co/datasets/claudios/code_search_net) | — | Good |
| soda | 359,329 | Apache 2.0 | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) (SODA) | — | Good |
| wikipedia | 309,800 | CC BY-SA 3.0 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en) | — | Good |
| openmath-instruct-2 | 262,060 | CC BY 4.0 | [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | — | Good |
| simple-wikipedia | 241,787 | CC BY-SA 3.0 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.simple) | — | Medium |
| cosmopedia-wikihow | 179,000 | Apache 2.0 | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (wikihow) | — | Good |
| stackexchange | 156,594 | CC BY-SA 4.0 | [HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) | — | Good |
| glaive-function-calling | 112,960 | Apache 2.0 | [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | — | Good |
| sharegpt | 76,041 | Apache 2.0 | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) (ShareGPT) | — | Good |
| fineweb-edu | 28,765 | ODC-By | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | — | Good |
| swe-bench | 19,000 | MIT | [princeton-nlp/SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | — | Good |
| wizard-of-wikipedia | 18,430 | Apache 2.0 | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) (wizard_of_wikipedia) | — | Good |
| empathetic-dialogues | 17,802 | Apache 2.0 | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) (Empathetic) | — | Low |
| samsum | 14,731 | Apache 2.0 | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) (SAMSum) | — | Good |
| math | 12,500 | MIT | [qwedsacf/competition_math](https://huggingface.co/datasets/qwedsacf/competition_math) | — | Good |
| code-contests | 11,095 | CC BY 4.0 | [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) | Has C++/Python3/Java solution | Good |
| toolace | 11,300 | Apache 2.0 | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | — | Good |
| rag-dataset-12000 | 9,598 | Apache 2.0 | [neural-bridge/rag-dataset-12000](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) | — | Good |
| multiwoz | 8,437 | Apache 2.0 | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) (MULTIWOZ2_2) | — | Medium |
| ecqa | 7,598 | CDLA-Sharing-1.0 | [tasksource/ecqa](https://huggingface.co/datasets/tasksource/ecqa) | — | Medium |
| gsm8k | 7,473 | MIT | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | — | Good |
| fineweb-edu-long | 5,094 | ODC-By | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-100BT) | ≥120K chars, quality ≥3 | Good |
| openbookqa | 4,957 | Apache 2.0 | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | — | Good |
| ragbench-hotpotqa | 1,883 | CC BY 4.0 | [rungalileo/ragbench](https://huggingface.co/datasets/rungalileo/ragbench) (hotpotqa) | — | Good |
| hermes-function-calling | 1,893 | Apache 2.0 | [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | — | Good |

*Quality as assessed through manual inspection of a small number of samples: **Good** = high-quality, appropriate for training; **Medium** = some issues but acceptable; **Low** = quality concerns but kept for coverage.

## Data Preprocessing and Formatting

Each sample is a JSON object with three fields:

```json
{"source": "gsm8k", "index": 42, "text": "Problem: ..."}
```

- `source`: Original dataset name
- `index`: Sample index within the source dataset
- `text`: Preprocessed text ready for tokenization

### Text Formats by Category

**Free text** (output as-is):
- Wikipedia articles with markdown headings
- Code functions with docstrings
- Short factual statements

**Structured reasoning** (Problem/Solution format):
```
Problem: A store sells apples for $2 each...

Solution: Let's solve this step by step.
1. First, calculate the total cost...
...
The answer is \boxed{42}.
```

**Conversational** (Qwen3 chat format):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
How do I sort a list in Python?<|im_end|>
<|im_start|>assistant
You can use the sorted() function...<|im_end|>
```

**Function calling** (formats vary by source dataset):
- `<functioncall>{"name": "func", "arguments": {...}}</functioncall>` (glaive-function-calling)
- `<tool_call>{"name": "func", ...}</tool_call>` (hermes-function-calling)
- `[func(param=value)]` (toolace)

### Preprocessing Steps

All samples were preprocessed with dataset-specific transformations:

1. **Filtering**: Quality and other property filters applied per-dataset (see Filter column in table above)
2. **Preprocessing**: Sample data fields extracted and normalized per dataset
3. **Templating**: Preprocessed fields applied to text templates (e.g., Qwen3 chat format for conversational datasets)
4. **Global shuffling**: Samples randomly shuffled across all sources

### Source Code

For full details on data acquisition, filtering, and preprocessing, see the code [here](https://github.com/nuhame/mlpug/blob/feature/pretraining-agentic-llm-example-29122025/examples/agentic_llm_pretraining/datasets/), specifically:

- [training_metadata.json](https://github.com/nuhame/mlpug/blob/feature/pretraining-agentic-llm-example-29122025/examples/agentic_llm_pretraining/datasets/training_metadata.json) — Acquisition, filtering, and preprocessing metadata per dataset
- [preprocessing.py](https://github.com/nuhame/mlpug/blob/feature/pretraining-agentic-llm-example-29122025/examples/agentic_llm_pretraining/datasets/preprocessing.py) — Sample preprocessing functions per dataset
- [dataset_templates.py](https://github.com/nuhame/mlpug/blob/feature/pretraining-agentic-llm-example-29122025/examples/agentic_llm_pretraining/datasets/dataset_templates.py) — Templates for representing preprocessed data as text
- [create_splits.py](https://github.com/nuhame/mlpug/blob/feature/pretraining-agentic-llm-example-29122025/examples/agentic_llm_pretraining/datasets/create_splits.py) — Global shuffling and train/val/test split creation

## Usage

### Load the dataset

```python
from datasets import load_dataset

# Load all splits
dataset = load_dataset("visionscaper/agentic-llm-pretraining-1.7b")

# Load specific split
train = load_dataset("visionscaper/agentic-llm-pretraining-1.7b", split="train")
```

### Filter by source

```python
# Get only math reasoning samples
math_data = train.filter(lambda x: x["source"] in ["gsm8k", "math", "openmath-instruct-1"])

# Get only dialogue samples
dialogue_sources = ["soda", "sharegpt", "wizard-of-wikipedia", "multiwoz"]
dialogue_data = train.filter(lambda x: x["source"] in dialogue_sources)

# Get only agentic/function-calling samples
agentic_sources = ["toolace", "hermes-function-calling", "glaive-function-calling"]
agentic_data = train.filter(lambda x: x["source"] in agentic_sources)
```

### Tokenization example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=32768)

tokenized = train.map(tokenize, batched=True, remove_columns=["source", "index", "text"])
```

## Known Limitations

- **fineweb-edu**: Quality score filtering (`int_score >= 3`) was not applied. All quality levels (1-5) are included. Should be fixed in v2.
- **empathetic-dialogues**: Quality is low (role confusion, peer-to-peer chat style). Consider an alternative in the future.

## License

This dataset is released under **CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0 International).

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material for any purpose, including commercial

Under the following terms:
- **Attribution** — You must give appropriate credit and indicate if changes were made
- **ShareAlike** — If you remix or transform the material, you must distribute under the same license

Individual samples retain their original source licenses. All source datasets permit commercial use. See the per-dataset license column in the table above for details.

## Citation

```bibtex
@dataset{snijder2025agentic,
  author = {Snijder, Freddy},
  title = {Agentic LLM Pretraining Dataset},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/visionscaper/agentic-llm-pretraining-1.7b}
}
```

If you use this dataset, please also cite the original source datasets appropriately.

## Contact

Created by [Visionscaper](https://github.com/visionscaper) for the Minimal NTP → RLP research project.
