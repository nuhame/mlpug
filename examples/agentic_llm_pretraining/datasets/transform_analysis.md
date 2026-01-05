# Dataset Transform Analysis for Agentic LLM Pretraining

This document analyzes how to transform each dataset into serialized text for next-token prediction (NTP) training.

## General Principles

1. **Plain text output**: All samples are transformed into plain text sequences
2. **Semantic formatting**: Use markdown headings only for actual hierarchy (e.g., document titles), use `Label: content` for flat Q&A/problem-solution structures
3. **Preserve reasoning structure**: Show step-by-step reasoning when available
4. **Consistent labels**: Use clear section labels (e.g., `Problem:`, `Solution:`, `Question:`, `Answer:`)
5. **Qwen3 chat format for conversations**: Conversational datasets (agentic, dialogue) use Qwen3 chat syntax with `<|im_start|>role` and `<|im_end|>` tokens. This teaches the model conversation structure during pretraining since we skip separate chat fine-tuning.
6. **Document separator**: `<|endoftext|>` (ID 151643) separates documents during packing

### Qwen3 Chat Format Reference

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

Token IDs: `<|im_start|>` = 151644, `<|im_end|>` = 151645, `<|endoftext|>` = 151643

### Data Mixing: Structure vs Free Text

When using finetuning datasets during pretraining (NTP), the model learns all patterns including template structures. Unlike finetuning where loss masking hides the structure, pretraining learns everything. To prevent over-fitting to structured formats:

- **Free text should be 30-50% of the dataset** (fineweb-edu, wikipedia, simple-wikipedia, codesearchnet, generics-kb)
- Structured formats (Q&A, chat, problem/solution) provide useful signal but create learned patterns
- The diversity across ~26 datasets provides natural regularization
- Repeated structures (e.g., "Question: ... Answer: ...") are acceptable as they represent natural patterns

---

## 1. Grammar/Language Datasets

### fineweb-edu
**Purpose**: Grammar and general knowledge from educational web content

**Relevant Fields**:
- `text`: Educational web content (string, ~200-5000 chars)
- `score`: Educational quality score 1-5 (float)

**Usage**: Direct text, no transformation needed. High-quality educational content teaches grammar, style, and factual knowledge.

**Template**:
```
{text}
```

**Processing**: None. Use as-is.

---

### wikipedia
**Purpose**: Grammar and encyclopedic knowledge

**Relevant Fields**:
- `title`: Article title (string)
- `text`: Article content (string, ~500-10000 chars)

**Usage**: Article format provides structured factual writing.

**Template**:
```
# {title}

{text}
```

**Processing**: Prepend title as heading.

---

### simple-wikipedia
**Purpose**: Simplified grammar, accessible knowledge

**Relevant Fields**:
- `title`: Article title (string)
- `text`: Simplified article content (string, ~100-1000 chars)

**Usage**: Teaches clear, accessible writing style.

**Template**:
```
# {title}

{text}
```

**Processing**: Same as wikipedia.

---

## 2. Reasoning/Math Datasets

### gsm8k
**Purpose**: Grade school math with step-by-step reasoning

**Relevant Fields**:
- `question`: Math word problem (string)
- `answer`: Step-by-step solution with calculations (string)

**Usage**: Teaches structured problem-solving with explicit reasoning steps. The `<<...>>` notation shows calculations.

**Template**:
```
Problem: {question}

Solution: {answer}
```

**Processing**: Parse answer to identify final numeric answer if needed. Keep calculation annotations.

---

### math (competition_math)
**Purpose**: Competition-level math with LaTeX

**Relevant Fields**:
- `problem`: Math problem with LaTeX (string)
- `level`: Difficulty level (string, e.g., "Level 3")
- `type`: Math category (string, e.g., "Prealgebra", "Algebra")
- `solution`: Detailed solution with LaTeX (string)

**Usage**: Advanced reasoning with mathematical notation.

**Template**:
```
Problem ({type}, {level}): {problem}

Solution: {solution}
```

**Processing**: Include metadata for context. Keep LaTeX notation.

---

### openmath-instruct-1
**Purpose**: Synthetic math problems with code solutions

**Relevant Fields**:
- `question`: Math problem (string)
- `generated_solution`: Solution, often with Python code in `<llm-code>` tags (string)
- `expected_answer`: Correct answer (string)
- `is_correct`: Whether generated solution is correct (bool)

**Usage**: Shows programmatic problem-solving approach. Only use samples where `is_correct=True`.

**Template**:
```
Problem: {question}

Correct generated solution: {generated_solution}
```

**Processing**: Filter for `is_correct=True`. Keep `<llm-code>` tags as-is (provides HTML/XML syntax exposure and signals LLM-generated content). Answer is already in the solution (in `\boxed{}`).

---

### openmath-instruct-2
**Purpose**: High-quality synthetic math from Llama 3.1-405B

**Relevant Fields**:
- `problem`: Math problem (string)
- `generated_solution`: Detailed solution (string)
- `expected_answer`: Final answer (string)

**Usage**: High-quality reasoning chains. All solutions are correct (no filtering needed).

**Template**:
```
Problem: {problem}

Solution: {generated_solution}
```

**Processing**: None needed. Answer is already in the solution (in `\boxed{}`).

---

### ecqa
**Purpose**: Commonsense reasoning with explanations

**Relevant Fields**:
- `q_text`: Question (string)
- `q_op1` through `q_op5`: Answer choices (strings)
- `q_ans`: Correct answer (string)
- `taskA_pos`: Positive properties of correct answer (string)
- `taskA_neg`: Negative properties of wrong answers (string)
- `taskB`: Free-form explanation (string)

**Usage**: Teaches comparative reasoning - why one answer is right and others are wrong.

**Template**:
```
Question: {q_text}

Options:
A) {q_op1}
B) {q_op2}
C) {q_op3}
D) {q_op4}
E) {q_op5}

Reasoning:
{taskA_pos}
{taskA_neg}

Conclusion: {taskB}

Answer: {q_ans}
```

**Processing**: Map options to letters. taskA_pos explains why the correct answer is right, taskA_neg explains why other options are wrong.

---

## 3. Procedural Datasets

### cosmopedia-wikihow
**Purpose**: Procedural how-to knowledge

**Relevant Fields**:
- `prompt`: Generation prompt describing the task (string)
- `text`: Full how-to article (string, ~2000-8000 chars)

**Usage**: Teaches structured procedural writing with steps.

**Template**:
```
{text}
```

**Processing**: Use text directly (already well-formatted with steps). Optionally include prompt as context.

---

### stackexchange
**Purpose**: Q&A with community-validated answers

**Relevant Fields**:
- `question`: Question with HTML (string)
- `answers`: List of answers with `text`, `pm_score`, `selected` (list[dict])
- `metadata`: Tags/categories (list[str])

**Usage**: Technical problem-solving with ranked solutions.

**Template** (single best answer):
```
Question: {question}

Answer: {best_answer_text}
```

**Template** (multiple tied best answers):
```
Question: {question}

Answer 1: {best_answer_text_1}

Answer 2: {best_answer_text_2}
```

**Processing**:
1. Keep HTML as-is (provides HTML syntax exposure)
2. Find highest `pm_score`, include all answers with that score
3. Use single-answer template if one best, multi-answer template if tied

---

## 4. Code Datasets

### code-contests
**Purpose**: Competitive programming with test cases

**Relevant Fields**:
- `name`: Problem name (string)
- `description`: Problem statement (string)
- `public_tests`: Input/output test cases (dict with `input` and `output` lists)
- `solutions`: Correct solutions with `language` (list[int]) and `solution` (list[str])
- `cf_tags`: Problem categories (list[str])

**Language IDs**: 2=C++, 3=Python3, 4=Java

**Usage**: Algorithmic problem-solving with verified solutions.

**Template**:
```
Problem: {name}

{description}

Examples:
Input: {public_tests.input[0]}
Output: {public_tests.output[0]}

Input: {public_tests.input[1]}
Output: {public_tests.output[1]}
...

Solution ({language_name}):
```{language_ext}
{shortest_solution}
```
```

**Processing**:
1. Random language selection with weights: Python3 (0.5), C++ (0.25), Java (0.25)
2. Select shortest solution in chosen language
3. Include ALL public test cases

**Selection Code**:
```python
import random

LANGUAGE_IDS = {2: ('C++', 'cpp'), 3: ('Python3', 'python'), 4: ('Java', 'java')}
WEIGHTS = {3: 0.5, 2: 0.25, 4: 0.25}

def select_solution(solutions):
    available = set(solutions['language'])
    candidates = [lid for lid in WEIGHTS if lid in available]
    if not candidates:
        return None, None
    weights = [WEIGHTS[lid] for lid in candidates]
    chosen_id = random.choices(candidates, weights=weights)[0]
    indices = [i for i, lid in enumerate(solutions['language']) if lid == chosen_id]
    shortest_idx = min(indices, key=lambda i: len(solutions['solution'][i]))
    return LANGUAGE_IDS[chosen_id], solutions['solution'][shortest_idx]
```

---

### codesearchnet
**Purpose**: Code with documentation patterns

**Relevant Fields**:
- `whole_func_string`: Complete function code with docstring (string)
- `repository_name`: Source repository (string)
- `func_path_in_repository`: File path (string)
- `language`: Programming language (string)

**Note**: Originally designed for code search (retrieval). We use it for natural code exposure - teaching code structure and documentation patterns.

**Template**:
```
{whole_func_string}
```

**Processing**: None - just output the function code with embedded docstring as-is.

---

## 5. Agentic/Tool-Use Datasets

### toolace
**Purpose**: API/function calling with reasoning

**Relevant Fields**:
- `system`: System prompt with available functions (string)
- `conversations`: List of turns with `from` (user/assistant/tool) and `value` (string)

**Usage**: Teaches function selection and argument construction.

**Template** (2-turn, most common):
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{conversations[0].value}<|im_end|>
<|im_start|>assistant
{conversations[1].value}<|im_end|>
```

**Template** (multi-turn with tool results):
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{conversations[0].value}<|im_end|>
<|im_start|>assistant
{conversations[1].value}<|im_end|>
<|im_start|>tool
{conversations[2].value}<|im_end|>
<|im_start|>assistant
{conversations[3].value}<|im_end|>
...
```

**Processing**:
1. Map `from` values to Qwen3 roles: user→user, assistant→assistant, tool→tool
2. Keep function call format (e.g., `[FunctionName(param=value)]`)

---

### hermes-function-calling
**Purpose**: Function calling examples

**Relevant Fields**:
- `conversations`: List with `from` (system/human/gpt) and `value` (string)
- `category`: Task category (string) - metadata, not used in template
- `task`: Task description (string) - metadata, not used in template

**Usage**: Multi-turn function calling with observations.

**Template**:
```
<|im_start|>system
{conversations[0].value}<|im_end|>
<|im_start|>user
{conversations[1].value}<|im_end|>
<|im_start|>assistant
{conversations[2].value}<|im_end|>
```

**Processing**:
1. Map `from` values to Qwen3 roles: system→system, human→user, gpt→assistant
2. Keep function call format (e.g., `<tool_call>...</tool_call>`)

---

### glaive-function-calling
**Purpose**: Function calling examples with system prompts

**Relevant Fields**:
- `system`: System prompt with "SYSTEM: " prefix (string)
- `chat`: Pre-formatted conversation with USER:/ASSISTANT:/FUNCTION RESPONSE: prefixes (string)

**Usage**: Simple function calling patterns. ~58% of samples include function calls and responses.

**Template**:
```
<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
<|im_start|>tool
{function_response}<|im_end|>
<|im_start|>assistant
{next_assistant_response}<|im_end|>
...
```

**Processing**:
1. Strip "SYSTEM: " prefix from system field
2. Parse chat field by splitting on `USER:`, `ASSISTANT:`, `FUNCTION RESPONSE:`
3. Map to Qwen3 roles: USER→user, ASSISTANT→assistant, FUNCTION RESPONSE→tool
4. Remove existing `<|endoftext|>` tokens (added at document packing level)
5. Keep `<functioncall>` format as-is

---

## 6. Knowledge Datasets

### generics-kb
**Purpose**: Common sense facts

**Relevant Fields**:
- `generic_sentence`: Factual statement (string)
- `score`: Confidence score (float)

**Note**: Self-contained factual sentences. Term is already embedded in the sentence.

**Template**:
```
{generic_sentence}
```

**Processing**:
1. Output sentence as-is
2. Optionally filter by `score > 0.5` for quality

---

### openbookqa
**Purpose**: Core science facts with reasoning

**Relevant Fields**:
- `question_stem`: Science question (string)
- `choices`: Answer options with `text` and `label` (dict)
- `answerKey`: Correct answer label (string)

**Template**:
```
Question: {question_stem}

A) {choices.text[0]}
B) {choices.text[1]}
C) {choices.text[2]}
D) {choices.text[3]}

Answer: {answerKey}) {correct_choice_text}
```

**Processing**: Format choices as A) B) C) D). Include correct answer with label and text.

---

## 7. Dialogue Datasets (DialogStudio format)

All DialogStudio datasets share a common structure:

**Common Fields**:
- `log`: List of turns with `user utterance`, `system response`, `dialog history` (list[dict])
- `original dialog info`: JSON string with context/metadata (string)

### soda
**Purpose**: Social dialogues with emotional context

**Relevant Fields**:
- `log`: List of turns with `user utterance`, `system response` (list[dict])
- `original dialog info`: JSON string containing `narrative`, `speakers` (string)
- `prompt`: List of 6 alternative role-play instructions (list[str])

**Template**:
```
<|im_start|>system
{original_dialog_info["narrative"]}

{prompt[random_choice_index]}<|im_end|>
<|im_start|>user
{log[0]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[0]["system response"]}<|im_end|}
<|im_start|>user
{log[1]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[1]["system response"]}<|im_end|>
...
```

**Processing**:
1. Parse `original dialog info` JSON to extract `narrative`
2. Randomly select one of the 6 prompts
3. Combine narrative + prompt as system message
4. Format all turns from `log` as alternating user/assistant

---

### multiwoz
**Purpose**: Task-oriented dialogues (booking, info requests)

**Relevant Fields**:
- `log`: List of turns with `user utterance`, `system response` (list[dict])
- `prompt`: List of 5 alternative task prompts (list[str])
- `original dialog info`: JSON with `services` (domains like hotel, restaurant, train)

**Note**: This is a "Wizard of Oz" dataset - tool calls (database lookups, bookings) are implicit, not recorded. Teaches task-oriented dialogue patterns without explicit function call syntax.

**Template**:
```
<|im_start|>system
{prompt[random_choice_index]}<|im_end|>
<|im_start|>user
{log[0]["user utterance"]}<|im_end|}
<|im_start|>assistant
{log[0]["system response"]}<|im_end|>
<|im_start|>user
{log[1]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[1]["system response"]}<|im_end|>
...
```

**Processing**:
1. Randomly select one of the 5 prompts as system message
2. Format all turns from `log` as alternating user/assistant
3. DST/intent fields are annotation metadata, not included in training text

---

### wizard-of-wikipedia
**Purpose**: Knowledge-grounded conversation (RAG-style)

**Relevant Fields**:
- `log`: List of turns with `user utterance`, `system response` (list[dict])
- `original dialog info`: JSON containing `chosen_topic`, `chosen_topic_passage` (list[str]), `persona`

**Note**: ~80% of samples have assistant starting the conversation (empty first user utterance). The wizard uses Wikipedia passages to inform responses - teaches knowledge-grounded dialogue.

**Template**:
```
<|im_start|>system
You act as a person with the following persona: {persona}

Together with a user you will talk about: {chosen_topic}

Background info:
{chosen_topic_passage[0]}
{chosen_topic_passage[1]}
{chosen_topic_passage[2]}
...<|im_end|>
<|im_start|>user
{log[i]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[i]["system response"]}<|im_end|>
...
```

**Processing**:
1. Parse `original dialog info` JSON to extract `chosen_topic`, `chosen_topic_passage`, `persona`
2. Build instructional system message with persona, topic, and background passages
3. For each turn: skip empty user utterances, skip empty system responses
4. Handles both assistant-initiated (~80%) and user-initiated (~20%) conversations

---

### sharegpt
**Purpose**: General user-assistant conversations

**Relevant Fields**:
- `log`: List of turns with `user utterance`, `system response` (list[dict])
- `original dialog info`: Contains `model` (e.g., "gpt-3.5-turbo") - metadata only

**Note**: No system prompt in data. Real user↔GPT conversations teaching general helpful assistant behavior.

**Template**:
```
<|im_start|>user
{log[0]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[0]["system response"]}<|im_end|>
<|im_start|>user
{log[1]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[1]["system response"]}<|im_end|>
...
```

**Processing**:
1. No system message - model learns helpful behavior from conversations
2. Format all turns as alternating user/assistant
3. Skip empty utterances/responses

---

### empathetic-dialogues
**Purpose**: Emotionally aware peer conversations + emotion recognition

**Relevant Fields**:
- `log`: List of turns with `user utterance`, `system response` (list[dict])
- `original dialog info`: JSON containing `context` (emotion like "excited", "sad", "guilty")

**Note**: Peer-to-peer conversations where both parties share experiences and relate to each other. Not traditional user↔assistant dynamic.

**Template**:
```
<|im_start|>user
{log[0]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[0]["system response"]}<|im_end|>
<|im_start|>user
{log[1]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[1]["system response"]}<|im_end|>
...

Question: How does the user feel in the above conversation?
Answer: {emotion}
```

**Processing**:
1. Parse `original dialog info` JSON to extract `context` (emotion)
2. Format conversation as user/assistant turns (skip empty utterances/responses)
3. Append emotion recognition Q&A outside chat format
4. Teaches both peer conversation skills and emotion identification

---

### samsum
**Purpose**: Dialogue summarization

**Relevant Fields**:
- `log`: List of turns with `user utterance`, `system response` (list[dict])
- `original dialog info`: JSON containing `summary`

**Note**: Peer conversations (like empathetic-dialogues) followed by summarization task.

**Template**:
```
<|im_start|>user
{log[0]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[0]["system response"]}<|im_end|>
<|im_start|>user
{log[1]["user utterance"]}<|im_end|>
<|im_start|>assistant
{log[1]["system response"]}<|im_end|>
...

Question: Summarize the above conversation.
Summary: {summary}
```

**Processing**:
1. Parse `original dialog info` JSON to extract `summary`
2. Format conversation as user/assistant turns (skip empty utterances/responses)
3. Append summarization Q&A outside chat format
4. Teaches both peer conversation and dialogue summarization

---

## 8. RAG Datasets

### rag-dataset-12000
**Purpose**: Context-grounded question answering

**Relevant Fields**:
- `context`: Source passage from Falcon RefinedWeb (string, variable length)
- `question`: Question about the context (string)
- `answer`: Answer derived from context (string)

**Usage**: Teaches reading comprehension and answer extraction from provided context. Clean RAG format.

**Template**:
```
### Context
{context}

### Question
{question}

### Answer
{answer}
```

**Processing**: None needed. Already clean format.

---

### ragbench-hotpotqa
**Purpose**: Multi-hop QA with multiple retrieved documents

**Relevant Fields**:
- `question`: Query requiring multi-hop reasoning (string)
- `documents`: List of retrieved passages (list[str])
- `response`: Generated answer (string)

**Usage**: Teaches reasoning across multiple documents and synthesizing information.

**Template**:
```
### Documents

{formatted_documents}

### Question
{question}

### Answer
{response}
```

**Processing**:
1. Format documents as numbered list or separate sections
2. Optionally include document headers (Document 1, Document 2, etc.)

**Example Document Formatting**:
```python
def format_documents(documents: list[str]) -> str:
    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"**Document {i}:**\n{doc}")
    return "\n\n".join(parts)
```

---

## Summary Table

| Dataset | Category | Transform Complexity | Key Fields |
|---------|----------|---------------------|------------|
| fineweb-edu | Grammar | None | text |
| wikipedia | Grammar | Minimal | title, text |
| simple-wikipedia | Grammar | Minimal | title, text |
| gsm8k | Reasoning | Minimal | question, answer |
| math | Reasoning | Minimal | problem, solution |
| openmath-instruct-1 | Reasoning | Filter+format | question, generated_solution, is_correct |
| openmath-instruct-2 | Reasoning | Minimal | problem, generated_solution |
| ecqa | Reasoning | Format choices | q_text, q_op*, q_ans, taskB |
| cosmopedia-wikihow | Procedural | None | text |
| stackexchange | Procedural | HTML strip, select best | question, answers |
| swe-bench | Debugging | Format diff | problem_statement, patch |
| code-contests | Debugging | Select solution | description, solutions |
| codesearchnet | Code | Minimal | func_code_string, func_documentation_string |
| toolace | Agentic | Format conversation | system, conversations |
| hermes-function-calling | Agentic | Format conversation | conversations |
| glaive-function-calling | Agentic | None/minimal | system, chat |
| generics-kb | Knowledge | Bundle facts | generic_sentence, term |
| openbookqa | Knowledge | Format choices | question_stem, choices, answerKey |
| soda | Dialogue | Parse context, format | log, original dialog info |
| multiwoz | Dialogue | Format turns | log |
| wizard-of-wikipedia | Dialogue | Extract passage | log, original dialog info |
| sharegpt | Dialogue | Format turns | log |
| empathetic-dialogues | Dialogue | Extract emotion | log, original dialog info |
| samsum | Dialogue | Include summary | log, original dialog info |
| rag-dataset-12000 | RAG | None | context, question, answer |
| ragbench-hotpotqa | RAG | Format documents | question, documents, response |

---

## Implementation Notes

### Transform Function Signature
```python
def transform_sample(sample: dict, dataset_name: str, config: dict) -> str:
    """
    Transform a raw sample into serialized text for training.

    :param sample: Raw sample dict from JSONL
    :param dataset_name: Name of the dataset (for selecting transform)
    :param config: Optional configuration (e.g., max_length, include_metadata)
    :return: Serialized text string
    """
```

### Registry Pattern
```python
TRANSFORMS = {
    "fineweb-edu": transform_fineweb_edu,
    "wikipedia": transform_wikipedia,
    "gsm8k": transform_gsm8k,
    # ... etc
}

def transform_sample(sample, dataset_name, config=None):
    transform_func = TRANSFORMS.get(dataset_name)
    if transform_func is None:
        raise ValueError(f"No transform for dataset: {dataset_name}")
    return transform_func(sample, config or {})
```

### DialogStudio Common Transform
```python
def transform_dialogstudio(sample: dict, config: dict) -> str:
    """Common transform for all DialogStudio datasets."""
    log = sample.get("log", [])

    turns = []
    for turn in log:
        user = turn.get("user utterance", "")
        system = turn.get("system response", "")
        if user:
            turns.append(f"User: {user}")
        if system:
            turns.append(f"Assistant: {system}")

    return "\n".join(turns)
```

### Quality Filtering
Some datasets benefit from filtering:
- `openmath-instruct-1`: Filter `is_correct=True`
- `generics-kb`: Filter `score > 0.5`
- `fineweb-edu`: Filter `int_score >= 3` for higher quality
- `stackexchange`: Filter `pm_score > 0` for positive-scored answers