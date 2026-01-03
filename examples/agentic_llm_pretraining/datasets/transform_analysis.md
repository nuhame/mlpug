# Dataset Transform Analysis for Agentic LLM Pretraining

This document analyzes how to transform each dataset into serialized text for next-token prediction (NTP) training.

## General Principles

1. **Plain text output**: All samples are transformed into plain text sequences
2. **Semantic formatting**: Use markdown headings only for actual hierarchy (e.g., document titles), use `Label: content` for flat Q&A/problem-solution structures
3. **Preserve reasoning structure**: Show step-by-step reasoning when available
4. **Consistent labels**: Use clear section labels (e.g., `Problem:`, `Solution:`, `Question:`, `Answer:`)
5. **No special tokens yet**: Templates are token-agnostic; EOS tokens added during tokenization

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
**Purpose**: Code documentation patterns

**Relevant Fields**:
- `func_name`: Function name (string)
- `func_code_string`: Full function code (string)
- `func_documentation_string`: Docstring (string)
- `language`: Programming language (string)

**Usage**: Teaches code-documentation alignment.

**Template**:
```
### Function
{func_name}

### Documentation
{func_documentation_string}

### Implementation
```{language}
{func_code_string}
```
```

**Processing**: None needed. Already clean.

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
{system}

User: {conversations[0].value}
Assistant: {conversations[1].value}
```

**Template** (multi-turn with tool results):
```
{system}

User: {conversations[0].value}
Assistant: {conversations[1].value}
Tool: {conversations[2].value}
Assistant: {conversations[3].value}
...
```

**Processing**:
1. Map `from` values to labels: user→User, assistant→Assistant, tool→Tool
2. Keep function call format (e.g., `[FunctionName(param=value)]`)

---

### hermes-function-calling
**Purpose**: Function calling examples

**Relevant Fields**:
- `conversations`: List with `from` (human/gpt/function_call/observation) and `value` (list[dict])
- `category`: Task category (string)
- `task`: Task description (string)

**Usage**: Multi-turn function calling with observations.

**Template**:
```
### Task: {task}

{formatted_conversation}
```

**Processing**:
1. Map `from` values to readable roles (Human, Assistant, Function, Result)
2. Format function calls clearly

---

### glaive-function-calling
**Purpose**: Function calling examples with system prompts

**Relevant Fields**:
- `system`: System prompt with function definitions (string)
- `chat`: Pre-formatted conversation (string)

**Usage**: Simple function calling patterns.

**Template**:
```
{system}

{chat}
```

**Processing**: Already formatted. Use as-is or clean SYSTEM:/USER:/ASSISTANT: prefixes.

---

## 6. Knowledge Datasets

### generics-kb
**Purpose**: Common sense facts

**Relevant Fields**:
- `generic_sentence`: Factual statement (string)
- `term`: Subject term (string)
- `score`: Confidence score (float)

**Usage**: Short factual statements. Bundle multiple for context.

**Template** (bundle 5-10 related facts):
```
### Facts about {common_term_or_topic}

- {generic_sentence_1}
- {generic_sentence_2}
- {generic_sentence_3}
...
```

**Processing**:
1. Group by `term` or topic
2. Filter by `score > 0.5` for quality
3. Bundle into paragraph-length sequences

---

### openbookqa
**Purpose**: Core science facts with reasoning

**Relevant Fields**:
- `question_stem`: Science question (string)
- `choices`: Answer options with `text` and `label` (dict)
- `answerKey`: Correct answer label (string)

**Usage**: Science reasoning in Q&A format.

**Template**:
```
### Question
{question_stem}

{formatted_choices}

### Answer
{answerKey}) {correct_choice_text}
```

**Processing**: Format choices as A) B) C) D). Include correct answer with text.

---

## 7. Dialogue Datasets (DialogStudio format)

All DialogStudio datasets share a common structure:

**Common Fields**:
- `log`: List of turns with `user utterance`, `system response`, `dialog history` (list[dict])
- `original dialog info`: JSON string with context/metadata (string)

### soda
**Purpose**: Social dialogues with emotional context

**Extra Context**: `original dialog info` contains `head`, `relation`, `tail` (social situation)

**Template**:
```
### Situation
{parsed_situation}

### Conversation
{formatted_dialogue}
```

**Processing**: Parse `original dialog info` for context. Format turns as Speaker A/B.

---

### multiwoz
**Purpose**: Task-oriented dialogues (booking, info requests)

**Extra Fields**:
- `dst`: Dialog state tracking
- `intent knowledge`: Possible intents

**Template**:
```
### Conversation

{formatted_dialogue}
```

**Processing**: Format as User/Assistant turns. Optionally include intent/state info.

---

### wizard-of-wikipedia
**Purpose**: Knowledge-grounded conversation

**Extra Context**: `original dialog info` contains `chosen_topic`, `chosen_topic_passage`

**Template**:
```
### Topic: {chosen_topic}

### Background
{chosen_topic_passage}

### Conversation
{formatted_dialogue}
```

**Processing**: Extract Wikipedia passage for context. Show grounding.

---

### sharegpt
**Purpose**: General user-assistant conversations

**Template**:
```
User: {turn_1_user}
Assistant: {turn_1_assistant}
User: {turn_2_user}
...
```

**Processing**: Simple alternating User/Assistant format.

---

### empathetic-dialogues
**Purpose**: Emotionally aware conversations

**Extra Context**: `original dialog info` contains `context` (emotion like "excited", "sad")

**Template**:
```
### Emotional Context: {emotion}

### Conversation
{formatted_dialogue}
```

**Processing**: Extract emotion label. Format as Speaker A/B or User/Assistant.

---

### samsum
**Purpose**: Dialogue with summarization

**Extra Context**: `original dialog info` contains `summary`

**Template**:
```
### Conversation
{formatted_dialogue}

### Summary
{summary}
```

**Processing**: Include summary as a generation target. Teaches summarization.

---

### coqa
**Purpose**: Conversational question answering

**Extra Context**: `original dialog info` contains `story` (the passage being discussed)

**Template**:
```
### Passage
{story}

### Questions and Answers
Q: {question_1}
A: {answer_1}

Q: {question_2}
A: {answer_2}
...
```

**Processing**: Extract passage from context. Format Q&A pairs sequentially.

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
| coqa | Dialogue | Extract passage | log, original dialog info |
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