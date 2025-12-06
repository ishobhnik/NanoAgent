# NanoAgent

NanoAgent is a lightweight, modular framework designed to evaluate the agentic and reasoning capabilities of Small Language Models (SLMs).

This project is developed under Salesforce by a team of [Rajdeep](https://github.com/RajdeepBakolia2004/), [Krishna](https://github.com/krishna16032005), and [Shobhnik](https://ishobhnik.github.io/) with the goal of designing and evaluating language models on industry-specific agentic tasks such as finance.

While massive models (LLMs) dominate general chat, NanoAgent focuses on benchmarking efficient, local models (like Google's Gemma-2B and Alibaba's Qwen-1.5B) on specific enterprise tasks: data engineering, file system manipulation, financial QA, and entity extraction.

## Key Features

- **Multi-Model Support**: Built to test models including `google/gemma-2b-it`, `Qwen/Qwen2.5-1.5B-Instruct`, and `gemini-2.5-flash`.
- **Agentic Capabilities**: Implements ReAct (Reasoning + Acting) for tool usage, file system planning, and python code execution.
- **Specialized NLP Benchmarks**: Includes dedicated pipelines for Financial QA (Exact Match scoring) and Named Entity Recognition (F1 scoring with token alignment).
- **Enterprise Evaluation**: Automated "Gold Standard" comparison and statistical reporting (Confidence Intervals, RSE).

## Project Structure

The repository is organized by task domain:

```
NanoAgent/
├── AngenticTasks/
│   ├── Agentic-Data-Processing/   # Task 1: Data Engineering Agents
│   │   ├── final_evaluate.py      # Main evaluation loop (Pandas/Python agents)
│   │   └── gold_standard.py       # Truth generation from CRM data
│   │   
│   ├── Agentic-File-Evaluator/    # Task 2: OS/File System Agents
│   │   ├── eval.py                # Local evaluation (Qwen)
│   │   └── Evaluate-Gemini.py     # Cloud evaluation (Gemini)
│   │   
│   ├── Infromation Extraction/    # Task 3: NER & Entity Extraction
│   │   ├── eval.py                # NER evaluation loop (SeqEval)
│   │   └── FINER.parquet          # Financial dataset
│   │
│   └── Question Answering/        # Task 4: Financial QA
│       ├── eval.py                # Direct QA evaluation (Exact Match)
│       └── ...
│
└── requirements.txt               # Project dependencies
```

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/ishobhnik/NanoAgent.git
cd NanoAgent
```

### 2. Set up environment:

```bash
conda create -n nanoagent python=3.10
conda activate nanoagent
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configuration:

Create a `.env` file if using Gemini:

```bash
GOOGLE_API_KEY=your_api_key_here
```

## Benchmark Tasks

### Track 1: Data Engineering Agent

**Goal**: The agent acts as a Data Engineer using Python/Pandas tools to query raw CRM CSVs.

- **Model**: `google/gemma-2b-it`
- **Metric**: Exact Match against `gold_standard.json`

```bash
python AngenticTasks/Agentic-Data-Processing/final_evaluate.py
```

### Track 2: File System Agent

**Goal**: The agent executes multi-step OS commands to create complex directory structures and config files.

- **Model**: `Qwen/Qwen2.5-1.5B` or `Gemini`
- **Metric**: File existence and path verification

```bash
python AngenticTasks/Agentic-File-Evaluator/eval.py
```

### Track 3: Information Extraction (NER)

**Goal**: Extracts specific entities (Labels) from financial text using the FINER dataset.

- **Model**: `Qwen/Qwen2.5-1.5B`
- **Technique**: Generates predictions and aligns sub-word tokens to labels
- **Metric**: Token-level F1 Score

```bash
python "AngenticTasks/Infromation Extraction/eval.py"
```

### Track 4: Financial Question Answering

**Goal**: Answers specific financial questions based on provided contexts (text + tables).

- **Model**: `google/gemma-2b-it` (Default)
- **Technique**: Zero-shot prompting with strict output formatting
- **Metric**: Exact Match (EM) accuracy

```bash
python "AngenticTasks/Question Answering/eval.py" --data "AngenticTasks/Infromation Extraction/FINER.parquet"
```

## Metrics & Reporting

The framework calculates rigorous statistics for agent tasks:

- **Pooled Accuracy**: Success rate across all trials
- **Standard Deviation & RSE**: Measures the stability/volatility of the model's performance
- **Confidence Intervals**: 95% certainty range for the results

## Hardware Requirements

- **GPU**: Recommended 6GB+ VRAM (NVIDIA) for running Qwen-1.5B and Gemma-2B locally
- **Storage**: Ensure space for huggingface model weights (~5-10GB)

## Contact
For any questions or issues, you are welcome to open an issue in this repo, or contact us at shobhnikk@iisc.ac.in, rajdeepbakol@iisc.ac.in and krishnaagarw@iisc.ac.in